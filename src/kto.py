import os
import json
import random
import argparse

import torch
from trl import KTOTrainer, KTOConfig
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, PeftModel
from accelerate import Accelerator, PartialState

from utils import gen_clarify_q_prompt

def preprocess(ex, mode, rank_threshold=1.5):
    """
    Convert examples to KTO format.
    KTO expects: prompt, completion, and label (True for desirable, False for undesirable)
    
    The rank in the data represents quality: lower rank = better quality
    rank 1 is best, rank 2 is worse
    """
    kto_data = []
    for clarification in ex['dpo']['clarifications']:
        if mode == 'gen_clarify_q':
            # Convert rank to binary label
            # Lower rank means higher quality (rank 1 = best)
            # So rank < threshold means desirable
            label = clarification['rank'] < rank_threshold
            
            kto_data.append({
                'prompt': gen_clarify_q_prompt(
                    qa_input=ex['question'],
                    clarify_q=None,
                ),
                'completion': clarification['question'] + '\n',
                'label': label,
            })
        else:
            raise ValueError(f"Invalid mode: {mode}")
    return kto_data


def main(args):
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum_steps,
        mixed_precision="fp16",  # Enable mixed precision training
        log_with="all"
    )

    if args.model == 'llama2':
        args.base_model = 'meta-llama/Llama-3.1-8B-Instruct'
    elif args.model == 'gemma':
        args.base_model = 'google/gemma-7b'
    else:
        raise ValueError('Invalid base_model')

    experiment_dir = os.path.join(
        args.output_dir,
        args.model,
        args.mode,
        args.experiment_name,
    )
    os.makedirs(experiment_dir, exist_ok=True)

    with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
        f.write(json.dumps(vars(args), indent=2))

    train_data = {}
    for path in args.train_paths:
        with open(path, 'r') as f:
            data = [json.loads(l) for l in f]
            train_data[os.path.split(path)[-1]] = [
                kto_ex for ex in data for kto_ex in preprocess(ex, mode=args.mode, rank_threshold=args.rank_threshold)
            ][:args.test]
    
    dev_data = {}
    for path in args.dev_paths:
        with open(path, 'r') as f:
            data = [json.loads(l) for l in f]
            dev_data[os.path.split(path)[-1]] = [
                kto_ex for ex in data for kto_ex in preprocess(ex, mode=args.mode, rank_threshold=args.rank_threshold)
            ][:args.test]

    # Save preprocessed data
    for name, data in train_data.items():
        with open(os.path.join(experiment_dir, 'train.kto.' + name), 'w') as f:
            for ex in data:
                f.write(json.dumps(ex) + '\n')
    
    for name, data in dev_data.items():
        with open(os.path.join(experiment_dir, 'dev.kto.' + name), 'w') as f:
            for ex in data:
                f.write(json.dumps(ex) + '\n')

    # Flatten data from all sources
    train_data = Dataset.from_list([ex for data in train_data.values() for ex in data])
    dev_data = Dataset.from_list([ex for data in dev_data.values() for ex in data])

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_bos_token = False

    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.float16
    )

        # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map={'': PartialState().process_index},
    )
    
    # Enable gradient checkpointing after model creation
    model.gradient_checkpointing_enable()
    
    # Set static graph for DDP compatibility
    if hasattr(model, '_set_static_graph'):
        model._set_static_graph()
    
    # Load and merge checkpoint if provided
    if args.checkpoint:
        model = PeftModel.from_pretrained(
            model,
            os.path.join(args.checkpoint),
        )
        model = model.merge_and_unload()

    # Configure LoRA
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        task_type="CAUSAL_LM",
    )

    # Calculate per-device batch size
    per_device_batch_size = max(1, args.batch_size // (torch.cuda.device_count() * args.grad_accum_steps))
    effective_batch_size = per_device_batch_size * torch.cuda.device_count() * args.grad_accum_steps
    
    print(f'Device Count={torch.cuda.device_count()}')
    print(f'Input Batch Size={args.batch_size}')
    print(f'Grad Accum Steps={args.grad_accum_steps}')
    print(f'Per Device Batch Size={per_device_batch_size}')
    print(f'Effective Total Batch Size={effective_batch_size}')

    # Configure KTO training
    training_args = KTOConfig(
        output_dir=experiment_dir,
        eval_strategy='steps',
        eval_steps=args.eval_steps,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.grad_accum_steps,
        save_steps=args.save_steps or args.eval_steps,
        save_total_limit=None,
        logging_steps=50,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        torch_compile=True,  # Enable torch compile for better performance
        # KTO-specific parameters
        beta=args.beta,
        desirable_weight=args.desirable_weight,
        undesirable_weight=args.undesirable_weight,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
    )

    # Initialize KTO trainer
    trainer = KTOTrainer(
        model=model,
        ref_model=None,  # KTO will create reference model automatically
        args=training_args,
        train_dataset=train_data,
        eval_dataset=dev_data,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # Train with optimized CUDA kernels
    with torch.backends.cuda.sdp_kernel(
        enable_flash=True,
        enable_math=False,
        enable_mem_efficient=False
    ):
        trainer.train(resume_from_checkpoint=args.resume_from)
    
    # Save training history and final model
    with open(os.path.join(experiment_dir, 'log_history.json'), 'w') as f:
        f.write(json.dumps(trainer.state.log_history, indent=2))
    
    trainer.model.save_pretrained(os.path.join(experiment_dir, 'best_checkpoint'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_paths', nargs='+', default=["/scratch/sujayb/anlp/clarifying_questions/data/splits/final_train.jsonl"])
    parser.add_argument('--dev_paths', nargs='+', default=["/scratch/sujayb/anlp/clarifying_questions/data/splits/final_dev.jsonl"])
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--resume_from', default=None)
    parser.add_argument('--random_seed', type=int, default=88888888)
    parser.add_argument(
        '--output_dir',
        default='../data/'
    )
    parser.add_argument('--test', type=int, default=None)

    parser.add_argument('--model', default='llama2')
    parser.add_argument('--mode', default="gen_clarify_q")
    parser.add_argument('--experiment_name', required=True)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--max_prompt_length', type=int, default=128)

    # KTO Training Hyperparameters
    parser.add_argument('--beta', type=float, default=0.1,
                        help='KTO loss coefficient')
    parser.add_argument('--desirable_weight', type=float, default=1.0,
                        help='Weight for desirable examples')
    parser.add_argument('--undesirable_weight', type=float, default=1.0,
                        help='Weight for undesirable examples')
    parser.add_argument('--rank_threshold', type=float, default=1.5,
                        help='Rank threshold for binary labeling (rank < threshold = desirable)')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--eval_steps', type=int, default=750)
    parser.add_argument('--save_steps', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=5e-7)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--grad_accum_steps', type=int, default=8)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.1)

    # LoRA Hyperparameters
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--lora_bias', default="none")

    cli_args = parser.parse_args()
    random.seed(cli_args.random_seed)
    torch.manual_seed(cli_args.random_seed)
    main(cli_args)