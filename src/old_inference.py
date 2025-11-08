import os
import json
import math
import random
import argparse
from collections import Counter
from tqdm import tqdm
from datasets import Dataset

import torch
from transformers import DataCollatorForLanguageModeling, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import Accelerator, PartialState

from hf_models import generate_and_score

from utils import batched, partitioned
from utils import gen_direct_qa_output_prompt, gen_clarify_q_prompt, gen_clarify_a_prompt, gen_qa_output_prompt

def get_response(model, tokenizer, base_model, data, temperature=None, n_samples=None, max_length=256):
    prompts = [gen_direct_qa_output_prompt(qa_input=ex['question'], qa_output=None) for ex in data]
    messages = [[{"role": "user", "content": p}] for p in prompts]

    input_ids = [tokenizer.apply_chat_template(
        message,
        add_generation_prompt=True,
        tokenize=False,
    ) for message in messages]
    inputs = tokenizer(input_ids, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(model.device)

    # GREEDY PRED
    predictions = generate_and_score(
        model,
        tokenizer,
        base_model,
        inputs,
        temperature=None,
        max_length=max_length,
    )
    if predictions:
        for ex, pred in zip(data, predictions):
            ex['pred']['response'] = pred['text'].strip()
    else:
        for ex in data:
            ex['pred']['response'] = None
    # SAMPLE PREDS
    for ex in data:
        ex['pred']['response_samples'] = []
    for _ in range(n_samples):
        predictions = generate_and_score(
            model,
            tokenizer,
            base_model,
            inputs,
            temperature=temperature,
            max_length=max_length
        )
        if predictions:
            for ex, pred in zip(data, predictions):
                ex['pred']['response_samples'].append(pred['text'].strip())


def get_clarify_question(model, tokenizer, base_model, data, temperature=None, n_samples=None, max_length=256):
    prompts = [gen_direct_qa_output_prompt(qa_input=ex['question']) for ex in data]
    messages = [[
        {"role": "system", "content": "You are an expert at asking clarifying questions to better understand user queries. Please answer with only the clarifying question and nothing else."},
        {"role": "user", "content": p}
    ] for p in prompts]

    input_ids = [tokenizer.apply_chat_template(
        message,
        add_generation_prompt=True,
        tokenize=False,
    ) for message in messages]
    inputs = tokenizer(input_ids, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(model.device)

    for ex in data:
        ex['pred']['clarification'] = {
            'question': None,
            'answers': [],
        }

    # GREEDY PRED
    predictions = generate_and_score(
        model,
        tokenizer,
        base_model,
        inputs,
        temperature=None,
        max_length=max_length,
    )

    if predictions:
        for ex, pred in zip(data, predictions):
            ex['pred']['clarification']['question'] = pred['text']

    # SAMPLE PREDS
    for ex in data:
        ex['pred']['clarification_samples'] = []
    for _ in range(n_samples):
        predictions = generate_and_score(
            model,
            tokenizer,
            base_model,
            inputs,
            temperature=temperature,
            max_length=max_length
        )
        if predictions:
            for ex, pred in zip(data, predictions):
                question = pred['text']
                if not question:
                    continue
                if any(question == s['question'] for s in ex['pred']['clarification_samples']):
                    continue
                if question == ex['pred']['clarification']['question']:
                    continue
                ex['pred']['clarification_samples'].append({
                    'question': question,
                    'answers': [],
                })

def get_clarify_answers(
    model,
    tokenizer,
    base_model,
    qa_io_clarifications,
    max_length=256,
):
    # GREEDY PRED
    sys_prompt = """
You are simulating a human user who asked an ambiguous question and is now responding to a clarifying question.

CONTEXT:
- Original Question: {qa_input}
- Gold Response: {qa_output}

TASK:
Generate a natural human response to the clarifying question that:
1. Clarifies the intent behind the original question based on the gold response
2. Sounds like a real user clarifying their intent (not robotic or overly formal)

GUIDELINES:
- DO NOT REVEAL ANY INFORMATION IN THE GOLD RESPONSE TO THE USER!
- Keep it concise (1 sentence typically)
- Don't over-explain
- The response should clearly resolve the ambiguity
- Vary your phrasing across different examples

Generate only the simulated user's response, nothing else.
"""

    messages = [[
        {"role": "system", "content": sys_prompt.format(qa_input=qa_input, qa_output=qa_output, clarification=clarification)},
        {"role": "user", "content": clarification}
    ] for qa_input, qa_output, clarification in qa_io_clarifications
]

    input_ids = [tokenizer.apply_chat_template(
        message,
        add_generation_prompt=True,
        tokenize=False,
    ) for message in messages]
    inputs = tokenizer(input_ids, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(model.device)

    predictions = generate_and_score(
        model,
        tokenizer,
        base_model,
        inputs,
        temperature=None,
        max_length=max_length,
    )

    if predictions:
        for (_, qa_output, clarification), pred in zip(qa_io_clarifications, predictions):
            clarify_a = pred['text']
            if not clarify_a:
                continue
            clarification['answers'].append({
                'answer': clarify_a,
                'response': None,
                'gold_response': qa_output,
            })


def get_qa_outputs(model, tokenizer, base_model, ex_clarification_answers, max_length=256):
    prompts = [
        gen_qa_output_prompt(
            qa_input=ex['question'],
            clarify_q=clarification['question'],
            clarify_a=answer['answer'],
            qa_output=None
        )
        for ex, clarification, answer in ex_clarification_answers
    ]

    sys_prompt = "You are an expert at answering questions based on clarifying questions and answers. Given the question, the clarifying question asked to the user, and the user's clarifying answer, provide a concise and accurate answer to the original question. Please only output the final answer."

    messages = [[
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": ex['question']},
        {"role": "assistant", "content": clarification['question']},
        {"role": "user", "content": answer['answer']},
    ] for ex, clarification, answer in ex_clarification_answers
]
    print(messages[0])
    input_ids = [tokenizer.apply_chat_template(
        message,
        add_generation_prompt=True,
        tokenize=False,
    ) for message in messages]
    inputs = tokenizer(input_ids, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(model.device)

    predictions = generate_and_score(
        model,
        tokenizer,
        base_model,
        inputs,
        temperature=None,
        max_length=max_length,
    )

    if predictions:
        for (_, _, answer), pred in zip(ex_clarification_answers, predictions):
            answer['response'] = pred['text']


def main(args):
    accelerator = Accelerator()
    distributed_state = PartialState()

    # OUTPUT PATH SETUP
    output_name = os.path.split(args.dataset_path)[-1][:-len('.jsonl')]
    if args.output_name:
        output_name += f'.{args.output_name}'
    if args.mode == 'respond':
        output_name += '.respond_s{}'.format(args.n_samples)
    if args.mode == 'clarify_q':
        output_name += '.clarify_q_s{}'.format(args.n_samples)
    if args.mode == 'clarify_a':
        output_name += '.clarify_a'
    if args.mode == 'qa_output':
        output_name += '.qa_output'
    if args.mode == 'eval_qa_output':
        output_name += '.eval_qa_output'
    if args.shard_total:
        output_name += f'.{args.shard_idx}of{args.shard_total}'
    output_name += '.jsonl'
    
    # Determine output directory
    output_dir = args.checkpoint if args.checkpoint else os.path.dirname(args.dataset_path)
    output_path = os.path.join(output_dir, output_name)
    print(f"Output will be saved to: {output_path}")
    

    # LOADING DATA
    with open(args.dataset_path, 'r') as f:
        data = [json.loads(l) for l in f][:args.test]
    if args.shard_total:
        data = partitioned(data, args.shard_total)[args.shard_idx-1]

    data = partitioned(
        data,
        distributed_state.num_processes
    )[distributed_state.process_index]

    # LOADING MODEL AND TOKENIZER
    if args.model == 'llama3':
        args.base_model = 'meta-llama/Llama-3.2-3B-Instruct'
    elif args.model == 'gemma':
        args.base_model = 'google/gemma-7b'
    elif args.model == 'llama3-ft':
        args.base_model = '/scratch/sujayb/anlp/clarifying_questions/final_fix/final_kto_model_merged'
    else:
        raise ValueError('Invalid base_model')

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, dtype=torch.float32)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )
    
    # Determine the device
    device = f'cuda:{distributed_state.process_index}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(distributed_state.process_index)}")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        # quantization_config=bnb_config if torch.cuda.is_available() else None,
        # device_map={'': distributed_state.process_index} if torch.cuda.is_available() else None,
        device_map="auto" if torch.cuda.is_available() else None,
        dtype=torch.float32,
    )
    
    # If CUDA is not available or device_map didn't work, manually move to device
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU")
    else:
        print(f"Model loaded on GPU: {next(model.parameters()).device}")     
    
    for ex in data:
        if 'pred' not in ex:
            ex['pred'] = {}

    if args.mode == 'respond':
        all_batched_data = batched(data, args.batch_size)
        for batch_data in tqdm(
            all_batched_data,
            desc=f'Respond Predictions ({distributed_state.process_index})',
            position=distributed_state.process_index,
            total=len(all_batched_data)
        ):
            get_response(
                model,
                tokenizer,
                args.model,
                batch_data,
                n_samples=args.n_samples,
                temperature=args.temperature,
                max_length=args.max_length,
            )
    elif args.mode == 'clarify_q':
        all_batched_data = batched(data, args.batch_size)
        for batch_data in tqdm(
            all_batched_data,
            desc=f'Clarify Q Predictions ({distributed_state.process_index})',
            position=distributed_state.process_index,
            total=len(all_batched_data)
        ):
            get_clarify_question(
                model,
                tokenizer,
                args.model,
                batch_data,
                n_samples=args.n_samples,
                temperature=args.temperature,
                max_length=args.max_length,
            )
    elif args.mode == 'clarify_a':
        all_qa_io_clarifications = []
        for ex in data:
            if 'isambig' in ex and ex['isambig']:
                answers = set(ex['answers'])
            else:
                answers = set(ex['nq_answers'])
            for qa_output in answers:
                if ex['pred']['clarification']:
                    all_qa_io_clarifications.append(
                        (ex['question'], qa_output, ex['pred']['clarification'])
                    )
                for clarification in ex['pred']['clarification_samples']:
                    all_qa_io_clarifications.append(
                        (ex['question'], qa_output, clarification)
                    )
        all_batched_qa_io_clarifications = batched(all_qa_io_clarifications, args.batch_size)
        for batch_qa_io_clarifications in tqdm(
            all_batched_qa_io_clarifications,
            desc=f'Clarify A Predictions ({distributed_state.process_index})',
            position=distributed_state.process_index,
            total=len(all_batched_qa_io_clarifications)
        ):
            get_clarify_answers(
                model,
                tokenizer,
                args.model,
                batch_qa_io_clarifications,
                max_length=args.max_length,
            )
    elif args.mode == 'qa_output':
        all_ex_clarification_answers = []
        for ex in data:
            if ex['pred']['clarification']:
                for answer in ex['pred']['clarification']['answers']:
                    all_ex_clarification_answers.append((ex, ex['pred']['clarification'], answer))
            for clarification in ex['pred']['clarification_samples']:
                for answer in clarification['answers']:
                    all_ex_clarification_answers.append((ex, clarification, answer))
        all_batched_ex_clarification_answers = batched(all_ex_clarification_answers, args.batch_size)
        for batch_ex_clarification_answers in tqdm(
            all_batched_ex_clarification_answers,
            desc=f'QA Output Predictions ({distributed_state.process_index})',
            position=distributed_state.process_index,
            total=len(all_batched_ex_clarification_answers)
        ):
            get_qa_outputs(
                model,
                tokenizer,
                args.model,
                batch_ex_clarification_answers,
                max_length=args.max_length,
            )
    elif args.mode == 'eval_qa_output':
        all_ex_clarification_answers = []
        for ex in data:
            if ex['pred']['clarification']:
                for answer in ex['pred']['clarification']['eval_answers']:
                    if answer['answer']:
                        all_ex_clarification_answers.append((ex, ex['pred']['clarification'], answer))
            for clarification in ex['pred']['clarification_samples']:
                for answer in clarification['eval_answers']:
                    if answer['answer']:
                        all_ex_clarification_answers.append((ex, clarification, answer))
        all_batched_ex_clarification_answers = batched(all_ex_clarification_answers, args.batch_size)
        for batch_ex_clarification_answers in tqdm(
            all_batched_ex_clarification_answers,
            desc=f'Eval QA Output Predictions ({distributed_state.process_index})',
            position=distributed_state.process_index,
            total=len(all_batched_ex_clarification_answers)
        ):
            get_qa_outputs(
                model,
                tokenizer,
                args.model,
                batch_ex_clarification_answers,
                max_length=args.max_length,
            )

    # Setup temporary output dir and clear anything from past runs
    tmp_output_dir = os.path.join(output_dir, 'tmp')
    os.makedirs(tmp_output_dir, exist_ok=True)
    tmp_output_path = os.path.join(
        tmp_output_dir,
        f'{distributed_state.process_index}_{output_name}'
    )

    with open(tmp_output_path, 'w') as f:
        for ex in data:
            f.write(json.dumps(ex) + '\n')

    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        all_outputs = []
        for process_idx in range(accelerator.num_processes):
            tmp_output_path = os.path.join(
                tmp_output_dir,
                f'{process_idx}_{output_name}'
            )
            with open(tmp_output_path, 'r') as f:
                all_outputs.extend(json.loads(l) for l in f)

        with open(output_path, 'w') as f:
            for ex in all_outputs:
                f.write(json.dumps(ex) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # DATA PATHING
    parser.add_argument('--dataset_path', type=str, default="/scratch/sujayb/anlp/clarifying_questions/data/ambigqa.dev_4h.clarify.jsonl")

    # EVAL SETUP
    parser.add_argument('--mode', default='clarify_q')
    parser.add_argument('--n_samples', type=int, default=0)
    parser.add_argument('--output_name', default=None)
    parser.add_argument('--temperature', type=int, default=1.0)
    parser.add_argument('--shard_idx', type=int, default=None)
    parser.add_argument('--shard_total', type=int, default=None)

    # MODEL SETUP
    parser.add_argument('--model', default='llama3-ft')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--merge_checkpoint', default=None)
    parser.add_argument('--merge_checkpoint_2', default=None)
    parser.add_argument('--adapter', default=None)
    parser.add_argument('--task', type=int, default=None)

    # EVAL HYPERPARAMETERS
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for inference (increase for better GPU utilization)')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum generation length')
    parser.add_argument('--test', type=int, default=None)

    # MISC
    parser.add_argument('--random_seed', type=int, default=88888888)

    cli_args = parser.parse_args()
    random.seed(cli_args.random_seed)

    with torch.no_grad():
        main(cli_args)
