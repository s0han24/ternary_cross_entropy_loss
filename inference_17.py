import os
import json
import argparse
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator, PartialState

from utils import batched, partitioned
from utils import gen_direct_qa_output_prompt, gen_clarify_q_prompt, gen_clarify_a_prompt, gen_qa_output_prompt


def generate_batch(model, tokenizer, inputs, max_new_tokens=32, temperature=None):
    """Generate text for a batch of inputs."""
    with torch.no_grad():
        if temperature is None or temperature == 0:
            # Greedy decoding
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        else:
            # Sampling with temperature
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
            )
    
    # Decode outputs
    predictions = []
    for output in outputs:
        # Remove input tokens from output
        generated_tokens = output[inputs['input_ids'].shape[1]:]
        text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        predictions.append({'text': text.strip()})
    
    return predictions


def get_response(model, tokenizer, data, temperature=None, n_samples=None, max_new_tokens=256):
    """Get direct responses to questions."""
    prompts = [gen_direct_qa_output_prompt(qa_input=ex['question'], qa_output=None) for ex in data]
    
    messages = [[{"role": "user", "content": p}] for p in prompts]
    
    input_texts = [tokenizer.apply_chat_template(
        message,
        add_generation_prompt=True,
        tokenize=False,
    ) for message in messages]

    inputs = tokenizer(
        input_texts, 
        return_tensors='pt', 
        padding=True, 
        truncation=True,
    ).to(model.device)
    
    # GREEDY PREDICTION
    predictions = generate_batch(
        model,
        tokenizer,
        inputs,
        max_new_tokens=max_new_tokens,
        temperature=None,
    )
    
    if predictions:
        for ex, pred in zip(data, predictions):
            ex['pred']['response'] = pred['text']
    else:
        for ex in data:
            ex['pred']['response'] = None
    
    # SAMPLED PREDICTIONS
    for ex in data:
        ex['pred']['response_samples'] = []
    
    for _ in range(n_samples):
        predictions = generate_batch(
            model,
            tokenizer,
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        if predictions:
            for ex, pred in zip(data, predictions):
                ex['pred']['response_samples'].append(pred['text'])


def get_clarify_question(model, tokenizer, data, temperature=None, n_samples=None, max_new_tokens=256):
    """Generate clarifying questions."""
    prompts = [gen_clarify_q_prompt(qa_input=ex['question'], clarify_q=None) for ex in data]
    
    messages = [[
        {"role": "system", "content": "You are an expert at asking clarifying questions to better understand user queries. Please answer with only the clarifying question and nothing else."},
        {"role": "user", "content": p}
    ] for p in prompts]
    
    input_texts = [tokenizer.apply_chat_template(
        message,
        add_generation_prompt=True,
        tokenize=False,
    ) for message in messages]
    
    inputs = tokenizer(
        input_texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
    ).to(model.device)
    
    # Initialize clarification structure
    for ex in data:
        ex['pred']['clarification'] = {
            'question': None,
            'answers': [],
        }
    
    # GREEDY PREDICTION
    predictions = generate_batch(
        model,
        tokenizer,
        inputs,
        max_new_tokens=max_new_tokens,
        temperature=None,
    )
    
    if predictions:
        for ex, pred in zip(data, predictions):
            ex['pred']['clarification']['question'] = pred['text']
    
    # SAMPLED PREDICTIONS
    for ex in data:
        ex['pred']['clarification_samples'] = []
    
    for _ in range(n_samples):
        predictions = generate_batch(
            model,
            tokenizer,
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        if predictions:
            for ex, pred in zip(data, predictions):
                question = pred['text']
                if not question:
                    continue
                # Avoid duplicates
                if any(question == s['question'] for s in ex['pred']['clarification_samples']):
                    continue
                if question == ex['pred']['clarification']['question']:
                    continue
                ex['pred']['clarification_samples'].append({
                    'question': question,
                    'answers': [],
                })


def get_clarify_answers(model, tokenizer, qa_io_clarifications, max_new_tokens=256):
    """Generate answers to clarifying questions."""
    sys_prompt = """You are simulating a human user who asked an ambiguous question and is now responding to a clarifying question.

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

Generate only the simulated user's response, nothing else."""
    
    messages = [[
        {"role": "system", "content": sys_prompt.format(qa_input=qa_input, qa_output=qa_output)},
        {"role": "user", "content": clarification['question']}
    ] for qa_input, qa_output, clarification in qa_io_clarifications]
    
    input_texts = [tokenizer.apply_chat_template(
        message,
        add_generation_prompt=True,
        tokenize=False,
    ) for message in messages]
    
    inputs = tokenizer(
        input_texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
    ).to(model.device)
    
    predictions = generate_batch(
        model,
        tokenizer,
        inputs,
        max_new_tokens=max_new_tokens,
        temperature=None,
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


def get_qa_outputs(model, tokenizer, ex_clarification_answers, max_new_tokens=256):
    """Generate final answers based on clarifying Q&A."""
    sys_prompt = "You are an expert at answering questions based on clarifying questions and answers. Given the question, the clarifying question asked to the user, and the user's clarifying answer, provide a concise and accurate answer to the original question. Please only output the final answer."
    
    messages = [[
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": ex['question']},
        {"role": "assistant", "content": clarification['question']},
        {"role": "user", "content": answer['answer']},
    ] for ex, clarification, answer in ex_clarification_answers]
    
    input_texts = [tokenizer.apply_chat_template(
        message,
        add_generation_prompt=True,
        tokenize=False,
    ) for message in messages]
    
    inputs = tokenizer(
        input_texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
    ).to(model.device)
    
    predictions = generate_batch(
        model,
        tokenizer,
        inputs,
        max_new_tokens=max_new_tokens,
        temperature=None,
    )
    
    if predictions:
        for (_, _, answer), pred in zip(ex_clarification_answers, predictions):
            answer['response'] = pred['text']


def main(args):
    accelerator = Accelerator()
    distributed_state = PartialState()
    
    # OUTPUT PATH SETUP
    output_name = os.path.splitext(os.path.basename(args.dataset_path))[0]
    if args.output_name:
        output_name += f'.{args.output_name}'
    if args.mode == 'respond':
        output_name += f'.respond_s{args.n_samples}'
    elif args.mode == 'clarify_q':
        output_name += f'.clarify_q_s{args.n_samples}'
    elif args.mode == 'clarify_a':
        output_name += '.clarify_a'
    elif args.mode == 'qa_output':
        output_name += '.qa_output'
    elif args.mode == 'eval_qa_output':
        output_name += '.eval_qa_output'
    
    if args.shard_total:
        output_name += f'.{args.shard_idx}of{args.shard_total}'
    output_name += '.jsonl'
    
    output_dir = args.checkpoint if args.checkpoint else os.path.dirname(args.dataset_path)
    output_path = os.path.join(output_dir, output_name)
    print(f"Output will be saved to: {output_path}")
    
    # LOADING DATA
    with open(args.dataset_path, 'r') as f:
        data = [json.loads(l) for l in f]
    
    if args.test:
        data = data[:args.test]
    
    if args.shard_total:
        data = partitioned(data, args.shard_total)[args.shard_idx - 1]
    
    data = partitioned(
        data,
        distributed_state.num_processes
    )[distributed_state.process_index]
    
    # LOADING MODEL AND TOKENIZER
    print(f"Loading model from: {args.model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = f'cuda:{distributed_state.process_index}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model.eval()
    
    print(f"Model loaded on: {next(model.parameters()).device}")
    
    # Initialize pred dict
    for ex in data:
        if 'pred' not in ex:
            ex['pred'] = {}
    
    # RUN INFERENCE BASED ON MODE
    if args.mode == 'respond':
        all_batched_data = list(batched(data, args.batch_size))
        for batch_data in tqdm(
            all_batched_data,
            desc=f'Respond Predictions ({distributed_state.process_index})',
            position=distributed_state.process_index,
        ):
            get_response(
                model,
                tokenizer,
                batch_data,
                n_samples=args.n_samples,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
            )
    
    elif args.mode == 'clarify_q':
        all_batched_data = list(batched(data, args.batch_size))
        for batch_data in tqdm(
            all_batched_data,
            desc=f'Clarify Q Predictions ({distributed_state.process_index})',
            position=distributed_state.process_index,
        ):
            get_clarify_question(
                model,
                tokenizer,
                batch_data,
                n_samples=args.n_samples,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
            )
    
    elif args.mode == 'clarify_a':
        all_qa_io_clarifications = []
        for ex in data:
            if 'isambig' in ex and ex['isambig']:
                answers = set(ex['answers'])
            else:
                answers = set(ex['nq_answers'])
            
            for qa_output in answers:
                if ex['pred'].get('clarification'):
                    all_qa_io_clarifications.append(
                        (ex['question'], qa_output, ex['pred']['clarification'])
                    )
                for clarification in ex['pred'].get('clarification_samples', []):
                    all_qa_io_clarifications.append(
                        (ex['question'], qa_output, clarification)
                    )
        
        all_batched = list(batched(all_qa_io_clarifications, args.batch_size))
        for batch_qa_io in tqdm(
            all_batched,
            desc=f'Clarify A Predictions ({distributed_state.process_index})',
            position=distributed_state.process_index,
        ):
            get_clarify_answers(
                model,
                tokenizer,
                batch_qa_io,
                max_new_tokens=args.max_new_tokens,
            )
    
    elif args.mode == 'qa_output':
        all_ex_clarification_answers = []
        for ex in data:
            if ex['pred'].get('clarification'):
                for answer in ex['pred']['clarification'].get('answers', []):
                    all_ex_clarification_answers.append((ex, ex['pred']['clarification'], answer))
            for clarification in ex['pred'].get('clarification_samples', []):
                for answer in clarification.get('answers', []):
                    all_ex_clarification_answers.append((ex, clarification, answer))
        
        all_batched = list(batched(all_ex_clarification_answers, args.batch_size))
        for batch_ex_clarif in tqdm(
            all_batched,
            desc=f'QA Output Predictions ({distributed_state.process_index})',
            position=distributed_state.process_index,
        ):
            get_qa_outputs(
                model,
                tokenizer,
                batch_ex_clarif,
                max_new_tokens=args.max_new_tokens,
            )
    
    elif args.mode == 'eval_qa_output':
        all_ex_clarification_answers = []
        for ex in data:
            if ex['pred'].get('clarification'):
                for answer in ex['pred']['clarification'].get('eval_answers', []):
                    if answer.get('answer'):
                        all_ex_clarification_answers.append((ex, ex['pred']['clarification'], answer))
            for clarification in ex['pred'].get('clarification_samples', []):
                for answer in clarification.get('eval_answers', []):
                    if answer.get('answer'):
                        all_ex_clarification_answers.append((ex, clarification, answer))
        
        all_batched = list(batched(all_ex_clarification_answers, args.batch_size))
        for batch_ex_clarif in tqdm(
            all_batched,
            desc=f'Eval QA Output Predictions ({distributed_state.process_index})',
            position=distributed_state.process_index,
        ):
            get_qa_outputs(
                model,
                tokenizer,
                batch_ex_clarif,
                max_new_tokens=args.max_new_tokens,
            )
    
    # SAVE RESULTS
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
            tmp_path = os.path.join(
                tmp_output_dir,
                f'{process_idx}_{output_name}'
            )
            with open(tmp_path, 'r') as f:
                all_outputs.extend(json.loads(l) for l in f)
        
        with open(output_path, 'w') as f:
            for ex in all_outputs:
                f.write(json.dumps(ex) + '\n')
        
        print(f"Final output saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # DATA PATHING
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True, help='Path to model (base model or merged adapter)')
    
    # EVAL SETUP
    parser.add_argument('--mode', default='clarify_q', choices=['respond', 'clarify_q', 'clarify_a', 'qa_output', 'eval_qa_output'])
    parser.add_argument('--n_samples', type=int, default=0)
    parser.add_argument('--output_name', default=None)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--shard_idx', type=int, default=None)
    parser.add_argument('--shard_total', type=int, default=None)
    parser.add_argument('--checkpoint', default=None)
    
    # EVAL HYPERPARAMETERS
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--test', type=int, default=None)
    
    # MISC
    parser.add_argument('--random_seed', type=int, default=88888888)
    
    args = parser.parse_args()
    
    import random
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    
    with torch.no_grad():
        main(args)