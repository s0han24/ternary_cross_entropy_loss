"""
HuggingFace model utilities for text generation and scoring.
"""

import torch
from typing import Optional, List, Dict, Any


def generate_and_score(
    model,
    tokenizer,
    base_model: str,
    input_ids,
    temperature: Optional[float] = None,
    max_length: int = 256,
) -> List[Dict[str, Any]]:
    """
    Generate text completions from a HuggingFace causal language model.
    
    Args:
        model: HuggingFace AutoModelForCausalLM instance
        tokenizer: HuggingFace tokenizer
        base_model: Base model identifier (e.g., 'llama3', 'gemma')
        input_ids: Input token IDs tensor of shape (batch_size, seq_len)
        temperature: Sampling temperature. If None, uses greedy decoding.
        max_length: Maximum length of generated sequence
        
    Returns:
        List of dictionaries with 'text' key containing decoded predictions.
        Returns empty list if generation fails.
    """
    try:
        # Ensure input_ids are on the same device as the model
        # Prepare generation kwargs
        generation_kwargs = {
            'max_new_tokens': 128,  # Use max_new_tokens instead of max_length for efficiency
            'pad_token_id': tokenizer.eos_token_id,
            'eos_token_id': tokenizer.eos_token_id,
            'use_cache': True,  # Enable KV cache for faster generation
        }
        
        # Configure sampling vs greedy decoding
        if temperature is None or temperature == 0:
            # Greedy decoding
            generation_kwargs['do_sample'] = False
        else:
            # Sampling with temperature
            generation_kwargs['do_sample'] = True
            generation_kwargs['temperature'] = temperature
            generation_kwargs['top_p'] = 0.9  # nucleus sampling
            generation_kwargs['top_k'] = 50
        
        # Generate with batched processing
        with torch.no_grad():
            # Extract tensor if input_ids is a tokenizer output
            input_tensor = input_ids.input_ids if hasattr(input_ids, 'input_ids') else input_ids
            outputs = model.generate(input_tensor, **generation_kwargs)
        
        # Get the length of input to extract only new tokens (assistant's response)
        input_length = input_tensor.shape[1]
        
        # Extract only the newly generated tokens (assistant's output)
        new_tokens = outputs[:, input_length:]
        
        # Batch decode only the new tokens for efficiency (excludes system prompt and user input)
        decoded_texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        
        
        # Build predictions list
        predictions = []
        for i, decoded_text in enumerate(decoded_texts):
            predictions.append({
                'text': decoded_text,
                'tokens': outputs[i].tolist(),
            })
        
        return predictions
        
    except Exception as e:
        print(f"Error in generate_and_score: {e}")
        import traceback
        traceback.print_exc()
        return []
