# call_seed_tot.py

import os, sys, hydra
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'models/SEED'))
from environment import SEED_PROJECT_ROOT
import torch
from helper import set_seed, get_ft_path
from PIL import Image
from time import time
from configs import instruction_dict
from torch.utils.data import Dataset
from omegaconf import OmegaConf
from peft import PeftModel
import numpy as np


image_placeholder = "[IMG]" + "<image>" * 32 + "[/IMG]"
def get_generation_config(tokenizer, gen_mode='text'):
    if gen_mode == 'image':
        return {
            'temperature': 1.0,
            'top_p': 0.9,
            'do_sample': True,
            'max_new_tokens': 1024,
            'eos_token_id': tokenizer.eos_token_id,
            'pad_token_id': tokenizer.pad_token_id,
            'bos_token_id': tokenizer.bos_token_id,
        }
    else:
        return {
            'temperature': 0.7,         
            'top_p': 0.9,               
            'num_beams': 1,
            'do_sample': True,         
            'max_new_tokens': 160,      
            'eos_token_id': tokenizer.eos_token_id,
            'pad_token_id': tokenizer.pad_token_id,
            'bos_token_id': tokenizer.bos_token_id,
        }



s_token = "[INST] "
e_token = " [/INST]"
sep = "\n"

BOI_TOKEN = '<img>'
EOI_TOKEN = '</img>'
IMG_TOKEN = '<img_{:05d}>'

IMG_FLAG = '<image>'
NUM_IMG_TOKNES = 32
NUM_IMG_CODES = 8192
image_id_shift = 32000

def generate(tokenizer, input_tokens, generation_config, model):
    input_ids = tokenizer(input_tokens, add_special_tokens=False, return_tensors='pt').input_ids
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    generate_ids = model.generate(input_ids=input_ids, **generation_config)
    generate_ids = generate_ids[0][input_ids.shape[1]:]
    print("[DEBUG] Decoded output:\n", tokenizer.decode(generate_ids))
    print("[DEBUG] Output token IDs:", generate_ids.tolist())
    return generate_ids



import re
import torch


def extract_image_ids_from_ids(generate_ids, image_id_shift):
    try:
        if generate_ids is None:
            return None
            
        if torch.is_tensor(generate_ids):
            ids = generate_ids.detach().cpu().tolist()
        else:
            ids = list(generate_ids)
        
        img_ids = []
        for token_id in ids:
            if token_id >= image_id_shift:
                img_id = token_id - image_id_shift
                if 0 <= img_id < 8192:  
                    img_ids.append(img_id)
                else:
                    print(f"[WARNING] Invalid image token ID: {img_id} (after shift)")
        
        if not img_ids:
            print(f"[DEBUG] No valid image tokens found")
            print(f"[DEBUG] First 20 token IDs: {ids[:20]}")
            return None
        
        print(f"[DEBUG] Found {len(img_ids)} valid image tokens")
        print(f"[DEBUG] First 10 image token IDs: {img_ids[:10]}")
        
        if min(img_ids) < 0 or max(img_ids) >= 8192:
            print(f"[ERROR] Image token IDs out of range: min={min(img_ids)}, max={max(img_ids)}")
            img_ids = [id for id in img_ids if 0 <= id < 8192]
            print(f"[DEBUG] After filtering: {len(img_ids)} tokens")
        
        if not img_ids:
            return None
            
        return torch.tensor([img_ids], dtype=torch.long)
        
    except Exception as e:
        print(f"[ERROR] Error extracting image IDs: {e}")
        import traceback
        traceback.print_exc()
        return None


def decode_image_text(generate_ids, tokenizer, image_id_shift):
    try:
        full_text = tokenizer.decode(generate_ids, skip_special_tokens=False)
        print(f"[DEBUG] Decoded text (first 200 chars): {full_text[:200]}")
        
        image_ids_tensor = extract_image_ids_from_ids(generate_ids, image_id_shift)
        
        image = None
        if image_ids_tensor is not None:
            try:
                print(f"[DEBUG] Image tensor shape: {image_ids_tensor.shape}")
                print(f"[DEBUG] Image tensor dtype: {image_ids_tensor.dtype}")
                
                if image_ids_tensor.numel() == 0:
                    print("[WARNING] Empty image tensor")
                    return full_text, None
                
                device = next(tokenizer.parameters()).device if hasattr(tokenizer, 'parameters') else 'cpu'
                if device != image_ids_tensor.device:
                    image_ids_tensor = image_ids_tensor.to(device)
                
                if len(image_ids_tensor.shape) == 1:
                    image_ids_tensor = image_ids_tensor.unsqueeze(0)
                
                print(f"[DEBUG] Final tensor shape for decode: {image_ids_tensor.shape}")
                
                with torch.no_grad():
                    if hasattr(tokenizer, 'decode_image'):
                        image = tokenizer.decode_image(image_ids_tensor)
                        if isinstance(image, list) and len(image) > 0:
                            image = image[0]
                        print("[DEBUG] Image decoded successfully")
                    else:
                        print("[ERROR] Tokenizer has no decode_image method")
                        return full_text, None
                        
            except Exception as decode_error:
                print(f"[ERROR] Image decode error: {decode_error}")
                import traceback
                traceback.print_exc()
                image = None
        
        return full_text, image
        
    except Exception as e:
        print(f"[ERROR] Error in decode_image_text: {e}")
        import traceback
        traceback.print_exc()
        return "", None


def load_seed(
    device = 'cuda',
    seed = 123,
    finetuned = False,
    shot = 2,
    gen_mode = 'image',
    prompt_type = 'default',
    ft_mode = 'all',
    eval_task_theme = '',
):
    set_seed(seed)
    os.environ["PROJECT_ROOT"] = SEED_PROJECT_ROOT
    
    tokenizer_cfg_path = f'{root_dir}/models/SEED/configs/tokenizer/seed_llama_tokenizer_hf.yaml'
    tokenizer_cfg = OmegaConf.load(tokenizer_cfg_path)
    tokenizer = hydra.utils.instantiate(
    tokenizer_cfg, device=device, load_diffusion=(gen_mode == 'image')
    )
    special_tokens_dict = {'additional_special_tokens': ['<img>', '</img>']}
    tokenizer.add_special_tokens(special_tokens_dict)

    transform_cfg_path = f'{root_dir}/models/SEED/configs/transform/clip_transform.yaml'
    transform_cfg = OmegaConf.load(transform_cfg_path)
    transform = hydra.utils.instantiate(transform_cfg)

    model_cfg = OmegaConf.load(f'{root_dir}/models/SEED/configs/llm/seed_llama_8b_.yaml')
    model = hydra.utils.instantiate(model_cfg, torch_dtype=torch.float16)
    model = model.eval().to(device)
    
        
    model.resize_token_embeddings(len(tokenizer))
    
    if finetuned:
        ft_path = get_ft_path(
            'seed',
            gen_mode,
            shot,
            prompt_type,
            ft_mode,
            eval_task_theme,
        )['model']
        model = PeftModel.from_pretrained(model, ft_path)

    return model, tokenizer, transform

def process_image(
    image_input,
    transform,
    device,
    tokenizer,
):
    image = Image.open(image_input).convert('RGB')
    image_tensor = transform(image).to(device)
    img_ids = tokenizer.encode_image(image_torch=image_tensor)
    img_ids = img_ids.view(-1).cpu().numpy()
    img_tokens = BOI_TOKEN + ''.join([IMG_TOKEN.format(item)
                                    for item in img_ids]) + EOI_TOKEN
    return img_tokens

def preprocess(
    query,
    tokenizer,
    instruction,
    history,
    call_mode,
    transform,
    device,
    output_mode = 'eval_sample', # ['eval_sample', 'train_sample']
    max_len = 2048,
):
    text_inputs, image_inputs = query['text_inputs'], query['image_inputs']
    
    input_tokens = tokenizer.bos_token + s_token + instruction[0]
    if history is not None: input_tokens += history.replace(e_token, sep)
    
    for i in range(len(text_inputs)):
        input_tokens = input_tokens + text_inputs[i] + "\n"
        if call_mode == 'micl':
            if i < len(text_inputs) - 1:
                img_tokens = process_image(
                    image_inputs[i],
                    transform,
                    device,
                    tokenizer,
                )
                input_tokens = input_tokens + img_tokens

            if i == len(text_inputs) - 1:
                input_tokens = input_tokens + instruction[1] + e_token + sep
                
                if output_mode == 'train_sample':
                    img_tokens = process_image(
                        image_inputs[i],
                        transform,
                        device,
                        tokenizer,
                    )
                    output_tokens = input_tokens + 'I have created an image.' + img_tokens + e_token + sep
    if call_mode == 'text':
        input_tokens = input_tokens + instruction[1] + e_token + sep
                
    if output_mode == 'eval_sample':
        return input_tokens
    elif output_mode == 'train_sample': 
        input_ids, output_ids = [], []
        
        input_ids += tokenizer(
            input_tokens, 
            add_special_tokens=False, 
            return_tensors='pt',
        ).input_ids.squeeze()
        
        output_ids += tokenizer(
            output_tokens,
            add_special_tokens=False, 
            return_tensors='pt',
        ).input_ids.squeeze()
        
        output_ids[:len(input_ids)] = [IGNORE_TOKEN_ID] * len(input_ids)
        
        input_ids += [tokenizer.pad_token_id] * (max_len - len(input_ids))
        output_ids += [IGNORE_TOKEN_ID] * (max_len - len(output_ids))
        input_ids = torch.tensor(input_ids, dtype=torch.int)
        output_ids = torch.tensor(output_ids, dtype=torch.int)
        
        return dict(
            input_ids=input_ids,
            labels=output_ids,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
        )
    else:
        raise ValueError(f'output_mode {output_mode} not supported')

def call_seed(
    model,
    tokenizer,
    transform,
    text_inputs=["Red", "Green", "Yellow"],
    image_inputs=None,
    seed=123,
    gen_mode='text',
    device='cuda',
    instruction=None,  
    call_mode='micl',
    history=None,
    save_history=False,
):
    set_seed(seed)
    generation_config = get_generation_config(tokenizer, gen_mode=gen_mode)
    
    if instruction is None:
        if gen_mode == 'image':
            instruction = [
                "Generate an image for the description below. "
                "Output MUST contain <img> followed by image tokens and end with </img>. "
                "Do not output any other text.",
                ""
            ]
        else:
            instruction = [
                "Analyze the following description and provide detailed reasoning.",
                ""
            ]
    
    if image_inputs is None:
        image_inputs = []
    
    input_tokens = preprocess(
        query={'text_inputs': text_inputs, 'image_inputs': image_inputs},
        tokenizer=tokenizer,
        instruction=instruction,
        history=history,
        call_mode=call_mode,
        transform=transform,
        device=device,
        output_mode='eval_sample',
    )


    output_dict = {}
    if save_history: output_dict = {'history': input_tokens}
    
    seed_start = time()

    if gen_mode == 'image':
        generate_ids = generate(tokenizer, input_tokens, generation_config, model)
        
        print(f"[DEBUG] Generate IDs type: {type(generate_ids)}")
        print(f"[DEBUG] Generate IDs shape: {generate_ids.shape if hasattr(generate_ids, 'shape') else 'N/A'}")
        
        output_dict['description'], output_dict['image'] = decode_image_text(
            generate_ids, tokenizer, image_id_shift
        )
        
        if output_dict['image'] is None:
            print("[WARNING] Image decode failed, but SEED might have generated image tokens")
            
            img_token_count = 0
            if torch.is_tensor(generate_ids):
                ids = generate_ids.detach().cpu().tolist()
            else:
                ids = list(generate_ids)
                
            img_token_count = sum(1 for token_id in ids if token_id >= image_id_shift)
            
            if img_token_count > 0:
                print(f"[INFO] Found {img_token_count} image tokens in output")
                print(f"[INFO] This suggests SEED is working but decode_image() is failing")
                
                try:
                    import json
                    debug_data = {
                        "token_ids": ids,
                        "text": tokenizer.decode(generate_ids),
                        "has_img_tags": "<img>" in tokenizer.decode(generate_ids),
                        "image_token_count": img_token_count
                    }
                    
                    with open("seed_image_tokens_debug.json", "w") as f:
                        json.dump(debug_data, f)
                    print("[INFO] Debug data saved to seed_image_tokens_debug.json")
                    
                    decoded_text = tokenizer.decode(generate_ids)
                    if "<img_" in decoded_text:
                        print("[INFO] Found <img_XXXXX> tags in output")
                        print(f"[INFO] Sample: {decoded_text[:500]}")
                        
                except Exception as e:
                    print(f"[ERROR] Failed to save debug data: {e}")

    elif gen_mode == 'text':
        generate_ids = generate(tokenizer, input_tokens, generation_config, model)
        output_dict['description'], _ = decode_image_text(generate_ids, tokenizer, image_id_shift)
        if save_history: output_dict['history'] += ' ' + output_dict['description']
        seed_end = time()
        output_dict['time'] = seed_end - seed_start
    else:
        raise ValueError(f'gen_mode {gen_mode} not supported')

    return output_dict
