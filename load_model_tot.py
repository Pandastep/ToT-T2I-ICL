# load_model_tot.py
import os
root_dir = os.path.dirname(os.path.abspath(__file__))
from load_models.call_seed_tot import call_seed, load_seed

def load_model_tot(
    model,
    device='cuda',
    gen_mode='text',
    finetuned=False,
    shot=2,
    prompt_type='tot',
    ft_mode='all',
    eval_task_theme='',
):
    if finetuned and model not in ['qwen', 'seed']:
        raise ValueError(f"finetuned is only supported for qwen/seed. Got: {model}")
    if model != 'seed':
        raise ValueError(f"ToT reasoning supports only SEED. Got: {model}")

    model_obj, tokenizer, transform = load_seed(
        device=device,
        finetuned=finetuned,
        shot=shot,
        gen_mode=gen_mode,
        prompt_type=prompt_type,
        ft_mode=ft_mode,
        eval_task_theme=eval_task_theme,
    )

    def _call_model(cfg):
        allowed = {
            "model", "tokenizer", "transform",
            "text_inputs", "image_inputs",
            "seed", "gen_mode", "device",
            "instruction", "call_mode",
            "history", "save_history",
        }
        safe_cfg = {k: v for k, v in dict(cfg).items() if k in allowed}
        

        safe_cfg.setdefault("model", model_obj)
        safe_cfg.setdefault("tokenizer", tokenizer)
        safe_cfg.setdefault("transform", transform)
        safe_cfg.setdefault("text_inputs", [])
        safe_cfg.setdefault("image_inputs", [])
        safe_cfg.setdefault("seed", 123)
        safe_cfg.setdefault("gen_mode", gen_mode)
        safe_cfg.setdefault("device", device)
    
        safe_cfg.setdefault("call_mode", cfg.get("call_mode", "text"))
        safe_cfg.setdefault("save_history", False)
        
        
        if safe_cfg.get("gen_mode") == "image":
            
            if not safe_cfg.get("image_inputs"):
                safe_cfg["call_mode"] = "text"
        
        return call_seed(**safe_cfg)

    return {
        "call_model": _call_model,
        "model": model_obj,
        "tokenizer": tokenizer,
        "transform": transform,
    }