# image_generator.py 
import torch
import os
from PIL import Image
from typing import Optional, List
import gc

class ImageGenerator:
    
    _seed_model = None 
    _sd_model = None
    
    def __init__(self, device="cuda", use_seed=True, use_sd=True):
        self.device = device
        self.use_seed = use_seed
        self.use_sd = use_sd
        self.seed_loaded = False
        self.sd_loaded = False
        
    def _load_seed_model(self):
    
        if ImageGenerator._seed_model is not None:
            print("♻️  Reusing existing SEED model")
            return ImageGenerator._seed_model
            
        try:
            from load_models.call_seed_tot import load_seed
            
            print("📥 Loading SEED model (first time)...")
            model, tokenizer, transform = load_seed(
                device=self.device,
                gen_mode="image"
            )
            
            ImageGenerator._seed_model = (model, tokenizer, transform)
            self.seed_loaded = True
            print("✅ SEED model loaded")
            return ImageGenerator._seed_model
            
        except Exception as e:
            print(f"❌ SEED loading error: {e}")
            return None
    
    def _load_sd_model(self):
       
        if ImageGenerator._sd_model is not None:
            print("♻️  Reusing existing SD model")
            return ImageGenerator._sd_model
            
        try:
            from diffusers import StableDiffusionPipeline
            
            print("📥 Loading Stable Diffusion model (first time)...")
            model = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)
            
            
            model.enable_attention_slicing()
            
            ImageGenerator._sd_model = model
            self.sd_loaded = True
            print("✅ SD model loaded")
            return ImageGenerator._sd_model
            
        except ImportError:
            print("❌ diffusers not installed. Install with: pip install diffusers")
            return None
        except Exception as e:
            print(f"❌ SD loading error: {e}")
            return None
    
    def generate_with_seed(self, prompt: str, seed: int = 123) -> Optional[Image.Image]:
        """Генерация с SEED"""
        if not self.use_seed:
            return None
            
        try:
        
            model_data = self._load_seed_model()
            if model_data is None:
                return None
                
            model, tokenizer, transform = model_data
            
            from load_models.call_seed_tot import call_seed
            
            result = call_seed(
                model=model,
                tokenizer=tokenizer,
                transform=transform,
                text_inputs=[prompt],
                image_inputs=[],
                seed=seed,
                gen_mode="image",
                device=self.device,
                call_mode="text",
                instruction=None,
            )
            
            img = result.get("image", None)
            if img is not None:
                print("✅ SEED generation successful")
                return img
            else:
                print("❌ SEED returned no image")
                return None
                
        except Exception as e:
            print(f"❌ SEED generation error: {e}")
            return None
    
    def generate_with_sd(self, prompt: str, seed: int = 123) -> Optional[Image.Image]:
        """Генерация с Stable Diffusion"""
        if not self.use_sd:
            return None
            
        try:
            
            model = self._load_sd_model()
            if model is None:
                return None
            
            
            simple_prompt = self._simplify_prompt(prompt)
            print(f"🤖 SD prompt: {simple_prompt[:80]}...")
            
            
            generator = torch.Generator(device=self.device).manual_seed(seed)
            with torch.no_grad():
                image = model(
                    simple_prompt,
                    generator=generator,
                    num_inference_steps=30,
                    guidance_scale=7.5
                ).images[0]
            
            print("✅ SD generation successful")
            return image
            
        except Exception as e:
            print(f"❌ SD generation error: {e}")
            return None
    
    def _simplify_prompt(self, prompt: str) -> str:
        
        import re
        
        
        lines = prompt.split('\n')
        clean_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
           
            if any(keyword in line.lower() for keyword in 
                  ['generate an image', 'output must', 'no extra', 'keep identity']):
                continue
            
            
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    value = parts[1].strip()
                    if value:
                        clean_lines.append(value)
            elif len(line) > 10:
                clean_lines.append(line)
        
        if clean_lines:
            result = ". ".join(clean_lines[:2])  # Первые 2 описания
        else:
            result = prompt
        
       
        result = re.sub(r'\s+', ' ', result).strip()
        result = re.sub(r'\.{2,}', '.', result)
        
        return result[:150]
    
    def generate(self, prompt: str, seed: int = 123, 
                prefer_seed: bool = True) -> Optional[Image.Image]:
        
        print(f"\n🎨 Generating: {prompt[:60]}...")
        
        if prefer_seed and self.use_seed:
            print("🔄 Trying SEED first...")
            image = self.generate_with_seed(prompt, seed)
            if image is not None:
                return image
        
        if self.use_sd:
            print("🔄 Trying Stable Diffusion...")
            image = self.generate_with_sd(prompt, seed)
            if image is not None:
                return image
        
        print("❌ All generation methods failed")
        return None
    
    @staticmethod
    def clear_cache():
        
        print("🧹 Clearing model cache...")
        ImageGenerator._seed_model = None
        ImageGenerator._sd_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()