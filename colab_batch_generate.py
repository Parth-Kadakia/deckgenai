"""
Google Colab Batch Generation Script
Upload this to Google Colab and run with free GPU for fast generation!

Instructions:
1. Go to https://colab.research.google.com/
2. Create new notebook
3. Runtime → Change runtime type → GPU → Save
4. Copy this script to a cell and run
5. Or upload this file and run: !python colab_batch_generate.py
"""

# Install dependencies (run this first in Colab)
# !pip install torch diffusers transformers accelerate pillow

import torch
from diffusers import ZImagePipeline
from pathlib import Path
import json
import time
from datetime import datetime
import sys

# Card configuration
SUITS = ["Hearts", "Spades", "Diamonds", "Clubs"]
VALUES = ["Ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King"]

VALUE_ORDER = {
    "Ace": "01", "2": "02", "3": "03", "4": "04", "5": "05",
    "6": "06", "7": "07", "8": "08", "9": "09", "10": "10",
    "Jack": "11", "Queen": "12", "King": "13"
}

def build_prompt(value: str, suit: str, theme: str, technique: str, background: str) -> str:
    """Build a prompt for generating card artwork."""
    is_face_card = str(value).lower() in ['k', 'q', 'j', 'king', 'queen', 'jack']

    if is_face_card:
        subject = f"A majestic portrait of a character representing the {value} of {suit}"
        composition = "centered character bust, facing forward, vertical composition"
    else:
        subject = f"A symmetrical decorative arrangement of {value} distinct items representing {suit}"
        composition = "objects arranged in a tight central cluster, vertical composition"

    return f"""
**ART STYLE:** {technique}.
**THEME:** {theme}.
**FORMAT:** Vertical Art Print (9:16 aspect ratio).

**SUBJECT:**
{subject}.
The artwork must interpret the concept of "{value}" and "{suit}" using the visual language of {theme}.

**COMPOSITION RULES:**
- **Background:** {background} texture. Full bleed. No borders.
- **Layout:** {composition}.
- **Spacing:** Keep the important details clustered in the CENTER.
- **Style:** Detailed, high-contrast, clean lines.

**NEGATIVE PROMPT:**
- playing card, border, frame, corner text, numbers, letters, symbols, typography, 3d render, text, watermark.
"""


def build_joker_prompt(joker_num: int, theme: str, technique: str, background: str) -> str:
    """Build prompt for joker cards."""
    return f"""
**ART STYLE:** {technique}.
**THEME:** {theme}.
**FORMAT:** Vertical Art Print (9:16 aspect ratio).

**SUBJECT:**
A whimsical jester character representing Joker {joker_num}.

**COMPOSITION RULES:**
- **Background:** {background} texture.
- **Layout:** Centered character, dynamic pose.
- **Style:** Playful yet elegant, {theme} style.

**NEGATIVE PROMPT:**
- border, frame, text, watermark, 3d render.
"""


def build_card_back_prompt(theme: str, technique: str, background: str) -> str:
    """Build prompt for card back design."""
    return f"""
**ART STYLE:** {technique}.
**THEME:** {theme}.
**FORMAT:** Vertical Art Print (9:16 aspect ratio).

**SUBJECT:**
A symmetrical, ornate card back design reflecting {theme} theme.

**COMPOSITION RULES:**
- **Background:** {background} texture.
- **Layout:** Perfectly symmetrical (180-degree rotational symmetry).
- **Style:** Detailed, ornate, suitable for playing card back.

**NEGATIVE PROMPT:**
- card faces, numbers, letters, text, asymmetrical, 3d render.
"""


def generate_full_deck(
    theme: str = "Western Steampunk",
    technique: str = "Victorian engraving", 
    background: str = "aged parchment",
    output_dir: str = "/content/cards",
    height: int = 1152,
    width: int = 640,
    steps: int = 9
):
    """Generate all 55 cards with Colab GPU."""
    
    print("=" * 80)
    print("🎨 GOOGLE COLAB BATCH GENERATION")
    print("=" * 80)
    print(f"Theme: {theme}")
    print(f"Output: {output_dir}")
    print(f"Device: CUDA GPU (Colab)")
    print("=" * 80)
    print()
    
    # Check GPU
    if not torch.cuda.is_available():
        print("❌ CUDA GPU not available!")
        print("Go to: Runtime → Change runtime type → GPU → Save")
        return
    
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Load pipeline
    print("🔧 Loading Z-Image-Turbo pipeline...")
    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    pipe.to("cuda")
    print("✅ Pipeline ready!\n")
    
    # Statistics
    start_time = time.time()
    completed = 0
    total = 55
    results = []
    
    # Generate standard cards (52)
    print("📋 Generating standard cards (52)...")
    for suit in SUITS:
        for value in VALUES:
            card_start = time.time()
            
            prompt = build_prompt(value, suit, theme, technique, background)
            value_num = VALUE_ORDER[value]
            filename = f"{suit}_{value_num}_{value}.png"
            
            try:
                generator = torch.Generator("cuda")
                image = pipe(
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_inference_steps=steps,
                    guidance_scale=0.0,
                    generator=generator,
                ).images[0]
                
                image.save(output_path / filename)
                
                elapsed = time.time() - card_start
                completed += 1
                
                progress = (completed / total) * 100
                bar_length = 40
                filled = int(bar_length * completed / total)
                bar = "█" * filled + "░" * (bar_length - filled)
                
                print(f"[{bar}] {progress:5.1f}% | {completed}/{total} | {value} of {suit:8s} | {elapsed:5.2f}s")
                
                results.append({
                    "card": f"{value} of {suit}",
                    "filename": filename,
                    "status": "success",
                    "time": f"{elapsed:.2f}s"
                })
            except Exception as e:
                print(f"❌ Failed: {value} of {suit}: {e}")
    
    # Generate jokers (2)
    print("\n🃏 Generating jokers (2)...")
    for joker_num in [1, 2]:
        card_start = time.time()
        
        prompt = build_joker_prompt(joker_num, theme, technique, background)
        filename = f"ZZ_Joker_{13+joker_num}_Joker{joker_num}.png"
        
        try:
            generator = torch.Generator("cuda")
            image = pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=0.0,
                generator=generator,
            ).images[0]
            
            image.save(output_path / filename)
            
            elapsed = time.time() - card_start
            completed += 1
            
            progress = (completed / total) * 100
            bar_length = 40
            filled = int(bar_length * completed / total)
            bar = "█" * filled + "░" * (bar_length - filled)
            
            print(f"[{bar}] {progress:5.1f}% | {completed}/{total} | Joker {joker_num:14s} | {elapsed:5.2f}s")
        except Exception as e:
            print(f"❌ Failed: Joker {joker_num}: {e}")
    
    # Generate card back (1)
    print("\n🔙 Generating card back (1)...")
    card_start = time.time()
    
    prompt = build_card_back_prompt(theme, technique, background)
    filename = "ZZ_ZZ_00_Card-Back.png"
    
    try:
        generator = torch.Generator("cuda")
        image = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=0.0,
            generator=generator,
        ).images[0]
        
        image.save(output_path / filename)
        
        elapsed = time.time() - card_start
        completed += 1
        
        print(f"[{'█' * 40}] 100.0% | {completed}/{total} | Card Back        | {elapsed:5.2f}s")
    except Exception as e:
        print(f"❌ Failed: Card Back: {e}")
    
    # Summary
    total_time = time.time() - start_time
    avg_time = total_time / completed if completed > 0 else 0
    
    print("\n" + "=" * 80)
    print("✅ BATCH GENERATION COMPLETE!")
    print("=" * 80)
    print(f"Total Time: {int(total_time // 60)}m {int(total_time % 60)}s")
    print(f"Average Time per Card: {avg_time:.2f}s")
    print(f"Completed: {completed}/{total}")
    print(f"Output: {output_path}")
    print("=" * 80)
    print("\n💡 To download: Right-click on the 'cards' folder → Download")
    print("   Or use: !zip -r cards.zip /content/cards")


# Main execution
if __name__ == "__main__":
    # Customize these settings
    THEME = "Western Steampunk"
    TECHNIQUE = "Victorian engraving"
    BACKGROUND = "aged parchment"
    
    # For different themes, uncomment one:
    # THEME, TECHNIQUE, BACKGROUND = "Cyberpunk", "neon digital art", "dark holographic"
    # THEME, TECHNIQUE, BACKGROUND = "Art Nouveau", "flowing organic lines", "cream canvas"
    # THEME, TECHNIQUE, BACKGROUND = "Japanese Ukiyo-e", "woodblock print", "rice paper"
    
    generate_full_deck(
        theme=THEME,
        technique=TECHNIQUE,
        background=BACKGROUND,
        output_dir="/content/cards",
        height=1152,
        width=640,
        steps=9
    )
