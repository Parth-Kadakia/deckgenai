"""
Batch card generation script - mimics the cloud API batch processing
Generates all 54 cards (52 standard + 2 jokers + 1 back) with progress tracking
"""

import torch
from diffusers import ZImagePipeline
from pathlib import Path
import json
import time
from datetime import datetime
from typing import Optional
import sys

# Card configuration
SUITS = ["Hearts", "Spades", "Diamonds", "Clubs"]
VALUES = ["Ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King"]
JOKERS = [("Joker", "14", "Joker1"), ("Joker", "15", "Joker2")]

# Numeric order for file naming
VALUE_ORDER = {
    "Ace": "01", "2": "02", "3": "03", "4": "04", "5": "05",
    "6": "06", "7": "07", "8": "08", "9": "09", "10": "10",
    "Jack": "11", "Queen": "12", "King": "13"
}


def get_device():
    """Detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


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
- **Spacing:** Keep the important details clustered in the CENTER. Leave empty negative space around the edges.
- **Style:** Detailed, high-contrast, clean lines.

**NEGATIVE PROMPT:**
- playing card, border, frame, corner text, numbers, letters, symbols, typography, zoomed out, table surface, 3d render, text, watermark.
"""


def build_joker_prompt(joker_num: int, theme: str, technique: str, background: str) -> str:
    """Build prompt for joker cards."""
    return f"""
**ART STYLE:** {technique}.
**THEME:** {theme}.
**FORMAT:** Vertical Art Print (9:16 aspect ratio).

**SUBJECT:**
A whimsical, mischievous jester or trickster character representing a Joker card (Joker #{joker_num}).
The character should be playful, mysterious, and embody chaos and unpredictability.

**COMPOSITION RULES:**
- **Background:** {background} texture. Full bleed.
- **Layout:** Centered character, dynamic pose, vertical composition.
- **Character:** Jester, fool, or trickster in the {theme} style.
- **Style:** Detailed, high-contrast, playful yet elegant.

**NEGATIVE PROMPT:**
- playing card, border, frame, corner text, numbers, letters, typography, zoomed out, 3d render, text, watermark.
"""


def build_card_back_prompt(theme: str, technique: str, background: str) -> str:
    """Build prompt for card back design."""
    return f"""
**ART STYLE:** {technique}.
**THEME:** {theme}.
**FORMAT:** Vertical Art Print (9:16 aspect ratio).

**SUBJECT:**
A decorative card back design for a playing card deck. This is the BACK of the card, not the front.
The design should be symmetrical, ornate, and reflect the {theme} theme.

**COMPOSITION RULES:**
- **Background:** {background} texture. Full bleed.
- **Layout:** Perfectly symmetrical design (180-degree rotational symmetry).
- **Central Element:** An ornate medallion, crest, or decorative motif centered on the card.
- **Border:** Intricate repeating pattern forming a decorative frame around the edges.
- **Pattern:** Fill the space between the border and center with repeating {theme}-themed decorative elements.
- **Style:** Detailed, high-contrast, clean lines, suitable for the back of playing cards.

**NEGATIVE PROMPT:**
- playing card faces, numbers, letters, asymmetrical design, text, watermark, 3d render.
"""


def initialize_pipeline(device: str = None) -> ZImagePipeline:
    """Initialize the Z-Image pipeline."""
    if device is None:
        device = get_device()
    
    print(f"🔧 Loading Z-Image-Turbo pipeline on {device.upper()}...")
    
    # Choose appropriate dtype based on device
    if device == "cuda":
        dtype = torch.bfloat16
    elif device == "mps":
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    
    if device == "mps":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    
    print(f"✅ Pipeline ready on {device.upper()}!\n")
    return pipe


class BatchGenerator:
    """Batch card generator with progress tracking."""
    
    def __init__(self, output_dir: Path, theme: str, technique: str, background: str,
                 height: int = 1152, width: int = 640, steps: int = 9, device: str = None):
        self.output_dir = output_dir
        self.theme = theme
        self.technique = technique
        self.background = background
        self.height = height
        self.width = width
        self.steps = steps
        self.device = device if device else get_device()
        
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Statistics
        self.total_cards = 55  # 52 + 2 jokers + 1 back
        self.completed = 0
        self.failed = 0
        self.start_time = None
        self.results = []
        
    def generate_card(self, pipe: ZImagePipeline, value: str, suit: str, 
                     card_type: str = "standard") -> dict:
        """Generate a single card and return result info."""
        start = time.time()
        
        try:
            # Build prompt based on card type
            if card_type == "joker":
                joker_num = int(value)
                prompt = build_joker_prompt(joker_num, self.theme, self.technique, self.background)
                filename = f"ZZ_Joker_{value}_Joker{joker_num}.png"
                display_name = f"Joker {joker_num}"
            elif card_type == "back":
                prompt = build_card_back_prompt(self.theme, self.technique, self.background)
                filename = "ZZ_ZZ_00_Card-Back.png"
                display_name = "Card Back"
            else:
                prompt = build_prompt(value, suit, self.theme, self.technique, self.background)
                value_num = VALUE_ORDER[value]
                filename = f"{suit}_{value_num}_{value}.png"
                display_name = f"{value} of {suit}"
            
            output_path = self.output_dir / filename
            
            # Generate
            generator = torch.Generator(self.device)
            image = pipe(
                prompt=prompt,
                height=self.height,
                width=self.width,
                num_inference_steps=self.steps,
                guidance_scale=0.0,
                generator=generator,
            ).images[0]
            
            # Save
            image.save(output_path)
            
            elapsed = time.time() - start
            self.completed += 1
            
            result = {
                "card": display_name,
                "filename": filename,
                "status": "success",
                "time": f"{elapsed:.2f}s"
            }
            
            # Progress bar
            progress = (self.completed / self.total_cards) * 100
            bar_length = 40
            filled = int(bar_length * self.completed / self.total_cards)
            bar = "█" * filled + "░" * (bar_length - filled)
            
            # Calculate ETA
            if self.start_time:
                elapsed_total = time.time() - self.start_time
                avg_time = elapsed_total / self.completed
                remaining = (self.total_cards - self.completed) * avg_time
                eta = f"ETA: {int(remaining // 60)}m {int(remaining % 60)}s"
            else:
                eta = "Calculating..."
            
            print(f"[{bar}] {progress:5.1f}% | {self.completed}/{self.total_cards} | {display_name:20s} | {elapsed:5.2f}s | {eta}")
            
            return result
            
        except Exception as e:
            self.failed += 1
            print(f"❌ Failed to generate {display_name}: {e}")
            return {
                "card": display_name,
                "filename": filename if 'filename' in locals() else "unknown",
                "status": "failed",
                "error": str(e)
            }
    
    def generate_all(self, pipe: ZImagePipeline):
        """Generate all 55 cards in batch."""
        print("=" * 80)
        print(f"🎨 BATCH CARD GENERATION")
        print("=" * 80)
        print(f"Theme: {self.theme}")
        print(f"Technique: {self.technique}")
        print(f"Output: {self.output_dir}")
        print(f"Device: {self.device.upper()}")
        print(f"Total Cards: {self.total_cards} (52 standard + 2 jokers + 1 back)")
        print("=" * 80)
        print()
        
        self.start_time = time.time()
        
        # Generate standard cards (52)
        print("📋 Generating standard cards (52)...")
        for suit in SUITS:
            for value in VALUES:
                result = self.generate_card(pipe, value, suit, "standard")
                self.results.append(result)
        
        # Generate jokers (2)
        print("\n🃏 Generating jokers (2)...")
        for joker_suit, joker_num, joker_name in JOKERS:
            result = self.generate_card(pipe, joker_num, joker_suit, "joker")
            self.results.append(result)
        
        # Generate card back (1)
        print("\n🔙 Generating card back (1)...")
        result = self.generate_card(pipe, "", "", "back")
        self.results.append(result)
        
        # Summary
        total_time = time.time() - self.start_time
        avg_time = total_time / self.completed if self.completed > 0 else 0
        
        print("\n" + "=" * 80)
        print("✅ BATCH GENERATION COMPLETE!")
        print("=" * 80)
        print(f"Total Time: {int(total_time // 60)}m {int(total_time % 60)}s")
        print(f"Average Time per Card: {avg_time:.2f}s")
        print(f"Successful: {self.completed}/{self.total_cards}")
        print(f"Failed: {self.failed}/{self.total_cards}")
        print(f"Output Directory: {self.output_dir}")
        print("=" * 80)
        
        # Save results to JSON
        self.save_results(total_time)
    
    def save_results(self, total_time: float):
        """Save batch results to JSON file."""
        results_file = self.output_dir / "batch_results.json"
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "theme": self.theme,
            "technique": self.technique,
            "background": self.background,
            "device": self.device,
            "total_cards": self.total_cards,
            "completed": self.completed,
            "failed": self.failed,
            "total_time": f"{int(total_time // 60)}m {int(total_time % 60)}s",
            "avg_time_per_card": f"{total_time / self.completed:.2f}s" if self.completed > 0 else "N/A",
            "cards": self.results
        }
        
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📊 Results saved to: {results_file}")


def main():
    """Main batch generation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch generate all 54 playing cards + back")
    parser.add_argument("--output-dir", type=str, default="card_images_local", 
                       help="Output directory for images")
    parser.add_argument("--theme", type=str, default="Western Steampunk", 
                       help="Visual theme for the deck")
    parser.add_argument("--technique", type=str, default="Victorian engraving", 
                       help="Art technique/style")
    parser.add_argument("--background", type=str, default="aged parchment", 
                       help="Background texture")
    parser.add_argument("--height", type=int, default=1152, 
                       help="Image height")
    parser.add_argument("--width", type=int, default=640, 
                       help="Image width (9:16 ratio)")
    parser.add_argument("--steps", type=int, default=9, 
                       help="Number of inference steps")
    parser.add_argument("--device", type=str, choices=["cuda", "mps", "cpu"], 
                       default=None, help="Device to use (auto-detect if not specified)")
    
    args = parser.parse_args()
    
    # Detect device
    device = args.device if args.device else get_device()
    
    # Warning for CPU
    if device == "cpu":
        print("⚠️  WARNING: CPU generation detected!")
        print("This will take 2-5 HOURS to complete all 55 cards.")
        print("Consider using:")
        print("  • Google Colab with free GPU")
        print("  • Cloud API (app.py)")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Initialize pipeline
    pipe = initialize_pipeline(device)
    
    # Create batch generator
    generator = BatchGenerator(
        output_dir=Path(args.output_dir),
        theme=args.theme,
        technique=args.technique,
        background=args.background,
        height=args.height,
        width=args.width,
        steps=args.steps,
        device=device
    )
    
    # Generate all cards
    try:
        generator.generate_all(pipe)
    except KeyboardInterrupt:
        print("\n\n⚠️  Generation interrupted by user!")
        print(f"Completed: {generator.completed}/{generator.total_cards}")
        print(f"Partial results saved to: {generator.output_dir}")
    except Exception as e:
        print(f"\n\n❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
