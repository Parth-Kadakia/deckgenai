"""
Local card generation adapted for Mac (CPU/MPS)
This version works on Macs without NVIDIA GPU by using CPU or Apple Silicon's MPS.
"""

import torch
from diffusers import ZImagePipeline
from pathlib import Path
import argparse
from typing import Optional
import platform

# Card configuration
SUITS = ["Hearts", "Spades", "Diamonds", "Clubs"]
VALUES = ["Ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King"]

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
        return "mps"  # Apple Silicon
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
- **Spacing:** Keep the important details clustered in the CENTER. Leave empty negative space around the edges (so it doesn't get cut off by a frame later).
- **Style:** Detailed, high-contrast, clean lines.

**NEGATIVE PROMPT:**
- playing card, border, frame, corner text, numbers, letters, symbols, typography, zoomed out, table surface, 3d render, text, watermark.
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
    """
    Initialize the Z-Image pipeline.
    
    Args:
        device: Target device (cuda/mps/cpu). If None, auto-detect.
    
    Returns:
        Initialized ZImagePipeline
    """
    if device is None:
        device = get_device()
    
    print(f"Loading Z-Image-Turbo pipeline on {device.upper()}...")
    
    # Choose appropriate dtype based on device
    if device == "cuda":
        dtype = torch.bfloat16
    elif device == "mps":
        dtype = torch.float16  # MPS works better with float16
    else:
        dtype = torch.float32  # CPU uses float32
    
    # Load the pipeline
    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    
    # Move to device
    if device == "mps":
        # MPS has some quirks, use enable_model_cpu_offload for better compatibility
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    
    print(f"Pipeline ready on {device.upper()}!")
    if device == "cpu":
        print("⚠️  Note: CPU generation is very slow. Consider using a GPU cloud service.")
    
    return pipe


def generate_card(
    pipe: ZImagePipeline,
    value: str,
    suit: str,
    output_dir: Path,
    theme: str = "Western Steampunk",
    technique: str = "Victorian engraving",
    background: str = "aged parchment",
    height: int = 1152,
    width: int = 640,
    num_inference_steps: int = 9,
    seed: Optional[int] = None,
    device: str = None
) -> Path:
    """Generate a single card image."""
    if device is None:
        device = get_device()
    
    # Build prompt
    prompt = build_prompt(value, suit, theme, technique, background)
    
    # Generate filename
    value_num = VALUE_ORDER[value]
    filename = f"{suit}_{value_num}_{value}.png"
    output_path = output_dir / filename
    
    print(f"Generating {value} of {suit}...")
    
    # Set up generator
    generator = torch.Generator(device)
    if seed is not None:
        generator.manual_seed(seed)
    
    # Generate image
    image = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=0.0,
        generator=generator,
    ).images[0]
    
    # Save image
    image.save(output_path)
    print(f"Saved to {output_path}")
    
    return output_path


def generate_card_back(
    pipe: ZImagePipeline,
    output_dir: Path,
    theme: str = "Western Steampunk",
    technique: str = "Victorian engraving",
    background: str = "aged parchment",
    height: int = 1152,
    width: int = 640,
    num_inference_steps: int = 9,
    seed: Optional[int] = None,
    device: str = None
) -> Path:
    """Generate card back design."""
    if device is None:
        device = get_device()
    
    prompt = build_card_back_prompt(theme, technique, background)
    
    filename = "ZZ_00_Card-Back.png"
    output_path = output_dir / filename
    
    print("Generating card back...")
    
    generator = torch.Generator(device)
    if seed is not None:
        generator.manual_seed(seed)
    
    image = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=0.0,
        generator=generator,
    ).images[0]
    
    image.save(output_path)
    print(f"Saved to {output_path}")
    
    return output_path


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Generate playing card artwork locally (Mac compatible)")
    parser.add_argument("--output-dir", type=str, default="card_images_local", help="Output directory for images")
    parser.add_argument("--theme", type=str, default="Western Steampunk", help="Visual theme for the deck")
    parser.add_argument("--technique", type=str, default="Victorian engraving", help="Art technique/style")
    parser.add_argument("--background", type=str, default="aged parchment", help="Background texture")
    parser.add_argument("--height", type=int, default=1152, help="Image height")
    parser.add_argument("--width", type=int, default=640, help="Image width (9:16 ratio)")
    parser.add_argument("--steps", type=int, default=9, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, choices=["cuda", "mps", "cpu"], default=None, help="Device to use (auto-detect if not specified)")
    parser.add_argument("--single-card", nargs=2, metavar=("VALUE", "SUIT"), help="Generate a single card (e.g., --single-card Ace Hearts)")
    parser.add_argument("--card-back", action="store_true", help="Generate only the card back")
    parser.add_argument("--full-deck", action="store_true", help="Generate all 52 cards plus back")
    
    args = parser.parse_args()
    
    # Detect device
    device = args.device if args.device else get_device()
    print(f"\nUsing device: {device.upper()}")
    if device == "cpu":
        print("⚠️  Warning: CPU generation is very slow (minutes per card).")
        print("   Consider using Google Colab with GPU: https://colab.research.google.com/")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize pipeline
    pipe = initialize_pipeline(device)
    
    # Generate based on mode
    if args.single_card:
        value, suit = args.single_card
        if value not in VALUES or suit not in SUITS:
            print(f"Error: Invalid card. Value must be in {VALUES}, Suit must be in {SUITS}")
            return
        
        generate_card(
            pipe, value, suit, output_dir,
            theme=args.theme,
            technique=args.technique,
            background=args.background,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            seed=args.seed,
            device=device
        )
    
    elif args.card_back:
        generate_card_back(
            pipe, output_dir,
            theme=args.theme,
            technique=args.technique,
            background=args.background,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            seed=args.seed,
            device=device
        )
    
    elif args.full_deck:
        print("Generating full deck (52 cards + back)...")
        print(f"This will take approximately {52 * 2 if device == 'cpu' else 52 * 0.1} minutes on {device.upper()}")
        
        # Generate all cards
        for suit in SUITS:
            for value in VALUES:
                generate_card(
                    pipe, value, suit, output_dir,
                    theme=args.theme,
                    technique=args.technique,
                    background=args.background,
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.steps,
                    seed=args.seed,
                    device=device
                )
        
        # Generate card back
        generate_card_back(
            pipe, output_dir,
            theme=args.theme,
            technique=args.technique,
            background=args.background,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            seed=args.seed,
            device=device
        )
        
        print("\nFull deck generation complete!")
    
    else:
        print("Please specify a generation mode:")
        print("  --single-card VALUE SUIT : Generate one card")
        print("  --card-back             : Generate card back only")
        print("  --full-deck             : Generate all 52 cards + back")
        parser.print_help()


if __name__ == "__main__":
    main()
