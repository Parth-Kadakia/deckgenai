"""
Fast batch generation with optimized settings for Mac
Reduces quality slightly for much faster generation
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from batch_generate import BatchGenerator, initialize_pipeline, get_device

def main():
    """Fast batch generation with reduced steps."""
    import argparse
    
    parser = argparse.ArgumentParser(description="FAST batch generation (optimized for Mac)")
    parser.add_argument("--output-dir", type=str, default="card_images_fast", 
                       help="Output directory for images")
    parser.add_argument("--theme", type=str, default="Western Steampunk", 
                       help="Visual theme for the deck")
    parser.add_argument("--technique", type=str, default="Victorian engraving", 
                       help="Art technique/style")
    parser.add_argument("--background", type=str, default="aged parchment", 
                       help="Background texture")
    parser.add_argument("--height", type=int, default=896,  # Reduced from 1152
                       help="Image height (reduced for speed)")
    parser.add_argument("--width", type=int, default=512,   # Reduced from 640
                       help="Image width (reduced for speed)")
    parser.add_argument("--steps", type=int, default=4,     # Reduced from 9
                       help="Number of inference steps (reduced for speed)")
    
    args = parser.parse_args()
    
    device = get_device()
    
    print("⚡ FAST GENERATION MODE")
    print("=" * 60)
    print("⚠️  Settings optimized for SPEED over quality:")
    print(f"   • Image size: {args.width}x{args.height} (vs 640x1152)")
    print(f"   • Inference steps: {args.steps} (vs 9)")
    print(f"   • Expected time per card: ~8-12 minutes on Mac MPS")
    print(f"   • Total time for 55 cards: ~7-11 hours")
    print("=" * 60)
    
    response = input("\nContinue with fast generation? (y/n): ")
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
