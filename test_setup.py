"""
Quick test script to verify Z-Image-Turbo setup
Generates a single test card to confirm everything works.
"""

import torch
from diffusers import ZImagePipeline
from pathlib import Path

def test_setup():
    """Test the local generation setup."""
    print("=" * 60)
    print("Z-Image-Turbo Setup Test")
    print("=" * 60)
    
    # Check CUDA availability
    print(f"\n1. Checking CUDA availability...")
    if not torch.cuda.is_available():
        print("   ❌ CUDA is not available. You need an NVIDIA GPU with CUDA support.")
        return False
    
    print(f"   ✅ CUDA is available")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Check bfloat16 support
    print(f"\n2. Checking bfloat16 support...")
    if torch.cuda.is_bf16_supported():
        print("   ✅ bfloat16 is supported")
    else:
        print("   ⚠️  bfloat16 might not be fully supported on this GPU")
    
    # Try loading the pipeline
    print(f"\n3. Loading Z-Image-Turbo pipeline...")
    print("   (This will download ~6GB on first run)")
    
    try:
        pipe = ZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
        )
        pipe.to("cuda")
        print("   ✅ Pipeline loaded successfully")
    except Exception as e:
        print(f"   ❌ Failed to load pipeline: {e}")
        return False
    
    # Generate a test image
    print(f"\n4. Generating test image...")
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    prompt = "A majestic ace of spades playing card design, Victorian steampunk style, ornate golden gears and clockwork elements, centered composition, detailed engraving technique, aged parchment background."
    
    try:
        image = pipe(
            prompt=prompt,
            height=1152,
            width=640,
            num_inference_steps=9,
            guidance_scale=0.0,
            generator=torch.Generator("cuda").manual_seed(42),
        ).images[0]
        
        output_path = output_dir / "test_card.png"
        image.save(output_path)
        print(f"   ✅ Test image generated successfully")
        print(f"   Saved to: {output_path}")
    except Exception as e:
        print(f"   ❌ Failed to generate image: {e}")
        return False
    
    # Success!
    print("\n" + "=" * 60)
    print("✅ All tests passed! Your setup is ready.")
    print("=" * 60)
    print("\nNext steps:")
    print("  • Generate a single card: python generate_local.py --single-card Ace Hearts")
    print("  • Generate full deck:    python generate_local.py --full-deck")
    print("  • See all options:       python generate_local.py --help")
    print()
    
    return True

if __name__ == "__main__":
    try:
        test_setup()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
