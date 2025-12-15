"""
Quick test script for Mac (works with CPU or Apple Silicon MPS)
"""

import torch
from diffusers import ZImagePipeline
from pathlib import Path
import platform

def get_device():
    """Detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def test_setup():
    """Test the local generation setup on Mac."""
    print("=" * 60)
    print("Z-Image-Turbo Setup Test (Mac)")
    print("=" * 60)
    
    # System info
    print(f"\n1. System Information...")
    print(f"   Platform: {platform.system()} {platform.machine()}")
    print(f"   Python: {platform.python_version()}")
    print(f"   PyTorch: {torch.__version__}")
    
    # Check available device
    print(f"\n2. Detecting compute device...")
    device = get_device()
    
    if device == "cuda":
        print(f"   ✅ CUDA GPU detected")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    elif device == "mps":
        print(f"   ✅ Apple Silicon (MPS) detected")
        print(f"   This will use Metal Performance Shaders for acceleration")
    else:
        print(f"   ⚠️  CPU only (no GPU acceleration)")
        print(f"   Generation will be VERY slow (~2-5 minutes per card)")
        print(f"   Consider using Google Colab with free GPU instead")
    
    print(f"\n   Selected device: {device.upper()}")
    
    # Choose dtype
    if device == "cuda":
        dtype = torch.bfloat16
    elif device == "mps":
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    print(f"   Using dtype: {dtype}")
    
    # Try loading the pipeline
    print(f"\n3. Loading Z-Image-Turbo pipeline...")
    print("   (This will download ~6GB on first run)")
    
    try:
        pipe = ZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        
        if device == "mps":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)
        
        print("   ✅ Pipeline loaded successfully")
    except Exception as e:
        print(f"   ❌ Failed to load pipeline: {e}")
        return False
    
    # Generate a test image
    print(f"\n4. Generating test image...")
    if device == "cpu":
        print("   ⚠️  This will take 2-5 minutes on CPU...")
    
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    prompt = "A majestic ace of spades playing card design, Victorian steampunk style, ornate golden gears and clockwork elements, centered composition, detailed engraving technique, aged parchment background."
    
    try:
        generator = torch.Generator(device)
        generator.manual_seed(42)
        
        image = pipe(
            prompt=prompt,
            height=1152,
            width=640,
            num_inference_steps=9,
            guidance_scale=0.0,
            generator=generator,
        ).images[0]
        
        output_path = output_dir / "test_card.png"
        image.save(output_path)
        print(f"   ✅ Test image generated successfully")
        print(f"   Saved to: {output_path}")
    except Exception as e:
        print(f"   ❌ Failed to generate image: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Success!
    print("\n" + "=" * 60)
    print("✅ All tests passed! Your setup is ready.")
    print("=" * 60)
    
    if device == "cpu":
        print("\n⚠️  CPU Performance Warning:")
        print("   Generation on CPU is very slow. Consider:")
        print("   • Using Google Colab with free GPU")
        print("   • Using the cloud API (app.py) instead")
    
    print("\nNext steps:")
    print("  • Generate a single card:")
    print("    python generate_local_mac.py --single-card Ace Hearts")
    print("\n  • Generate card back:")
    print("    python generate_local_mac.py --card-back")
    print("\n  • See all options:")
    print("    python generate_local_mac.py --help")
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
