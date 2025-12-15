# Local Card Generation

This guide explains how to generate playing card artwork locally using GPU acceleration with the Z-Image-Turbo model.

## Prerequisites

- NVIDIA GPU with CUDA support
- Python 3.10 or higher
- At least 8GB VRAM (16GB recommended)
- CUDA 11.8 or higher installed

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements_local.txt
```

2. (Optional) For better performance, install Flash Attention:

```bash
pip install flash-attn --no-build-isolation
```

## Usage

### Generate a Single Card

```bash
python generate_local.py --single-card Ace Hearts
```

### Generate Card Back Only

```bash
python generate_local.py --card-back
```

### Generate Full Deck (52 cards + back)

```bash
python generate_local.py --full-deck
```

### Custom Theme and Style

```bash
python generate_local.py --single-card King Spades \
  --theme "Cyberpunk" \
  --technique "neon digital art" \
  --background "dark holographic"
```

## Command Line Options

- `--output-dir DIR` - Output directory for images (default: `card_images_local`)
- `--theme THEME` - Visual theme for the deck (default: "Western Steampunk")
- `--technique TECH` - Art technique/style (default: "Victorian engraving")
- `--background BG` - Background texture (default: "aged parchment")
- `--height H` - Image height (default: 1152)
- `--width W` - Image width (default: 640, maintains 9:16 ratio)
- `--steps N` - Number of inference steps (default: 9)
- `--seed SEED` - Random seed for reproducibility
- `--flash-attention` - Enable Flash Attention for better efficiency
- `--compile` - Compile model for faster inference (slower first run)

## Performance Optimization

### Flash Attention

Enable Flash Attention for better memory efficiency and speed:

```bash
python generate_local.py --full-deck --flash-attention
```

### Model Compilation

Compile the model for faster inference (first run will be slower):

```bash
python generate_local.py --full-deck --compile
```

### Combined Optimizations

```bash
python generate_local.py --full-deck --flash-attention --compile
```

## Examples

### Victorian Steampunk Deck
```bash
python generate_local.py --full-deck \
  --theme "Western Steampunk" \
  --technique "Victorian engraving" \
  --background "aged parchment"
```

### Cyberpunk Deck
```bash
python generate_local.py --full-deck \
  --theme "Cyberpunk" \
  --technique "neon digital art" \
  --background "dark holographic" \
  --seed 42
```

### Art Nouveau Deck
```bash
python generate_local.py --full-deck \
  --theme "Art Nouveau" \
  --technique "flowing organic lines" \
  --background "cream canvas"
```

### Japanese Ukiyo-e Deck
```bash
python generate_local.py --full-deck \
  --theme "Japanese Ukiyo-e" \
  --technique "woodblock print" \
  --background "rice paper"
```

## Memory Requirements

- **Single Card:** ~6GB VRAM
- **Full Deck:** ~6GB VRAM (cards generated sequentially)
- **With Flash Attention:** ~4GB VRAM
- **With Model Compilation:** First run requires more memory, subsequent runs are faster

## Troubleshooting

### CUDA Out of Memory

If you encounter OOM errors:

1. Try enabling Flash Attention: `--flash-attention`
2. Reduce image size: `--height 896 --width 512`
3. Close other GPU applications
4. Enable CPU offloading (edit `generate_local.py` and uncomment `pipe.enable_model_cpu_offload()`)

### Model Download

The first run will download the Z-Image-Turbo model (~6GB). This only happens once and the model is cached locally.

### Slow Generation

First-time generation is slower due to model loading. Use `--compile` for faster subsequent runs, but note the first run with compilation will be even slower.

## File Naming Convention

Generated files follow this pattern:
- `{Suit}_{OrderNumber}_{Value}.png` (e.g., `Hearts_01_Ace.png`, `Spades_13_King.png`)
- `ZZ_00_Card-Back.png` (card back design)

## Comparing with Cloud Generation

| Feature | Local (Z-Image) | Cloud (Google Gemini) |
|---------|----------------|----------------------|
| Cost | Free (after GPU) | Pay per generation |
| Speed | ~2-3 sec/card | ~30-60 sec/card |
| Quality | High | Very High |
| Customization | Full control | Full control |
| Requirements | GPU needed | Internet only |

## License

This script uses the Z-Image-Turbo model from Tongyi-MAI. Please check the model's license for commercial use restrictions.
