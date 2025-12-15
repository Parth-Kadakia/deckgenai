# Mac Setup Guide

## ✅ Your Setup Status

You're running on:
- **Platform:** macOS (Apple Silicon / M-series chip)
- **Python:** 3.13.7
- **Acceleration:** Metal Performance Shaders (MPS)

## Quick Start

The dependencies are installed and the model is downloading (~6GB). Once complete:

### 1. Run a test (if not already running):
```bash
.venv/bin/python test_setup_mac.py
```

### 2. Generate your first card:
```bash
.venv/bin/python generate_local_mac.py --single-card Ace Hearts
```

### 3. Generate a card back:
```bash
.venv/bin/python generate_local_mac.py --card-back
```

## Important Notes for Mac Users

### Performance
- **With MPS (Apple Silicon):** ~10-30 seconds per card ⚡
- **Without MPS (Intel Mac):** 2-5 minutes per card 🐌

### Alternative Option: Cloud-Based Generation
Since local generation on Mac can be slow, you might prefer using the cloud API:

```bash
# Use the existing Google Gemini API instead
.venv/bin/python app.py
```

Benefits of cloud generation:
- Higher quality output
- No large model download
- Works on any Mac (Intel or Apple Silicon)
- Consistent speed

## Commands

All commands should use the virtual environment:

```bash
# Activate virtual environment manually (optional):
source .venv/bin/activate

# Or use full path:
.venv/bin/python generate_local_mac.py [options]
```

### Examples

**Single card:**
```bash
.venv/bin/python generate_local_mac.py --single-card King Spades
```

**Custom theme:**
```bash
.venv/bin/python generate_local_mac.py --single-card Queen Hearts \
  --theme "Art Nouveau" \
  --technique "flowing organic lines" \
  --background "cream canvas"
```

**Full deck (warning: takes 10-30 minutes):**
```bash
.venv/bin/python generate_local_mac.py --full-deck
```

## Troubleshooting

### If generation is too slow:
1. **Use cloud API instead** - See `app.py`
2. **Reduce image size:**
   ```bash
   .venv/bin/python generate_local_mac.py --single-card Ace Hearts \
     --height 896 --width 512
   ```
3. **Use Google Colab** (free GPU in browser)

### If you get MPS errors:
The script automatically falls back to CPU if MPS has issues.

### Memory issues:
Close other applications to free up RAM.

## What's Happening During First Run

1. ✅ Virtual environment created (`.venv/`)
2. ✅ Dependencies installed (torch, diffusers, etc.)
3. 🔄 Model downloading (~6GB, one-time only)
4. 🔄 Test image generating

This initial setup takes 5-10 minutes, but subsequent runs are much faster!

## File Structure

```
carddeckgen/
├── .venv/                    # Virtual environment (don't commit)
├── generate_local_mac.py     # Mac-compatible generator
├── test_setup_mac.py         # Mac-compatible test
├── app.py                    # Cloud generation (Google Gemini)
├── card_images_local/        # Output from local generation
└── test_output/              # Test outputs
```

## Next Steps After Test Completes

1. Check the test output in `test_output/test_card.png`
2. If satisfied, generate cards you need
3. If too slow, switch to cloud API (`app.py`)

## Cost Comparison

| Method | Speed | Quality | Setup | Cost |
|--------|-------|---------|-------|------|
| Local (Mac MPS) | ~20s/card | Good | Complex | Free* |
| Cloud API | ~40s/card | Excellent | Easy | $$ per card |
| Google Colab GPU | ~3s/card | Good | Medium | Free** |

*After initial setup time
**With usage limits
