# Card Deck Generator AI

Generate custom playing card artwork using AI. Supports both cloud-based (Google Gemini) and local GPU (Z-Image-Turbo) generation.

## Features

- 🎨 Generate complete 52-card decks with custom themes
- 🎭 Support for various art styles (Victorian, Cyberpunk, Art Nouveau, etc.)
- ☁️ Cloud generation with Google Gemini (high quality, slower)
- 🖥️ Local GPU generation with Z-Image-Turbo (fast, requires GPU)
- 🎯 Consistent card layouts and proportions
- 🔄 Batch processing support

## Generation Methods

### Method 1: Cloud Generation (Google Gemini)

**Pros:**
- Highest quality output
- No GPU required
- Works from any device

**Cons:**
- Requires API key and credits
- Slower (~30-60 sec per card)
- Costs money per generation

**Setup:** See `app.py`

### Method 2: Local GPU Generation (Z-Image-Turbo)

**Pros:**
- Fast generation (~2-3 sec per card)
- Free after initial setup
- Full offline capability
- No API limits

**Cons:**
- Requires NVIDIA GPU (8GB+ VRAM)
- Requires setup and dependencies
- Slightly lower quality than cloud

**Setup:** See [README_LOCAL.md](README_LOCAL.md)

## Quick Start (Local Generation)

1. **Install dependencies:**
   ```bash
   .venv/bin/pip install -r requirements_local.txt
   # Or: python3 -m pip install -r requirements_local.txt
   ```

2. **Test your setup:**
   ```bash
   .venv/bin/python test_setup_mac.py
   ```

3. **Generate a single card:**
   ```bash
   .venv/bin/python generate_local_mac.py --single-card Ace Hearts
   ```

4. **Generate full deck (all 55 cards):**
   ```bash
   # Option 1: Simple full deck generation
   .venv/bin/python generate_local_mac.py --full-deck
   
   # Option 2: Batch generation with progress tracking (recommended)
   .venv/bin/python batch_generate.py
   ```

## Project Structure

```
carddeckgen/
├── app.py                    # Cloud generation (Google Gemini)
├── generate_local.py         # Local generation (Z-Image-Turbo, CUDA)
├── generate_local_mac.py     # Local generation (Mac compatible)
├── batch_generate.py         # Batch generation with progress tracking
├── test_setup.py            # Setup verification script (CUDA)
├── test_setup_mac.py        # Setup verification script (Mac)
├── requirements_local.txt   # Local generation dependencies
├── README.md                # This file
├── README_LOCAL.md          # Detailed local generation guide
├── MAC_SETUP.md             # Mac-specific setup guide
├── deck.jsonl               # Deck configuration
├── playing-cards-batch.jsonl # Batch processing config
├── card_images/             # Generated cards (cloud)
├── card_images_local/       # Generated cards (local)
│   └── batch_results.json   # Batch generation results
└── batch_output/            # Batch processing results
```

## Usage Examples

### Local Generation

```bash
# Single card
.venv/bin/python generate_local_mac.py --single-card King Spades

# Card back
.venv/bin/python generate_local_mac.py --card-back

# Batch: Full deck (52 cards + 2 jokers + 1 back = 55 total)
.venv/bin/python batch_generate.py

# Batch: Custom theme
.venv/bin/python batch_generate.py \
  --theme "Cyberpunk" \
  --technique "neon digital art" \
  --background "dark holographic"

# Simple full deck generation
.venv/bin/python generate_local_mac.py --full-deck
```

### Customization

Both methods support extensive customization:

- **Themes:** Western Steampunk, Cyberpunk, Art Nouveau, Japanese Ukiyo-e, Egyptian, Medieval, etc.
- **Techniques:** Victorian engraving, woodblock print, oil painting, watercolor, digital art, etc.
- **Backgrounds:** Aged parchment, holographic, canvas, rice paper, marble, etc.

## Card Naming Convention

Generated cards follow this naming pattern:
- `{Suit}_{OrderNumber}_{Value}.png`
  - Example: `Hearts_01_Ace.png`, `Spades_13_King.png`
- `ZZ_00_Card-Back.png` (card back design)

## Requirements

### For Local Generation
- Python 3.10+
- NVIDIA GPU with 8GB+ VRAM
- CUDA 11.8+
- See `requirements_local.txt`

### For Cloud Generation
- Python 3.8+
- Google Gemini API key
- Internet connection

## Performance Comparison

| Method | Time per Card | Full Deck (55) | Quality | Cost | Setup |
|--------|---------------|----------------|---------|------|-------|
| CUDA GPU | ~2-3 seconds | ~3-5 minutes | High | Free* | Medium |
| Mac MPS | ~20-25 minutes | ~18-23 hours | High | Free* | Easy |
| Google Colab GPU | ~2-3 seconds | ~3-5 minutes | High | Free** | Easy |
| Cloud API | ~30-60 seconds | ~30-60 minutes | Excellent | $2-5 | Easy |

*After hardware purchase  
**Free tier with usage limits

⚠️ **Mac Users:** Local generation is very slow. See [SPEED_OPTIMIZATION.md](SPEED_OPTIMIZATION.md) for faster alternatives.

## Tips

### For Best Results (Local)
1. Use `--flash-attention` for better memory efficiency
2. Use `--compile` for faster inference (after first run)
3. Set `--seed` for reproducible results
4. Adjust `--steps` (4-16) to balance speed vs quality

### For Best Results (Cloud)
1. Use detailed, structured prompts
2. Specify art style and composition clearly
3. Use batch processing for full decks
4. Monitor API usage and costs

## Troubleshooting

### Local Generation Issues

**CUDA Out of Memory:**
- Enable Flash Attention: `--flash-attention`
- Reduce image size: `--height 896 --width 512`
- Close other GPU applications

**Slow Performance:**
- Use `--compile` flag (slower first run, faster after)
- Ensure GPU drivers are up to date
- Check GPU is being used: `nvidia-smi`

**Model Download Fails:**
- Check internet connection
- Ensure you have enough disk space (~6GB)
- Try using a VPN if geoblocked

### Cloud Generation Issues

**API Errors:**
- Verify API key is correct
- Check account credits/quota
- Ensure internet connection is stable

## Contributing

Contributions are welcome! Areas for improvement:
- Additional art styles and themes
- Better prompt engineering
- UI/web interface
- Batch processing optimizations

## License

This project uses:
- Z-Image-Turbo model (check model license)
- Google Gemini API (subject to Google's terms)

See individual model/API licenses for commercial use restrictions.

## Acknowledgments

- [Tongyi-MAI Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) for the local generation model
- Google Gemini for cloud-based image generation
- Diffusers library for model integration

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check the detailed guides in README_LOCAL.md
- Run `test_setup.py` to diagnose problems
