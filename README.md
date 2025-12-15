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
   pip install -r requirements_local.txt
   ```

2. **Test your setup:**
   ```bash
   python test_setup.py
   ```

3. **Generate a single card:**
   ```bash
   python generate_local.py --single-card Ace Hearts
   ```

4. **Generate full deck:**
   ```bash
   python generate_local.py --full-deck
   ```

## Project Structure

```
carddeckgen/
├── app.py                    # Cloud generation (Google Gemini)
├── generate_local.py         # Local generation (Z-Image-Turbo)
├── test_setup.py            # Setup verification script
├── requirements_local.txt   # Local generation dependencies
├── README.md                # This file
├── README_LOCAL.md          # Detailed local generation guide
├── deck.jsonl               # Deck configuration
├── playing-cards-batch.jsonl # Batch processing config
├── card_images/             # Generated cards (cloud)
├── card_images_local/       # Generated cards (local)
└── batch_output/            # Batch processing results
```

## Usage Examples

### Local Generation

```bash
# Single card
python generate_local.py --single-card King Spades

# Card back
python generate_local.py --card-back

# Full deck with custom theme
python generate_local.py --full-deck \
  --theme "Cyberpunk" \
  --technique "neon digital art" \
  --background "dark holographic"

# With performance optimizations
python generate_local.py --full-deck --flash-attention --compile
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

| Metric | Local (GPU) | Cloud (API) |
|--------|-------------|-------------|
| Speed per card | ~2-3 seconds | ~30-60 seconds |
| Quality | High | Very High |
| Cost | Free* | $$ per card |
| Setup complexity | Medium | Easy |
| Requirements | GPU required | API key only |

*After initial GPU purchase

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
