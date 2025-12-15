# Batch Generation Guide

Generate all 55 cards at once (52 standard cards + 2 jokers + 1 card back).

## Quick Start

### Recommended: Batch Generation with Progress Tracking

```bash
.venv/bin/python batch_generate.py
```

This will:
- ✅ Generate all 55 cards automatically
- ✅ Show real-time progress bar
- ✅ Display ETA for completion
- ✅ Track success/failure for each card
- ✅ Save results summary to `batch_results.json`
- ✅ Handle errors gracefully

### Alternative: Simple Full Deck Generation

```bash
.venv/bin/python generate_local_mac.py --full-deck
```

## What Gets Generated

| Type | Count | Examples |
|------|-------|----------|
| Standard Cards | 52 | Ace-King of Hearts, Spades, Diamonds, Clubs |
| Jokers | 2 | Joker 1, Joker 2 |
| Card Back | 1 | Universal back design |
| **Total** | **55** | Complete playable deck |

## Output Files

Cards are saved with this naming convention:

```
card_images_local/
├── Hearts_01_Ace.png
├── Hearts_02_2.png
├── ...
├── Spades_13_King.png
├── ZZ_Joker_14_Joker1.png
├── ZZ_Joker_15_Joker2.png
├── ZZ_ZZ_00_Card-Back.png
└── batch_results.json         # Generation statistics
```

## Example Output (Progress Tracking)

```
🎨 BATCH CARD GENERATION
═══════════════════════════════════════════════════════════
Theme: Western Steampunk
Output: card_images_local
Device: MPS
Total Cards: 55 (52 standard + 2 jokers + 1 back)
═══════════════════════════════════════════════════════════

📋 Generating standard cards (52)...
[████████████░░░░░░░░░░░░] 32.7% | 18/55 | 2 of Hearts | 12.34s | ETA: 8m 23s
```

## Customization Examples

### Cyberpunk Deck
```bash
.venv/bin/python batch_generate.py \
  --theme "Cyberpunk" \
  --technique "neon digital art" \
  --background "dark holographic"
```

### Art Nouveau Deck
```bash
.venv/bin/python batch_generate.py \
  --theme "Art Nouveau" \
  --technique "flowing organic lines" \
  --background "cream canvas"
```

### Japanese Ukiyo-e Deck
```bash
.venv/bin/python batch_generate.py \
  --theme "Japanese Ukiyo-e" \
  --technique "woodblock print" \
  --background "rice paper"
```

### Fantasy Medieval Deck
```bash
.venv/bin/python batch_generate.py \
  --theme "Fantasy Medieval" \
  --technique "illuminated manuscript" \
  --background "vellum parchment"
```

## Command Line Options

```bash
.venv/bin/python batch_generate.py [OPTIONS]

Options:
  --output-dir DIR        Output directory (default: card_images_local)
  --theme THEME          Visual theme (default: Western Steampunk)
  --technique TECH       Art technique (default: Victorian engraving)
  --background BG        Background texture (default: aged parchment)
  --height H             Image height (default: 1152)
  --width W              Image width (default: 640)
  --steps N              Inference steps (default: 9)
  --device DEVICE        cuda/mps/cpu (default: auto-detect)
```

## Time Estimates

| Device | Time per Card | Total Time (55 cards) |
|--------|---------------|----------------------|
| CUDA GPU | ~2-3 seconds | ~2-3 minutes |
| Apple Silicon (MPS) | ~10-30 seconds | ~10-25 minutes |
| Intel Mac (CPU) | ~2-5 minutes | ~2-4 hours ⚠️ |

## Features

### Progress Tracking
- Real-time progress bar
- Cards completed counter
- Current card being generated
- Time per card
- Estimated time remaining

### Error Handling
- Continues if individual cards fail
- Tracks failed cards
- Saves partial results
- Shows summary at end

### Results Summary
The `batch_results.json` file contains:
```json
{
  "timestamp": "2025-12-14T20:15:30",
  "theme": "Western Steampunk",
  "device": "mps",
  "total_cards": 55,
  "completed": 55,
  "failed": 0,
  "total_time": "18m 45s",
  "avg_time_per_card": "20.45s",
  "cards": [
    {
      "card": "Ace of Hearts",
      "filename": "Hearts_01_Ace.png",
      "status": "success",
      "time": "19.23s"
    },
    ...
  ]
}
```

## Interrupting Generation

Press `Ctrl+C` to stop generation. Progress will be saved and partial results will remain in the output directory.

## Tips for Success

1. **Check disk space:** 55 cards × ~2-5MB = ~110-275MB needed
2. **Close other apps:** Free up GPU/CPU resources
3. **Use consistent settings:** Same theme/technique for cohesive deck
4. **Start with single card:** Test your theme before full batch
5. **Monitor progress:** Watch for any failed generations

## Comparison: Batch vs Individual

| Method | Use Case | Progress | Resume | Results |
|--------|----------|----------|--------|---------|
| `batch_generate.py` | Full deck | ✅ Real-time | ❌ No | ✅ JSON summary |
| `--full-deck` flag | Full deck | ❌ Basic | ❌ No | ❌ No summary |
| `--single-card` | Testing/Individual | N/A | N/A | N/A |

**Recommendation:** Use `batch_generate.py` for full decks - it provides better visibility and tracking.

## Troubleshooting

### Generation is too slow
- Use `--steps 4` for faster (lower quality) generation
- Reduce image size: `--height 896 --width 512`
- Consider cloud API for better speed

### Out of memory
- Close other applications
- Reduce `--height` and `--width`
- Restart and try again

### Some cards failed
- Check `batch_results.json` for failed cards
- Re-run with `--single-card` for specific failed cards
- Verify disk space available

## Next Steps After Generation

1. **Review the cards** in your output directory
2. **Check batch_results.json** for any failures
3. **Re-generate failed cards** individually if needed
4. **Use the cards** in your project!

Need higher quality? Consider using the cloud API (`app.py`) for professional-grade output.
