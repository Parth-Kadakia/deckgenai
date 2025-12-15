# Speed Optimization Guide for Mac

## 🐌 Current Performance Issue

Based on your test results:
- **Current speed**: ~2.5 minutes per inference step
- **Per card (9 steps)**: ~20-25 minutes
- **Full deck (55 cards)**: ~18-23 hours ⚠️

This is too slow for practical use on Mac.

## ⚡ Solutions

### Option 1: Fast Local Generation (Recommended for Testing)

Use reduced quality settings for much faster generation:

```bash
.venv/bin/python batch_generate_fast.py
```

**Fast settings:**
- Image size: 512x896 (vs 640x1152)
- Steps: 4 (vs 9)
- Expected time: ~8-12 minutes per card
- Full deck: ~7-11 hours

**Trade-off:** Lower resolution and quality, but 2-3x faster.

### Option 2: Cloud API (Recommended for Production)

Use the Google Gemini cloud API instead:

```bash
.venv/bin/python app.py
```

**Benefits:**
- Much faster (~30-60 seconds per card)
- Higher quality output
- Full deck: ~30-60 minutes
- No model download needed

**Cost:** Requires API credits (~$0.05-0.10 per card)

### Option 3: Google Colab (FREE GPU)

Use Google Colab's free GPU for fast generation:

1. Go to: https://colab.research.google.com/
2. Upload your scripts
3. Enable GPU: Runtime → Change runtime type → GPU
4. Run generation with CUDA

**Benefits:**
- FREE GPU access
- ~2-3 seconds per card
- Full deck: ~3-5 minutes
- No local setup needed

### Option 4: Overnight Generation

Let your Mac run overnight with standard settings:

```bash
# Start before bed
.venv/bin/python batch_generate.py
```

**Note:** Will take 18-23 hours. Make sure:
- Mac won't sleep (System Settings → Energy)
- Plenty of disk space
- Stable power supply

## Comparison Table

| Method | Time per Card | Full Deck (55) | Quality | Cost |
|--------|---------------|----------------|---------|------|
| Mac MPS (standard) | 20-25 min | 18-23 hours | Good | Free |
| Mac MPS (fast) | 8-12 min | 7-11 hours | Lower | Free |
| Google Colab GPU | 2-3 sec | 3-5 minutes | Good | Free* |
| Cloud API | 30-60 sec | 30-60 minutes | Excellent | $2-5 |

*Free tier has usage limits

## Recommended Workflow

### For Testing/Prototyping:
```bash
# Generate just a few cards to test theme
.venv/bin/python generate_local_mac.py --single-card Ace Hearts
.venv/bin/python generate_local_mac.py --single-card King Spades
.venv/bin/python generate_local_mac.py --card-back
```

### For Production Deck:
**Best option:** Use Google Colab with GPU

1. Create a Colab notebook
2. Install dependencies:
   ```python
   !pip install torch diffusers transformers accelerate pillow
   ```
3. Upload your `batch_generate.py` script
4. Run with GPU enabled

**Second option:** Use cloud API
```bash
.venv/bin/python app.py
```

### For Experimentation (if time isn't critical):
```bash
# Run overnight or over weekend
.venv/bin/python batch_generate.py
```

## Quick Commands Reference

```bash
# Single card (for testing themes) - ~20 min
.venv/bin/python generate_local_mac.py --single-card Ace Hearts

# Fast mode (lower quality) - ~7-11 hours
.venv/bin/python batch_generate_fast.py

# Standard mode (overnight) - ~18-23 hours
.venv/bin/python batch_generate.py

# Cloud API (recommended) - ~30-60 minutes
.venv/bin/python app.py
```

## Google Colab Setup (Step by Step)

1. **Go to Colab:**
   - Visit: https://colab.research.google.com/

2. **Create new notebook:**
   - File → New notebook

3. **Enable GPU:**
   - Runtime → Change runtime type → Hardware accelerator → GPU → Save

4. **Install dependencies:**
   ```python
   !pip install torch diffusers transformers accelerate pillow
   ```

5. **Upload your script:**
   - Use the Files panel (📁 icon on left)
   - Upload `batch_generate.py`

6. **Run generation:**
   ```python
   !python batch_generate.py --output-dir /content/cards
   ```

7. **Download results:**
   - Right-click on output folder → Download

**Time:** ~3-5 minutes for full deck with free GPU!

## Why is Mac So Slow?

1. **MPS (Metal) is still maturing** - Not as optimized as CUDA
2. **Memory bandwidth** - Shared memory between CPU/GPU
3. **Thermal throttling** - Mac may slow down to stay cool
4. **Model size** - 6GB model is large for on-device processing

The Z-Image model is designed for NVIDIA GPUs with CUDA, so Mac performance is not optimal.

## Bottom Line Recommendation

**For your use case, I recommend:**

1. **Test your theme locally** (generate 2-3 cards)
2. **Use Google Colab** for full deck generation
3. **Alternative:** Use cloud API if Colab is too complex

This will save you 18+ hours of waiting! ⚡
