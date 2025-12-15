import json
import time
import base64
from pathlib import Path
from google import genai
from google.genai import types

# Directories
OUTPUT_RESULTS_DIR = Path("batch_output")
OUTPUT_IMAGES_DIR = Path("card_images")

OUTPUT_RESULTS_DIR.mkdir(exist_ok=True)
OUTPUT_IMAGES_DIR.mkdir(exist_ok=True)

client = genai.Client()

SUITS = ["Hearts", "Spades", "Diamonds", "Clubs"]
VALUES = ["Ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King"]

# Numeric order for file naming (so files sort properly)
VALUE_ORDER = {
    "Ace": "01", "2": "02", "3": "03", "4": "04", "5": "05",
    "6": "06", "7": "07", "8": "08", "9": "09", "10": "10",
    "Jack": "11", "Queen": "12", "King": "13"
}


# ---------------------------------------------------------
# CONSISTENT STYLE PROMPT (locked deck style)
# ---------------------------------------------------------

# def build_prompt(value, suit, theme, technique, background):
#     suit_symbol = "♥" if suit in ["hearts", "heart"] else "♦" if suit in ["diamonds", "diamond"] else "♣" if suit in [
#         "clubs", "club"] else "♠"
#
#     # logic to handle number vs face cards for layout consistency
#     is_face_card = value.lower() in ['k', 'q', 'j', 'king', 'queen', 'jack']
#
#     layout_instruction = ""
#     if is_face_card:
#         layout_instruction = f"single central portrait of a character representing {value}, framed inside the central oval."
#     else:
#         layout_instruction = f"symmetrical arrangement of {value} distinct items representing the suit, strictly arranged in a grid or radial pattern."
#
#     return f"""
# **type:** flat 2d production asset / texture scan.
# **subject:** playing card design: {value} of {suit}.
# **theme:** {theme}.
#
# **strict layout rules (do not deviate):**
# - **canvas:** the image must show the full rectangular card on a plain white background (allow for bleed margin).
# - **card ratio:** 2.5 x 3.5 vertical aspect.
# - **card surface:** {background}.
# - **outer border:** a solid, continuous dark brown rectangular line exactly 5% from the edge.
# - **indices:** the letter "{value}" and symbol "{suit_symbol}" must be printed in the top-left and bottom-right corners using a standard serif font.
# - **inner frame:** a precise oval vignette frame centered in the middle of the card.
#
# **artwork content (technique: {technique}):**
# - **center:** {layout_instruction}
# - **visuals:** the imagery must strictly adhere to the {theme} theme.
# - **details:** high contrast, clean lines, suitable for commercial printing.
# - **colors:** restrict palette to {theme} specific colors, keeping the background texture visible.
#
# **negative prompt:**
# - no perspective, no drop shadows, no 3d rendering, no photorealism, no curved card edges, no objects overlapping the text indices, no text spelling mistakes.
# """
# def build_prompt(value, suit, theme, technique, background):
#     suit_symbol = "♥" if suit in ["Hearts", "Heart"] else "♦" if suit in ["Diamonds", "Diamond"] else "♣" if suit in [
#         "Clubs", "Club"] else "♠"
#
#     # helper to ensure we use single letters for consistency if possible,
#     # or keep full names if you strictly require them (though full names cause font variance).
#     # For best consistency, we ask the prompt to use the standard abbreviation.
#     idx = value[0].upper() if str(value).lower() in ['jack', 'king', 'queen', 'ace'] else value
#
#     return f"""
# **FORMAT:** Digital 2D Vector Graphic / UI Asset / Poker Card Template.
# **SUBJECT:** {value} of {suit} ({theme} Theme).
#
# **STRICT TEMPLATE STRUCTURE (DO NOT CHANGE):**
# - **Canvas:** Vertical 9:16 aspect ratio.
# - **Background Texture:** {background}, applied uniformly across the entire card surface.
# - **Outer Frame:** A thick, dark, solid chocolate-brown border (fixed width 40px) surrounding the edges.
# - **Inner Frame:** A perfectly centered VERTICAL OVAL MEDALLION with a gold rim.
# - **Typography:** The text "{idx}" and symbol "{suit_symbol}" placed in Top-Left and Bottom-Right corners.
# - **Font Style:** Bold Western Slab-Serif font. White text color. EXACTLY IDENTICAL SIZE for every card.
#
# **CENTRAL ARTWORK (Inside the Oval):**
# - **Technique:** {technique}.
# - **Content:** A symmetrical illustration of {value} ({theme}).
# - **Composition:** The artwork must stay STRICTLY inside the central oval frame. Do not let art bleed into the corners.
# - **Details:** {theme} visual elements, clean lines, professional illustration.
#
# **NEGATIVE PROMPT:**
# - irregular borders, varying font sizes, blurry text, messy typography, 3d render, perspective tilt, crooked lines, art spilling over border, photorealistic photography.
# """

def build_prompt(value, suit, theme, technique, background):
    # Logic: Face cards = Portraits / Characters. Number cards = Symmetrical Clusters.
    is_face_card = str(value).lower() in ['k', 'q', 'j', 'king', 'queen', 'jack']

    if is_face_card:
        # Focus on a character bust/portrait
        subject = f"A majestic portrait of a character representing the {value} of {suit}"
        composition = "centered character bust, facing forward, vertical composition"
    else:
        # Focus on a decorative arrangement of objects
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
# ---------------------------------------------------------
# Build card back prompt (standard for all cards)
# ---------------------------------------------------------
def build_card_back_prompt(theme, technique, background):
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
- playing card front, face cards, numbers, suit symbols (hearts, spades, diamonds, clubs), asymmetrical design, text, letters, portraits, faces, 3d render, watermark.
"""


# ---------------------------------------------------------
# Build all requests
# ---------------------------------------------------------
def build_requests(theme: str, technique: str, background: str):
    requests = []

    # Generate 52 standard cards (4 suits × 13 values)
    for suit in SUITS:
        for value in VALUES:
            prompt = build_prompt(value, suit, theme, technique, background)
            num = VALUE_ORDER.get(value, "00")
            key = f"{suit}_{num}_{value}"

            requests.append({
                "key": key,
                "request": {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generation_config": {"responseModalities": ["TEXT", "IMAGE"]}
                }
            })

    # Add 2 Jokers
    for i in range(1, 3):
        prompt = build_prompt("Joker", "Wild", theme, technique, background)
        requests.append({
            "key": f"ZZ_Joker_{13 + i:02d}_Joker{i}",
            "request": {
                "contents": [{"parts": [{"text": prompt}]}],
                "generation_config": {"responseModalities": ["TEXT", "IMAGE"]}
            }
        })

    # Add card back design request (ZZ prefix so it appears last)
    card_back_prompt = build_card_back_prompt(theme, technique, background)
    requests.append({
        "key": "ZZ_ZZ_00_Card-Back",
        "request": {
            "contents": [{"parts": [{"text": card_back_prompt}]}],
            "generation_config": {"responseModalities": ["TEXT", "IMAGE"]}
        }
    })

    return requests


# ---------------------------------------------------------
# Write JSONL
# ---------------------------------------------------------
def write_jsonl(requests, filename="deck.jsonl"):
    path = Path(filename)
    with open(path, "w", encoding="utf-8") as f:
        for r in requests:
            f.write(json.dumps(r) + "\n")
    return path


# ---------------------------------------------------------
# Build retry requests for failed keys
# ---------------------------------------------------------
def build_retry_requests(failed_keys, theme, technique, background):
    requests = []
    
    for key in failed_keys:
        # Parse the key to determine what card it is
        if "Card-Back" in key:
            prompt = build_card_back_prompt(theme, technique, background)
        elif "Joker" in key:
            prompt = build_prompt("Joker", "Wild", theme, technique, background)
        else:
            # Parse key like "Hearts_01_Ace" or "Clubs_06_6"
            parts = key.split("_")
            if len(parts) >= 3:
                suit = parts[0]
                value = parts[2]
                prompt = build_prompt(value, suit, theme, technique, background)
            else:
                print(f"⚠️  Cannot parse key: {key}, skipping")
                continue
        
        requests.append({
            "key": key,
            "request": {
                "contents": [{"parts": [{"text": prompt}]}],
                "generation_config": {"responseModalities": ["TEXT", "IMAGE"]}
            }
        })
    
    return requests


# ---------------------------------------------------------
# Submit Batch Job
# ---------------------------------------------------------
def run_batch(jsonl_path):
    start_time = time.time()
    uploaded = client.files.upload(
        file=str(jsonl_path),
        config=types.UploadFileConfig(display_name=jsonl_path.name, mime_type="jsonl")
    )
    print("Uploaded:", uploaded.name)

    job = client.batches.create(
        model="gemini-2.5-flash-image",
        src=uploaded.name,
        config={"display_name": f"card-deck-{jsonl_path.stem}"}
    )
    print("Created job:", job.name)

    # Poll
    done = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"}
    status = client.batches.get(name=job.name)
    while status.state.name not in done:
        print("Current state:", status.state.name)
        print(f"Elapsed: {int(time.time() - start_time)}s")
        time.sleep(10)
        status = client.batches.get(name=job.name)

    print(f"Total processing time: {int(time.time() - start_time)}s")
    print("Final state:", status.state.name)
    return status, job.name


# ---------------------------------------------------------
# Download and process results (without saving heavy JSONL)
# ---------------------------------------------------------
def download_results(batch_job):
    if batch_job.state.name != "JOB_STATE_SUCCEEDED":
        print("Batch job failed.")
        return None

    file_name = batch_job.dest.file_name
    bytes_content = client.files.download(file=file_name)
    text = bytes_content.decode("utf-8")
    
    # Return the text directly instead of saving to file
    # This avoids storing large base64 data on disk
    print("Downloaded results from Google (not saving to disk)")
    
    return text


# ---------------------------------------------------------
# Extract Base64 Images & Calculate Cost
# ---------------------------------------------------------
def extract_images(results_text, job_name, expected_keys=None):
    # Create subfolder using job name (extract just the ID part)
    # job.name format is typically "batches/xxxxx" so we extract the ID
    job_id = job_name.split("/")[-1] if "/" in job_name else job_name
    output_folder = OUTPUT_IMAGES_DIR / job_id
    output_folder.mkdir(exist_ok=True)
    
    print(f"\n📁 Saving images to: {output_folder}")
    
    # Cost tracking
    total_prompt_tokens = 0
    total_candidates_tokens = 0
    total_images = 0
    failed_keys = []
    successful_keys = set()
    
    # Process results text line by line
    for line in results_text.strip().split("\n"):
        if not line.strip():
            continue

        obj = json.loads(line)
        key = obj.get("key")

        if "response" not in obj:
            print(f"❌ No response for {key}")
            failed_keys.append(key)
            continue
        
        response = obj["response"]
        
        # Check for error in response
        if "error" in response:
            print(f"❌ Error for {key}: {response['error']}")
            failed_keys.append(key)
            continue
        
        # Check if candidates exist
        if "candidates" not in response or not response["candidates"]:
            print(f"❌ No candidates for {key}")
            failed_keys.append(key)
            continue

        # Extract usage metadata for cost calculation
        if "usageMetadata" in response:
            usage = response["usageMetadata"]
            total_prompt_tokens += usage.get("promptTokenCount", 0)
            total_candidates_tokens += usage.get("candidatesTokenCount", 0)

        try:
            parts = response["candidates"][0]["content"]["parts"]
            image_found = False
            
            for p in parts:
                if "inlineData" in p:
                    data = base64.b64decode(p["inlineData"]["data"])
                    mime = p["inlineData"]["mimeType"]
                    ext = ".png" if "png" in mime else ".jpg"

                    out = output_folder / f"{key}{ext}"
                    with open(out, "wb") as img:
                        img.write(data)
                        
                        total_images += 1
                        successful_keys.add(key)
                        image_found = True
                        print(f"✅ Saved: {out.name}")
            
            if not image_found:
                print(f"❌ No image data for {key}")
                failed_keys.append(key)
                
        except (KeyError, IndexError) as e:
            print(f"❌ Parse error for {key}: {e}")
            failed_keys.append(key)
    
    # Check for missing keys (requests that weren't in results at all)
    if expected_keys:
        for key in expected_keys:
            if key not in successful_keys and key not in failed_keys:
                print(f"❌ Missing from results: {key}")
                failed_keys.append(key)
    
    # Calculate costs
    # Token cost: $0.15 per 1 million tokens
    # Image cost: $0.0195 per image
    total_tokens = total_prompt_tokens + total_candidates_tokens
    token_cost = (total_tokens / 1_000_000) * 0.15
    image_cost = total_images * 0.0195
    total_cost = token_cost + image_cost
    
    # Print cost summary
    print(f"\n{'─' * 60}")
    print(f"  💰 COST SUMMARY")
    print(f"{'─' * 60}")
    print(f"  Prompt Tokens:      {total_prompt_tokens:,}")
    print(f"  Candidates Tokens:  {total_candidates_tokens:,}")
    print(f"  Total Tokens:       {total_tokens:,}")
    print(f"  Images Generated:   {total_images}")
    if failed_keys:
        print(f"  ⚠️  Failed/Missing:   {len(failed_keys)}")
    print(f"{'─' * 60}")
    print(f"  Token Cost:         ${token_cost:.4f}")
    print(f"  Image Cost:         ${image_cost:.4f}")
    print(f"  ─────────────────────────────")
    print(f"  TOTAL COST:         ${total_cost:.4f}")
    print(f"{'─' * 60}")
    
    return output_folder, failed_keys


# ---------------------------------------------------------
# TECHNIQUE & BACKGROUND OPTIONS
# ---------------------------------------------------------
TECHNIQUES = {
    "1": ("Vintage Lithograph", "vintage lithograph with rich earthy tones and fine crosshatching"),
    "2": ("Watercolor", "loose watercolor illustration with soft edges and color bleeds"),
    "3": ("Digital Ink", "clean digital ink illustration with bold outlines and flat colors"),
    "4": ("Woodcut Engraving", "traditional woodcut engraving with dramatic black and white contrast"),
    "5": ("Art Nouveau", "elegant Art Nouveau style with flowing organic lines and decorative flourishes"),
    "6": ("Art Deco", "geometric Art Deco style with bold shapes, gold accents, and symmetrical patterns"),
    "7": ("Japanese Ukiyo-e", "traditional Japanese ukiyo-e woodblock print style with flat colors and bold outlines"),
    "8": ("Stained Glass", "medieval stained glass window style with bold black outlines and jewel-tone colors"),
    "9": ("Illuminated Manuscript", "ornate illuminated manuscript style with gold leaf details and intricate borders"),
    "10": ("Pop Art", "bold pop art style with halftone dots, primary colors, and comic book aesthetics"),
    "11": ("Steampunk", "detailed steampunk illustration with brass gears, Victorian machinery, and sepia tones"),
    "12": ("Pixel Art", "retro pixel art style with limited color palette and crisp 16-bit aesthetic"),
    "13": ("Chalk Pastel", "soft chalk pastel illustration with textured strokes and blended colors"),
    "14": ("Etching", "detailed copper plate etching with fine lines and crosshatch shading"),
    "15": ("Risograph", "modern risograph print style with limited spot colors and slight misregistration"),
    "16": ("Tarot Card", "mystical tarot card illustration with symbolic imagery and ornate gold details"),
}

BACKGROUNDS = {
    "1": ("Parchment", "vintage aged parchment paper with subtle tea stains and worn edges"),
    "2": ("Clean White", "pristine bright white premium card stock"),
    "3": ("Dark Marble", "luxurious dark marbled stone texture with gold veining"),
    "4": ("Soft Gradient", "smooth subtle neutral gradient from cream to white"),
    "5": ("Velvet Black", "rich deep black velvet texture"),
    "6": ("Crimson Red", "deep royal crimson red with subtle fabric texture"),
    "7": ("Forest Green", "classic forest green felt texture like a poker table"),
    "8": ("Navy Blue", "sophisticated navy blue with subtle linen texture"),
    "9": ("Gold Foil", "shimmering brushed gold foil metallic surface"),
    "10": ("Silver Metallic", "sleek brushed silver metallic surface"),
    "11": ("Kraft Paper", "natural brown kraft paper with visible fibers"),
    "12": ("Starry Night", "deep cosmic blue-black with scattered tiny stars"),
    "13": ("Wood Grain", "warm polished mahogany wood grain texture"),
    "14": ("Concrete", "modern industrial concrete texture with subtle variations"),
    "15": ("Watercolor Wash", "soft abstract watercolor wash in muted tones"),
    "16": ("Leather", "rich embossed leather texture in burgundy"),
}


def display_options(options_dict, title):
    """Display options in a nice formatted grid."""
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")
    
    items = list(options_dict.items())
    # Display in two columns
    half = (len(items) + 1) // 2
    for i in range(half):
        left = f"  {items[i][0]:>2}) {items[i][1][0]:<25}"
        right = ""
        if i + half < len(items):
            right = f"  {items[i + half][0]:>2}) {items[i + half][1][0]}"
        print(f"{left}{right}")
    print()


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    print("\n" + "═" * 60)
    print("  🎴  PLAYING CARD DECK GENERATOR  🎴")
    print("═" * 60)
    
    theme = input("\n🎨 Enter theme for your deck (e.g., Pirates, Cyberpunk, Medieval, Cats): ").strip()
    if theme == "":
        theme = "Baseball"
    
    # Technique selection
    display_options(TECHNIQUES, "✏️  ART TECHNIQUES")
    tech_choice = input("Select technique (1-16): ").strip()
    if tech_choice not in TECHNIQUES:
        tech_choice = "1"
    technique = TECHNIQUES[tech_choice][1]
    print(f"  → Selected: {TECHNIQUES[tech_choice][0]}")
    
    # Background selection
    display_options(BACKGROUNDS, "🖼️  BACKGROUND STYLES")
    bg_choice = input("Select background (1-16): ").strip()
    if bg_choice not in BACKGROUNDS:
        bg_choice = "1"
    background = BACKGROUNDS[bg_choice][1]
    print(f"  → Selected: {BACKGROUNDS[bg_choice][0]}")
    
    print(f"\n{'─' * 60}")
    print(f"  📋 SUMMARY")
    print(f"{'─' * 60}")
    print(f"  Theme:      {theme}")
    print(f"  Technique:  {TECHNIQUES[tech_choice][0]}")
    print(f"  Background: {BACKGROUNDS[bg_choice][0]}")
    print(f"  Cards:      54 fronts + 1 card back design")
    print(f"{'─' * 60}\n")
    
    confirm = input("🚀 Ready to generate? (y/n): ").strip().lower()
    if confirm != 'y' and confirm != 'yes' and confirm != '':
        print("Cancelled.")
        return

    requests = build_requests(theme, technique, background)
    expected_keys = [r["key"] for r in requests]
    jsonl = write_jsonl(requests)

    batch_job, job_name = run_batch(jsonl)
    if batch_job.state.name == "JOB_STATE_SUCCEEDED":
        results = download_results(batch_job)
        output_folder, failed_keys = extract_images(results, job_name, expected_keys)
        
        # Retry loop for failed cards
        retry_count = 0
        max_retries = 3
        
        while failed_keys and retry_count < max_retries:
            retry_count += 1
            print(f"\n{'═' * 60}")
            print(f"  ⚠️  {len(failed_keys)} cards failed. Retry {retry_count}/{max_retries}")
            print(f"{'═' * 60}")
            print(f"  Failed cards: {', '.join(failed_keys)}")
            
            retry = input(f"\n🔄 Retry failed cards? (y/n): ").strip().lower()
            if retry != 'y' and retry != 'yes':
                break
            
            # Build retry requests
            retry_requests = build_retry_requests(failed_keys, theme, technique, background)
            retry_jsonl = write_jsonl(retry_requests, f"deck_retry_{retry_count}.jsonl")
            
            retry_batch, retry_job_name = run_batch(retry_jsonl)
            if retry_batch.state.name == "JOB_STATE_SUCCEEDED":
                retry_results = download_results(retry_batch)
                # Extract to same folder, only pass the failed keys as expected
                _, failed_keys = extract_images(retry_results, job_name, failed_keys)
        
        if failed_keys:
            print(f"\n⚠️  Warning: {len(failed_keys)} cards could not be generated:")
            for key in failed_keys:
                print(f"     - {key}")
        
        print(f"\n🎉 Done! Cards saved in {output_folder}\n")
    else:
        print("\n❌ Batch job failed. No images saved.\n")


if __name__ == "__main__":
    main()