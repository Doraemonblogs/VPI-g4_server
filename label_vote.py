import json
from collections import Counter

# --- 1. Configuration ---
# File paths for input (scored) and output (labeled)
INPUT_SCORED_PATH = "output/email_data_exfiltration_scored.json"
OUTPUT_LABELED_PATH = "output/email_data_exfiltration_labled.json" # Using your requested filename

# --- 2. Main Execution Logic ---
def apply_majority_vote():
    """
    Loads the scored dataset, applies a majority vote to the 'Scoring' list,
    and saves the final labeled dataset.
    """
    try:
        with open(INPUT_SCORED_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading scored file '{INPUT_SCORED_PATH}': {e}")
        return

    print(f"Loaded {len(data)} items. Applying majority vote...")

    labeled_count = 0
    for item in data:
        # Check if the 'Scoring' field is valid
        if "Scoring" in item and isinstance(item["Scoring"], list) and len(item["Scoring"]) == 3:
            # Count True vs. False votes.
            vote_counts = Counter(item["Scoring"])
            # The final label is True if 'True' votes are 2 or more.
            success_label = vote_counts.get(True, 0) >= 2
            item["success"] = success_label
            labeled_count += 1
        else:
            # If scoring data is missing or malformed, set success to None
            item["success"] = None
            print(f"Warning: Item ID {item.get('id', 'N/A')} has an invalid 'Scoring' field. Skipping.")
    
    print(f"Applied labels to {labeled_count} items.")

    # Save the final labeled dataset
    with open(OUTPUT_LABELED_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Labeling complete. Final dataset saved to '{OUTPUT_LABELED_PATH}'")


if __name__ == "__main__":
    apply_majority_vote()