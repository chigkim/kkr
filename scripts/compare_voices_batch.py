import torch
import argparse
from pathlib import Path
import itertools

def compare_all_voices(voices_dir):
    voices_path = Path(voices_dir)
    voice_files = sorted(list(voices_path.glob("*.pt")))

    if len(voice_files) < 2:
        print(f"Error: Found {len(voice_files)} voicepacks in {voices_dir}. Need at least 2 to compare.")
        return

    print(f"\nComparing {len(voice_files)} voicepacks in {voices_dir}...")
    print(f"{'Voice A':<20} | {'Voice B':<20} | {'Acoustic Sim':<12} | {'Prosodic Sim':<12}")
    print("-" * 75)

    # Load all style vectors once to be efficient
    data = {}
    for f in voice_files:
        try:
            vp = torch.load(f, map_location='cpu', weights_only=True)
            # VP shape [510, 1, 256] -> style [256]
            style = vp[0, 0]
            data[f.name] = (style[:128], style[128:])
        except Exception as e:
            print(f"Skipping {f.name}: {e}")

    # Pairwise comparison
    for (name1, (a1, p1)), (name2, (a2, p2)) in itertools.combinations(data.items(), 2):
        sim_a = torch.nn.functional.cosine_similarity(a1.unsqueeze(0), a2.unsqueeze(0)).item()
        sim_p = torch.nn.functional.cosine_similarity(p1.unsqueeze(0), p2.unsqueeze(0)).item()
        
        # High similarity warning
        flag = "!" if sim_a > 0.99 else " "
        
        print(f"{name1[:20]:<20} | {name2[:20]:<20} | {sim_a:12.4f} {flag} | {sim_p:12.4f}")

    print("-" * 75)
    print("(!) = Voices are nearly identical (Similarity > 0.99)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare all Kokoro voicepacks in a directory.")
    parser.add_argument("voices_dir", nargs="?", default="voices", help="Directory containing .pt voicepacks")
    args = parser.parse_args()

    compare_all_voices(args.voices_dir)
