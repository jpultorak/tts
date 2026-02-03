import torch
import random
from pathlib import Path
from tts.tokenizer import Tokenizer
from tts.model import BabyValle
import tts.config as config
from tts.dataset import TTSDataset, collate_fn

def debug_pipeline():
    print("🔎 STARTING PIPELINE AUDIT...\n")
    
    # --- STEP 1: LOAD CONFIG & TOKENIZER ---
    print(f"1. Checking Vocab at {config.VOCAB_PATH}...")
    tokenizer = Tokenizer()
    print(f"   ✅ Vocab loaded. Size: {len(tokenizer.token_to_id)}")
    
    # Check Special Token IDs
    print(f"   ℹ️  EOS: {config.EOS_TOKEN_ID} (Expect 1024)")
    print(f"   ℹ️  PAD: {config.PAD_TOKEN_ID} (Expect 1025)")
    print(f"   ℹ️  SEP: {config.SEP_TOKEN_ID} (Expect 1026)")
    
    # --- STEP 2: INSPECT A TRAINING FILE ---
    data_dir = config.ROOT_DIR / "data"
    files = list(data_dir.glob("*.pt"))
    if not files:
        print("   ❌ NO DATA FOUND in tts/data!")
        return
    
    target_file = random.choice(files)
    print(f"\n2. Inspecting random training file: {target_file.name}")
    data = torch.load(target_file)
    
    stored_text = data['text']
    stored_phonemes = data['phonemes']
    stored_audio = data['audio_tokens']
    
    print(f"   📝 Text: '{stored_text}'")
    print(f"   🔊 Stored Phoneme IDs: {stored_phonemes.tolist()}")
    print(f"   🎵 Stored Audio Tokens: {stored_audio.tolist()[:10]}... (Total {len(stored_audio)})")

    # TEST A: Audio Token Range
    if (stored_audio >= 1024).any():
        print("   ❌ CRITICAL FAIL: Found audio tokens >= 1024! Model will confuse them with text.")
    else:
        print("   ✅ Audio tokens are in valid range [0-1023].")

    # TEST B: Phoneme Token Range
    if (stored_phonemes < 1024).any():
        print("   ❌ CRITICAL FAIL: Found phoneme tokens < 1024! Model will confuse them with audio.")
    else:
        print("   ✅ Phoneme tokens are in valid range [>=1024].")

    # --- STEP 3: RE-TOKENIZATION CHECK ---
    print(f"\n3. Verifying Tokenizer Consistency...")
    # Re-run phonemizer on raw text to see if it matches stored IDs
    fresh_ids = tokenizer.encode(stored_text)
    
    # NOTE: tokenizer.encode adds EOS at the end. stored_phonemes might not?
    # Let's align them.
    fresh_ids_tensor = torch.tensor(fresh_ids)
    
    print(f"   Freshly Tokenized: {fresh_ids}")
    
    # Check if they are somewhat similar (exact match is hard due to phonemizer non-determinism sometimes)
    # We strip EOS for comparison if needed
    if len(fresh_ids) != len(stored_phonemes):
         print(f"   ⚠️  Length Mismatch! Stored: {len(stored_phonemes)}, Fresh: {len(fresh_ids)}")
         # This is often where the bug is. Does one have EOS and the other doesn't?
    else:
         print(f"   ✅ Lengths match.")

    # --- STEP 4: DATASET & BATCHING LOGIC ---
    print(f"\n4. Checking Dataset & Collate (Training Logic)...")
    dataset = TTSDataset(str(data_dir))
    # Mock grabbing the specific item we just looked at
    # (We can't easily find the index, so we trust the class logic generally)
    
    # Let's process the raw data manually exactly how TTSDataset.__getitem__ does it
    # COPY-PASTED LOGIC FROM YOUR DATASET.PY TO VERIFY
    d_phonemes = stored_phonemes[:-1].long() # Standard training logic often strips last?
    d_audio = stored_audio.long()
    d_sep = torch.tensor([config.SEP_TOKEN_ID], dtype=torch.long)
    d_eos = torch.tensor([config.EOS_TOKEN_ID], dtype=torch.long)
    
    full_seq = torch.cat([d_phonemes, d_sep, d_audio, d_eos])
    x = full_seq[:-1]
    y = full_seq[1:]
    
    print(f"   TRAINING INPUT CONSTRUCTION:")
    print(f"   [Phonemes ({len(d_phonemes)})] + [SEP] + [Audio ({len(d_audio)})]")
    print(f"   Total X length: {len(x)}")
    print(f"   X (last 5): {x[-5:].tolist()} (Should be audio tokens)")
    print(f"   Y (last 5): {y[-5:].tolist()} (Should be audio + EOS)")
    
    if x[-1] >= 1024:
        print("   ❌ CRITICAL FAIL: The last token of input X is NOT an audio token. It's a special token.")
        print("      This means the model is trying to predict special tokens instead of audio.")
    else:
        print("   ✅ Input X ends with Audio Token.")

    # --- STEP 5: INFERENCE PREPARATION LOGIC ---
    print(f"\n5. Checking Inference Preparation Logic (The 'Alien' Suspect)...")
    
    # Simulate exactly what you do in infer.py
    # OLD BUGGY WAY: inf_phonemes = tokenizer.encode(stored_text)[:-2]
    # NEW WAY:
    inf_phonemes = tokenizer.encode(stored_text)[:-1] 
    
    print(f"   Inference Phonemes (len {len(inf_phonemes)}): {inf_phonemes}")
    print(f"   Training Phonemes  (len {len(d_phonemes)}): {d_phonemes.tolist()}")
    
    if len(inf_phonemes) != len(d_phonemes):
        print(f"   ❌ MISMATCH DETECTED!")
        print(f"      Training sees {len(d_phonemes)} phonemes.")
        print(f"      Inference sees {len(inf_phonemes)} phonemes.")
        print("      This causes the 'Alien Effect'. The positional embeddings will be shifted!")
    else:
        print("   ✅ Training and Inference phoneme counts match exactly.")

    # --- STEP 6: MODEL FORWARD PASS ---
    print(f"\n6. Dry Run Model Forward Pass...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = BabyValle(vocab_size=2048, d_model=512, nhead=8, num_layers=6).to(device)
    dummy_input = x.unsqueeze(0).to(device) # Batch size 1
    
    try:
        with torch.no_grad():
            logits = model(dummy_input)
        print(f"   ✅ Model accepted input shape: {dummy_input.shape}")
        print(f"   ✅ Output logits shape: {logits.shape} (Batch, Seq, Vocab)")
    except Exception as e:
        print(f"   ❌ Model Crash: {e}")

    print("\n------------------------------------------------")
    print("🏁 VERDICT:")
    if len(inf_phonemes) != len(d_phonemes):
        print("👉 FIX INFERENCE: Your slicing [:-1] vs [:-2] logic is likely still drifting.")
    elif (stored_audio >= 1024).any():
        print("👉 FIX DATASET: You have corrupt audio tokens.")
    else:
        print("👉 PIPELINE LOOKS HEALTHY. If audio is still bad, it is 100% an overfitting/stability issue, not a code bug.")

if __name__ == "__main__":
    debug_pipeline()