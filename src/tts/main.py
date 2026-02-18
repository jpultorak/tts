import torch
from tqdm import tqdm

import tts.config as config
from tts.infer import TTSInference

CHECKPOINT_PATH = config.ROOT_DIR / "final_models" / "model_epoch_40.pt"
TRAIN_DATA_DIR = config.ROOT_DIR / "test_training_data"
OUTPUT_DIR = config.ROOT_DIR / "output_model_2"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tts = TTSInference(CHECKPOINT_PATH)
    files = sorted(list(TRAIN_DATA_DIR.glob("*.pt")))
    print(f"--> Found {len(files)} samples in {TRAIN_DATA_DIR}")
    print(f"--> Generating audio to {OUTPUT_DIR}\n")

    for pt_file in tqdm(files, desc="Eval"):
        data = torch.load(pt_file)
        file_id = data.get("file_id", pt_file.stem)
        text = data.get("text")

        out_path = OUTPUT_DIR / f"{file_id}_generated.wav"

        print(f"\nGenerating: {file_id}")
        print(f'   Text: "{text}"')

        try:
            tts.generate_audio(
                text=text,
                output_path=str(out_path),
                max_new_tokens=750,
            )
        except Exception as e:
            print(f"Failed to generate {file_id}: {e}")

    print("Done")


def prompt_mode():
    PROMPT_DIR = OUTPUT_DIR / "prompts"
    PROMPT_DIR.mkdir(parents=True, exist_ok=True)

    tts = TTSInference(CHECKPOINT_PATH)
    prompt_idx = 0

    while True:
        text = input("\nEnter text > ").strip()
        if text == "":
            break

        prompt_idx += 1
        sample_dir = PROMPT_DIR / f"prompt_{prompt_idx:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        text_path = sample_dir / "text.txt"
        wav_path = sample_dir / "audio.wav"

        with open(text_path, "w", encoding="utf-8") as f:
            f.write(text)

        try:
            tts.generate_audio(
                text=text,
                output_path=str(wav_path),
                max_new_tokens=750,
            )
        except Exception as e:
            print(f"Failed: {e}")


if __name__ == "__main__":
    # main()
    prompt_mode()
