from tqdm import tqdm

import tts.config as config
from tts.infer import TTSInference

MODEL_PATH = config.ROOT_DIR / "final_models" / "model_epoch_120.pt"
INPUT_FILE = config.ROOT_DIR / "input.txt"
OUTPUT_DIR = config.ROOT_DIR / "output_model_2"


def process_input():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        raise ValueError("Empty input file")

    print(f"--> Found {len(lines)} sentences in {INPUT_FILE}")

    tts = TTSInference(MODEL_PATH)

    for i, text in enumerate(tqdm(lines, desc="Processing")):
        file_name = f"{i:03d}.wav"
        out_path = OUTPUT_DIR / file_name

        try:
            tts.generate_audio(
                text=text,
                output_path=str(out_path),
                max_new_tokens=750,
            )
        except Exception as e:
            print(f"\nFailed to generate {i}: {e}")

    print("Done")


def prompt_mode():
    PROMPT_DIR = OUTPUT_DIR / "prompts"
    PROMPT_DIR.mkdir(parents=True, exist_ok=True)

    tts = TTSInference(MODEL_PATH)
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
    process_input()
    # prompt_mode()
