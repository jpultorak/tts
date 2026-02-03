import torch
from tqdm import tqdm

import tts.config as config
from tts.infer import TTSInference

CHECKPOINT_PATH = config.ROOT_DIR / "checkpoints" / "model_epoch_5.pt"
TRAIN_DATA_DIR = config.ROOT_DIR / "test_training_data"
OUTPUT_DIR = config.ROOT_DIR / "output"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tts = TTSInference(CHECKPOINT_PATH)
    files = sorted(list(TRAIN_DATA_DIR.glob("*.pt")))
    print(f"--> Found {len(files)} training samples in {TRAIN_DATA_DIR}")
    print(f"--> Generating audio to {OUTPUT_DIR} ...\n")

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
                max_new_tokens=750,  # ~10 seconds limit
            )
        except Exception as e:
            print(f"❌ Failed to generate {file_id}: {e}")

    print(f"\n✅ Batch Inference Complete! Check {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
