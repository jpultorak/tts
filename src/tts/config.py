from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
VOCAB_PATH = ROOT_DIR / "vocab.json"
PHONEME_START_ID = 1024
ENCODEC_BANDWIDTH = 6.0  # TODO: Verify what this does exactly

SEP_TOKEN_ID = 1026
EOS_TOKEN_ID = 1024
PAD_TOKEN_ID = 1025
