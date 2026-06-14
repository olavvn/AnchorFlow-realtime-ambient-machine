import vocab
import numpy as np
import glob
import os
import pickle
import random

myvocab = vocab.Vocab()


# Check the paths for your own case
# the theme annotated (ambient) midi files
MIDI_FILES = "../ambient_midis/*.mid"

# the tokens converted from theme annotated midi files
MIDI_FILES_PKLs_DIR = "./ambient_midi_pkls"

# the output training data (train / val split at the piece level)
OUTPUT_DIR = "../data_pkl"
TRAIN_PKL = os.path.join(OUTPUT_DIR, "train_seg2_512.pkl")
VAL_PKL = os.path.join(OUTPUT_DIR, "val_seg2_512.pkl")
VAL_RATIO = 0.1     # fraction of pieces held out for validation
SPLIT_SEED = 42     # reproducible split

os.makedirs(MIDI_FILES_PKLs_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

all_mids = sorted(glob.glob(MIDI_FILES))
print("Found {} midi files".format(len(all_mids)))

for _midiFile in all_mids:
    # convert midi files to token representation and save as .pkl file
    base = os.path.basename(_midiFile).replace(".mid", ".pkl")
    output_pkl_fp = os.path.join(MIDI_FILES_PKLs_DIR, base)
    tsd_seq = myvocab.midi2TSD(_midiFile, theme_annotations=True, verbose=False)
    ret = myvocab.preprocessTSD(tsd_seq, always_include=True, max_seq_len=512, verbose=True)
    pickle.dump(ret, open(output_pkl_fp, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

def build_entries(pkl_files):
    """Flatten a list of per-piece pkl files into training entries (one per tgt segment)."""
    entries = []
    for fn in pkl_files:
        with open(fn, "rb") as f:
            data = pickle.load(f)
        src = data["src"]
        src_theme_binary_msk = data["src_theme_binary_msk"]
        for i_tgt, tgt in enumerate(data["tgt_segments"]):
            entries.append({
                "src": src,
                "tgt": tgt,
                "tgt_theme_msk": data["tgt_segments_theme_binary_msk"][i_tgt],
                "src_theme_msk": src_theme_binary_msk,
            })
    return entries


# collect all per-piece pkl files and split at the PIECE level (avoid train/val leakage)
all_pkls = sorted(glob.glob(os.path.join(MIDI_FILES_PKLs_DIR, "*.pkl")))
shuffled = all_pkls[:]
random.Random(SPLIT_SEED).shuffle(shuffled)

if len(shuffled) >= 2:
    n_val = max(1, int(round(len(shuffled) * VAL_RATIO)))
    n_val = min(n_val, len(shuffled) - 1)   # keep at least one piece in train
else:
    n_val = 0
val_files = shuffled[:n_val]
train_files = shuffled[n_val:]

train_data = build_entries(train_files)
val_data = build_entries(val_files)
if not val_data:
    print("WARNING: not enough pieces for a real val split -> using train as val (smoke-test only)")
    val_data = train_data

print("pieces  train/val : {}/{}".format(len(train_files), len(val_files)))
print("entries train/val : {}/{}".format(len(train_data), len(val_data)))

# dump to pkl files for training / validation
pickle.dump(train_data, open(TRAIN_PKL, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(val_data, open(VAL_PKL, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
print("saved:\n  {}\n  {}".format(TRAIN_PKL, VAL_PKL))






