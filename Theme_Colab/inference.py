"""Theme Transformer Inferencing Code (TSD / absolute-time, ambient version)

usage: inference.py [--model_path MODEL_PATH] --theme THEME
                    [--seed SEED] [--out_midi OUT_MIDI] [--cuda]
                    [--max_len MAX_LEN] [--temp TEMP] [--gen_seconds GEN_SECONDS]
"""
import argparse
import torch
import torch.optim

from mymodel import myLM
from preprocess.vocab import Vocab
from randomness import set_global_random_seed

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='./trained_model/model_ep2311.pt', help='model file')
parser.add_argument('--theme', type=str, required=True, help='theme midi file (MELODY=prog0, PAD=prog88)')
parser.add_argument('--seed', type=int, default=-1, help='random seed (-1 = random)')
parser.add_argument('--out_midi', type=str, default='output.mid', help='output midi file')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--max_len', type=int, default=512, help='decoder context window (tokens)')
parser.add_argument('--temp', type=float, default=1.2, help='temperature')
parser.add_argument('--gen_seconds', type=float, default=60.0, help='length of music to generate (seconds)')
parser.add_argument('--pitch_min', type=int, default=0, help='generated pitch floor (inclusive)')
parser.add_argument('--pitch_max', type=int, default=127, help='generated pitch ceiling (inclusive)')
args = parser.parse_args()

if not args.seed == -1:
    set_global_random_seed(args.seed)

# create vocab
myvocab = Vocab()

# Note-On token ids whose pitch is outside [pitch_min, pitch_max] -> masked out at generation
forbidden_pitch_ids = [tid for tok, tid in myvocab.token2id.items()
                       if tok.startswith("Note-On")
                       and not (args.pitch_min <= int(tok.split("_")[1]) <= args.pitch_max)]
if forbidden_pitch_ids:
    print("Restricting generated pitch to [{}, {}] ({} Note-On tokens masked)".format(
        args.pitch_min, args.pitch_max, len(forbidden_pitch_ids)))

# devices (auto-detect GPU; on Colab CUDA is used without needing --cuda)
use_cuda = args.cuda or torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

# model definition
model = myLM(myvocab.n_tokens, d_model=256, num_encoder_layers=6, xorpattern=[0, 0, 0, 1, 1, 1])
print("Loading model from {}".format(args.model_path))
model.load_state_dict(torch.load(args.model_path, map_location=device))
print("Using device {}".format(device))


def inference(target_seconds, strategies, params, theme_seq, prompt=None):
    """generate a TSD token sequence conditioned on theme_seq, until target_seconds of music."""
    model.eval()
    words = [[]]
    word2event = myvocab.id2token

    initial_flag = True
    fail_cnt = 0
    gen_time = 0.0              # accumulated generated time (seconds)
    max_total_events = 20000   # safety cap

    input_theme = torch.tensor(theme_seq).reshape((-1, 1)).to(device)

    label_list = []
    previous_labeled = False

    with torch.no_grad():
        while gen_time < target_seconds and len(words[0]) < max_total_events:
            print("events #{}  gen_time {:.1f}/{:.1f}s".format(
                len(words[0]), gen_time, target_seconds), end='\r')

            if fail_cnt > 1024:
                print('\nmodel stuck ... change a seed and inference again!')
                return words[0]

            # prepare input
            if initial_flag:
                if prompt is not None:
                    input_x = torch.tensor(prompt)
                    words[0].extend(prompt)
                    label_list = [0] * len(prompt)
                    for i, x in enumerate(prompt):
                        if myvocab.id2token[x] == "Theme_Start":
                            previous_labeled = True
                        elif myvocab.id2token[x] == "Theme_End":
                            previous_labeled = False
                        if previous_labeled:
                            label_list[i] = 1 if i == 0 else label_list[i - 1] + 1
                    label_input = torch.tensor(label_list)
                else:
                    input_x = torch.tensor([theme_seq[0]])
                    label_list = [0]
                    words[0].append(theme_seq[0])
                    if myvocab.id2token[theme_seq[0]] == "Theme_Start":
                        previous_labeled = True
                    label_input = torch.tensor(label_list)
                initial_flag = False
            else:
                input_x = torch.tensor(words[0][-args.max_len:])
                label_input = torch.tensor(label_list[-args.max_len:])

            input_x = input_x.reshape((-1, 1))
            label_input = label_input.reshape((-1, 1))
            input_x_att_msk = model.transformer_model.generate_square_subsequent_mask(input_x.shape[0])
            input_x = input_x.to(device)
            label_input = label_input.to(device)
            input_x_att_msk = input_x_att_msk.to(device)

            logits = model(src=input_theme, tgt=input_x, tgt_label=label_input, tgt_mask=input_x_att_msk)
            logits = torch.squeeze(logits[-1:]).cpu().numpy()

            # mask out-of-range pitches so they can never be generated
            if forbidden_pitch_ids:
                logits[forbidden_pitch_ids] = -1e9

            if 'temperature' in strategies:
                probs = model.temperature(logits=logits, temperature=params['t'])
            else:
                probs = model.temperature(logits=logits, temperature=1.)
            word = model.nucleus(probs=probs, p=params['p'])

            # skip padding
            if word in [0]:
                fail_cnt += 1
                continue

            prev = word2event[words[0][-1]]
            cur = word2event[word]

            # ── grammar checking (TSD) ───────────────────────────────
            # Note-On -> Note-Duration (same track)
            if 'Note-On' in prev and 'Note-Duration' not in cur:
                fail_cnt += 1; continue
            if 'Note-Duration' in cur and 'Note-On' not in prev:
                fail_cnt += 1; continue
            if 'Note-On' in prev and 'Note-Duration' in cur:
                if prev.split("_")[0].split("-")[2] != cur.split("_")[0].split("-")[2]:
                    fail_cnt += 1; continue
            # Note-Duration -> Note-Velocity (same track)
            if 'Note-Duration' in prev and 'Note-Velocity' not in cur:
                fail_cnt += 1; continue
            if 'Note-Velocity' in cur and 'Note-Duration' not in prev:
                fail_cnt += 1; continue
            if 'Note-Duration' in prev and 'Note-Velocity' in cur:
                if prev.split("_")[0].split("-")[2] != cur.split("_")[0].split("-")[2]:
                    fail_cnt += 1; continue

            # Theme region consistency
            if cur.startswith("Theme"):
                if cur == "Theme_Start" and not previous_labeled:
                    previous_labeled = True
                elif cur == "Theme_End" and previous_labeled:
                    previous_labeled = False
                else:
                    fail_cnt += 1; continue

            # ── accept ──────────────────────────────────────────────
            words[0].append(word)
            label_list.append(label_list[-1] + 1 if previous_labeled else 0)

            # advance generated time on Time-Shift
            if cur.startswith("Time-Shift"):
                gen_time += int(cur.split("_")[1]) * myvocab.time_resolution

            fail_cnt = 0

    print('\ngenerated {} events ({:.1f}s)'.format(len(words[0]), gen_time))
    return words[0]


# theme condition from a theme midi (no theme info track needed; wrap with Theme markers)
given_theme = myvocab.midi2TSD(args.theme, theme_annotations=False)
given_theme = [myvocab.token2id["Theme_Start"]] + given_theme + [myvocab.token2id["Theme_End"]]

model.to(device)
word_seq = inference(
    target_seconds=args.gen_seconds,
    strategies=['temperature', 'nucleus'],
    params={'t': args.temp, 'p': 0.9},
    theme_seq=given_theme,
    prompt=[myvocab.token2id["Theme_Start"]],
)

# save to disk
myvocab.TSDID2midi(word_seq, args.out_midi)
print("{} saved".format(args.out_midi))
