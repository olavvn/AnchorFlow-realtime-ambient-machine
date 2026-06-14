import numpy as np
import pretty_midi as pm
from preprocess.vocab import Vocab

myvocab = Vocab()

print(myvocab)

midi_path = "output_midi\Ambient01.mid"
output_id = myvocab.midi2TSD(midi_path,True,True)
output_tokens = [myvocab.id2token[id] for id in output_id]
print(output_tokens[:100])