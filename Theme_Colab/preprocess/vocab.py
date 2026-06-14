"""Vocabulary for theme-based transformer (TSD / absolute-time, ambient version)

    Original REMI version: Ian Shih (yjshih23@gmail.com), 2021/11/03
    TSD / ambient adaptation: 2026

    토큰 표현을 REMI(Bar/Position/Tempo 격자) -> TSD(절대 시간 Time-Shift)로 교체.
    - 트랙은 program number로 식별: {MELODY:0, PAD:88}
    - 시간은 절대 초 단위(pretty_midi)로 처리하여 tempo 의존 tick 변환을 제거.
"""
import numpy as np
import pretty_midi as pm


class Vocab(object):
    def __init__(self):
        # ── 절대 시간 설정 ───────────────────────────────
        # 시간 해상도(초). 앰비언트는 거칠게 잡아 시퀀스 길이를 억제. (튜닝 대상)
        self.time_resolution = 0.04  # 40ms

        # 트랙 ↔ program 매핑 (pretty_midi program number로 식별)
        self.track_programs = {"MELODY": 0, "PAD": 88}
        self.tracks = list(self.track_programs.keys())  # ["MELODY", "PAD"]
        self.theme_track_name = "theme info track"

        self.token2id = {}
        self.id2token = {}
        self.n_tokens = 0
        self.token_type_base = {}

        # midi pitch 1~127
        self._pitch_bins = np.arange(start=1, stop=128)
        # duration: 1~200 step(=40ms) → 최대 8초
        self._duration_bins = np.arange(start=1, stop=201)
        # velocity 1~126 (midi)
        self._velocity_bins = np.arange(start=1, stop=127)
        # time-shift: 1~100 step(=40ms) → 토큰 1개당 최대 4초, 더 길면 연속 사용
        self._timeshift_bins = np.arange(start=1, stop=101)

        self.build()

    def build(self):
        """build our vocab"""
        self.token2id = {}
        self.id2token = {}
        self.n_tokens = 0
        self.token_type_base = {}

        self.token2id['padding'] = 0
        self.n_tokens = 1

        # Note-On (트랙별)
        for track in self.tracks:
            self.token_type_base['Note-On-{}'.format(track)] = self.n_tokens
            for i in self._pitch_bins:
                self.token2id['Note-On-{}_{}'.format(track, i)] = self.n_tokens
                self.n_tokens += 1
        # Note-Duration (트랙별)
        for track in self.tracks:
            self.token_type_base['Note-Duration-{}'.format(track)] = self.n_tokens
            for d in self._duration_bins:
                self.token2id['Note-Duration-{}_{}'.format(track, d)] = self.n_tokens
                self.n_tokens += 1
        # Note-Velocity (트랙별)
        for track in self.tracks:
            self.token_type_base['Note-Velocity-{}'.format(track)] = self.n_tokens
            for v in self._velocity_bins:
                self.token2id['Note-Velocity-{}_{}'.format(track, v)] = self.n_tokens
                self.n_tokens += 1

        # Time-Shift (전역, 절대 시간)
        self.token_type_base['Time-Shift'] = self.n_tokens
        for s in self._timeshift_bins:
            self.token2id['Time-Shift_{}'.format(s)] = self.n_tokens
            self.n_tokens += 1

        # Theme
        self.token_type_base['Theme'] = self.n_tokens
        self.token2id['Theme_Start'] = self.n_tokens
        self.n_tokens += 1
        self.token2id['Theme_End'] = self.n_tokens
        self.n_tokens += 1

        for w, v in self.token2id.items():
            self.id2token[v] = w
        self.n_tokens = len(self.token2id)

    def getPitch(self, input_event):
        """Note-On이면 pitch 반환, 아니면 -1 (pitch augmentation에서 사용)"""
        if isinstance(input_event, int):
            input_event = self.id2token[input_event]
        elif not isinstance(input_event, str):
            input_event = self.id2token[int(input_event)]
        if not input_event.startswith("Note-On"):
            return -1
        return int(input_event.split("_")[1])

    # ── 절대시간(초) → Time-Shift 토큰들 (긴 공백은 연속으로 분할) ──
    def _time_to_shift_tokens(self, dt_sec):
        tokens = []
        steps = int(round(dt_sec / self.time_resolution))
        max_step = int(self._timeshift_bins[-1])
        while steps > 0:
            s = min(steps, max_step)
            tokens.append("Time-Shift_{}".format(s))
            steps -= s
        return tokens

    def midi2TSD(self, midi_path, theme_annotations=True, verbose=False):
        """MIDI → TSD 토큰 id 시퀀스 (절대 시간 기반)"""
        midi = pm.PrettyMIDI(midi_path)

        # program 번호로 트랙 식별
        track_notes = {t: [] for t in self.tracks}
        theme_track = None
        for inst in midi.instruments:
            if inst.name == self.theme_track_name:
                theme_track = inst
                continue
            for t, prog in self.track_programs.items():
                if inst.program == prog:
                    track_notes[t].extend(inst.notes)

        # 모든 노트를 (시간순) 이벤트로
        stream = []  # (time, priority, kind, payload)
        for t in self.tracks:
            for n in track_notes[t]:
                stream.append((n.start, 1, "note",
                               {"track": t, "pitch": n.pitch,
                                "dur": n.end - n.start, "vel": n.velocity}))

        # 테마 구간을 Theme_Start/End 이벤트로 삽입
        if theme_annotations:
            assert theme_track is not None, "theme info track not found"
            marker_pitch = min(x.pitch for x in theme_track.notes)
            regions = sorted([(x.start, x.end) for x in theme_track.notes
                              if x.pitch == marker_pitch])
            for s, e in regions:
                stream.append((s, 0, "theme_start", None))  # 같은 시각이면 노트보다 앞
                stream.append((e, 2, "theme_end", None))    # 같은 시각이면 노트보다 뒤

        # 시간 → 우선순위 → 트랙순 → pitch 순 정렬
        track_order = {t: i for i, t in enumerate(self.tracks)}
        stream.sort(key=lambda x: (x[0], x[1],
                                   track_order.get(x[3]["track"], 0) if x[2] == "note" else 0,
                                   x[3]["pitch"] if x[2] == "note" else 0)
                                   )

        events = []
        prev_time = stream[0][0] if stream else 0.0  # 앞쪽 무음은 트림
        for time, _, kind, payload in stream:
            events.extend(self._time_to_shift_tokens(time - prev_time))
            prev_time = time
            if kind == "note":
                tr = payload["track"]
                events.append("Note-On-{}_{}".format(tr, payload["pitch"]))
                dsteps = int(round(payload["dur"] / self.time_resolution))
                dsteps = max(1, min(dsteps, int(self._duration_bins[-1])))
                events.append("Note-Duration-{}_{}".format(tr, dsteps))
                vi = np.argmin(abs(payload["vel"] - self._velocity_bins))
                events.append("Note-Velocity-{}_{}".format(tr, int(self._velocity_bins[vi])))
            elif kind == "theme_start":
                events.append("Theme_Start")
            elif kind == "theme_end":
                events.append("Theme_End")

        return [self.token2id[x] for x in events]

    def preprocessTSD(self, seq, always_include=False, max_seq_len=512, strict=True, verbose=False):
        """토큰 시퀀스를 src(테마)/tgt 세그먼트로 슬라이스 (로직은 기존 preprocessREMI와 동일)"""
        theme_binary_msk = []
        in_theme = False
        for r in seq:
            if self.id2token[r] == "Theme_Start":
                in_theme = True
            elif self.id2token[r] == "Theme_End":
                in_theme = False
            theme_binary_msk.append(int(in_theme))
        for i in range(1, len(theme_binary_msk)):
            theme_binary_msk[i] = theme_binary_msk[i-1]*theme_binary_msk[i] + theme_binary_msk[i]

        start_first = seq.index(self.token2id["Theme_Start"])
        end_first = seq.index(self.token2id["Theme_End"])
        src = seq[start_first:end_first+1]
        src_theme_binary_msk = theme_binary_msk[start_first:end_first+1]

        tgt_segments, tgt_segments_theme_msk = [], []
        if strict:
            theme_start_pos = [i for i in range(len(seq)) if seq[i] == self.token2id["Theme_Start"]]
            for t in theme_start_pos:
                tgt_segments.append(seq[t:t+max_seq_len+1])
                tgt_segments_theme_msk.append(theme_binary_msk[t:t+max_seq_len+1])
        else:
            for x in range(0, len(seq), max_seq_len):
                if always_include:
                    if self.token2id["Theme_Start"] in seq[x:x+max_seq_len+1] or \
                       self.token2id["Theme_End"] in seq[x:x+max_seq_len+1]:
                        tgt_segments.append(seq[x:x+max_seq_len+1])
                        tgt_segments_theme_msk.append(theme_binary_msk[x:x+max_seq_len+1])
                else:
                    tgt_segments.append(seq[x:x+max_seq_len+1])
                    tgt_segments_theme_msk.append(theme_binary_msk[x:x+max_seq_len+1])

        return {
            "src": src,
            "src_theme_binary_msk": src_theme_binary_msk,
            "tgt_segments": tgt_segments,
            "tgt_segments_theme_binary_msk": tgt_segments_theme_msk,
        }

    def TSDID2midi(self, event_ids, midi_path, verbose=False):
        """TSD 토큰 id 시퀀스 → MIDI (절대 시간 복원, pretty_midi로 저장)"""
        events = [self.id2token[x] for x in event_ids]

        midi = pm.PrettyMIDI()
        insts = {t: pm.Instrument(program=self.track_programs[t], name=t)
                 for t in self.tracks}
        theme_inst = pm.Instrument(program=0, name=self.theme_track_name)

        cur_time = 0.0
        theme_start_time = None
        idx = 0
        while idx < len(events):
            ev = events[idx]
            if ev.startswith("Time-Shift"):
                cur_time += int(ev.split("_")[1]) * self.time_resolution
                idx += 1
            elif ev.startswith("Note-On"):
                assert events[idx+1].startswith("Note-Duration")
                assert events[idx+2].startswith("Note-Velocity")
                track = ev.split("_")[0].split("-")[2]
                pitch = int(ev.split("_")[1])
                dur = int(events[idx+1].split("_")[1]) * self.time_resolution
                vel = int(events[idx+2].split("_")[1])
                if track in insts:
                    insts[track].notes.append(
                        pm.Note(velocity=vel, pitch=pitch,
                                start=cur_time, end=cur_time + dur))
                idx += 3
            elif ev == "Theme_Start":
                theme_start_time = cur_time
                idx += 1
            elif ev == "Theme_End":
                if theme_start_time is not None:
                    theme_inst.notes.append(
                        pm.Note(velocity=1, pitch=60,
                                start=theme_start_time, end=cur_time))
                    theme_start_time = None
                idx += 1
            else:
                idx += 1

        midi.instruments.extend([insts[t] for t in self.tracks])
        midi.instruments.append(theme_inst)
        if verbose:
            print("Saving midi to ({})".format(midi_path))
        midi.write(midi_path)

    def __str__(self):
        ret = ""
        for w, i in self.token2id.items():
            ret += "{} : {}\n".format(w, i)
        ret += "\nTotal events #{}".format(len(self.id2token))
        return ret

    def __repr__(self):
        return self.__str__()


if __name__ == '__main__':
    myvocab = Vocab()
    print(myvocab)
