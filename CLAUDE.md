# ThemeTransformer 실시간 앰비언트 스트리밍 (feat/streaming)

학습된 ThemeTransformer(인코더-디코더 듀얼트랙 MIDI 생성기)를 이용해 **연속적으로** 앰비언트
음악을 생성하고, 웹 UI로 스트리밍하며 loopMIDI 가상 포트를 통해 DAW(Cakewalk)로 출력하는 시스템.

## 실행

```
Ambient/myenv/Scripts/python.exe webapp/server.py \
    --model-path trained_model/model_ep325.pt \
    --melody-port "melody" --pad-port "pad"
```

- venv: `D:\학교\26.1\Deep Learning for Music and Audio\Project\ambient_v2\Ambient\myenv\Scripts\python.exe`
- 웹: http://127.0.0.1:5000
- loopMIDI에서 가상 포트 2개("melody", "pad")를 미리 생성해야 함. `--melody-port`/`--pad-port`는
  포트 이름의 부분 문자열로 매칭됨.
- Cakewalk: 두 MIDI 포트를 Instrument Track(VST 악기 연결)으로 받아야 소리가 남. 단순 MIDI 트랙은
  악기가 없어 무음. Preferences → MIDI → Devices에서 두 포트 입력 활성화 + Audio → Devices 출력 확인.

## 아키텍처

```
AnchorQueue ─► ChunkGenerator(thread) ─► TokenChunkQueue ─► WebSession._reader_loop(thread)
                                                                ├─► TSDStreamingDecoder ─► SSE(피아노롤)
                                                                └─► MIDIScheduler ─► rtmidi(loopMIDI)
```

- **ChunkGenerator** (`src/streaming/chunk_gen.py`): 자기회귀 토큰 생성. 인코더 theme + 디코더
  컨텍스트 누적. 문법 제약(Note-On→Duration→Velocity 동일 트랙), nucleus 샘플링.
- **TSDStreamingDecoder** (`webapp/server.py`): TSD 토큰 → 절대시간 노트 이벤트. 청크 경계 넘어
  상태 유지(`_pending`, `_buf`).
- **MIDIScheduler** (`webapp/server.py`): wall-clock(`time.perf_counter()`) 기준 우선순위 큐로
  rtmidi에 정확히 전송. MELODY/PAD 각각 별도 포트.
- **WebSession** (`webapp/server.py`): 위 컴포넌트 통합.

## Seed / Anchor 시맨틱 (하이브리드)

- **Seed**: 스트림 시작 시 (1) 디코더 컨텍스트에 리터럴 삽입(그대로 재생) + (2) 초기 인코더 theme.
- **Anchor**: 중간 주입 시 (1) 디코더 컨텍스트에 리터럴 삽입 + (2) 새 인코더 theme로 교체 →
  이후 생성이 새 theme 조건으로 전환. `threading.Lock`으로 theme 교체 thread-safe.
- 둘 다 `Theme_Start + tokens + Theme_End`로 감싸 인코더 입력으로 사용.

## TSD 토큰화 (vocab.py)

- 절대 시간, `time_resolution = 0.04` (40ms).
- `Note-On-{MELODY|PAD}_{pitch}`, `Note-Duration-..._{steps}`, `Note-Velocity-..._{vel}`,
  `Time-Shift_{steps}`(1~100, 토큰당 최대 4초), `Theme_Start`/`Theme_End`.
- 트랙은 program으로 식별: MELODY=0, PAD=88.

---

## 해결한 문제들 (시간순)

### 1. MIDI panic (종료 후에도 소리 지속)
`MIDIScheduler.panic()`: 16채널 전체에 CC120(All Sound Off)+CC123(All Notes Off) 전송.
- `stop()`이 `panic()` 호출
- `/api/panic` 엔드포인트 + UI Panic 버튼
- `atexit` + SIGINT/SIGTERM 핸들러로 서버 종료 시에도 호출

### 2. 노트 간격이 너무 띄엄띄엄 (모델 미수정)
`TSDStreamingDecoder`에 `time_scale` 파라미터(기본 0.5). Time-Shift와 Note-Duration 양쪽에
곱해 시간 간격을 압축. UI 슬라이더(0.2~1.0)로 조정. 값이 작을수록 촘촘.

### 3. 재생되는데 화면에 안 뜨는 노트 (피치 범위)
피아노롤 표시 범위 `PITCH_MIN/MAX`를 24~96 → 12~108로 확장.

### 4. 피아노롤-MIDI 싱크 불일치 + 노트 미표시 (★ 클록 기준점 불일치)
**원인**: 서버는 `time.perf_counter()`(부팅 기준 단조시간)로 MIDI 스케줄, 클라이언트는
`performance.now()`(브라우저 기준 단조시간)로 wallStart 설정 → 두 클록 기준점이 달라
세션 생성 소요시간(수 초)만큼 피아노롤이 뒤처짐.
**수정**: 서버가 `/api/start` 응답에 `wall_start_epoch`(`time.time()`) 포함 →
클라이언트가 `wallStart = wall_start_epoch*1000`, `musicalNow()`는 `Date.now()` 사용
(서버와 동일 epoch 기준). + SSE drain 타임아웃 0.3s→0.05s. + WINDOW_FUTURE 20→30s, lead_cap 20→8s.

### 5. 피아노롤에 아무것도 안 뜸 (★★ init_seed label 버그 — 가장 치명적)
**원인**: `ChunkGenerator.init_seed`가 컨텍스트를 `[Theme_Start, *seed_tokens]`로만 구성하고
`Theme_End`를 누락 → `previous_labeled=True`인 채로 생성 시작 → 모델이 **이후 생성 토큰을
전부 "테마 내부"(label 1,2,3...)로 인식** → 훈련 분포와 완전히 다른 조건이 되어 0토큰/엉뚱한
시퀀스만 생성 → 피아노롤 공백.
**수정**: 컨텍스트를 `[Theme_Start, *seed_tokens, Theme_End]`로 구성하고 label도
(테마 내부 1..N+1, Theme_End 및 이후 생성 토큰은 0)으로 올바르게 설정. 훈련 포맷과 일치.

### 6. 오디오 공백·실시간 미달·생성량 급감 (★ 처리량 문제)
**진단**: forward 1회는 컨텍스트 길이와 무관하게 **고정 ~40ms/token**(컨텍스트 증가가 원인
아님 — 커널런치/동기화 오버헤드 지배). 진짜 원인은 (a) 문법 위반 토큰을 **reject 후 재샘플**
→ 위반마다 40ms forward 1회를 통째로 낭비, temp=1.2에서 드리프트가 누적되며 낭비율 폭증 →
수십 초 후 생성량 급감 + `fail_cnt>512` 자동정지. (b) 매 토큰마다 인코더(theme) 재실행 낭비.
**수정**:
- **문법을 로짓 마스킹으로** 강제(`chunk_gen._build_grammar_masks`/`_allow_mask`). 직전 토큰
  상태(free/dur-T/vel-T)별 허용 토큰만 남기고 나머지 logit=-1e9 → **forward 1회 = 유효토큰 1개**.
  reject 루프·`fail_cnt` 자동정지 제거. padding/Theme_*는 생성 중 금지.
- **인코더 메모리 캐시**(`myLM.encode_theme`/`decode_step`, `chunk_gen._ensure_memory`):
  theme 교체 시에만 재계산 → 39.5ms→26.6ms/token. 결과 실측 **realtime 4.1x**(time_scale=0.5).
- 기본 temperature 1.2→1.0(드리프트 감소).

### 7. 클록을 빈 버퍼로 시작 → seed 안 보임/싱크 깨짐 (★ 사전버퍼 부재)
**진단**: `start()`가 버퍼가 빈 상태로 wall-clock·플레이헤드를 즉시 시작 → seed가 묻히고
시작 직후 언더런 → 과거 시각으로 스케줄된 노트(소리는 나지만 롤엔 플레이헤드 뒤/표시 안 됨).
**수정**: `WebSession.start()`가 `prebuffer`(기본 4s)초 분량을 먼저 디코딩(워밍업)한 뒤에야
서버/클라 클록을 **같은 순간**에 기준점 설정하고 일괄 스케줄·SSE 방출. seed가 항상 맨 앞에서
리터럴 재생·표시됨.

### 8. anchor 지연(최대 20s)·테마 미교체 (★ 게이팅 구조 + 버퍼 깊이)
**진단**: (a) anchor가 lead_cap(8s)+토큰큐(8청크) 뒤 **생성 프론티어**에 추가돼 한참 뒤 재생.
(b) cross-attention이 **`tgt_label>0`에서만 적용**(myTransformer.py:567-573)되는데 anchor를
label-0 리터럴로 삽입 → 새 theme이 생성에 **전혀 조건화되지 않음**.
**수정**:
- anchor를 디코더 컨텍스트에 **`Theme_Start + tokens + Theme_End` 라벨드 테마 영역**으로 삽입
  (`_inject_anchor`) + 인코더 theme 교체. 컨텍스트 윈도가 슬라이드하며 옛 seed 테마가 밀려나고
  anchor가 지배 테마가 됨(=이전 테마가 새 anchor로 치환).
- 버퍼 축소(`token_q maxsize 8→2`, `lead_cap 8→5`)로 anchor가 플레이헤드 근처(~수초)에 삽입.
  생성기는 청크 도중에도 anchor 큐를 확인해 즉시 처리. 마커는 reader가 **실제 디코딩 시점**의
  음악시간으로 방출(마커-노트 정렬).

### 9. 지속음이 여러 노트로 쪼개져 재생 (★ 같은 pitch 겹침 + 스케줄러 보이스 미추적)
**진단**: 모델이 지속되는 패드음을 0.5~1초마다 **같은 pitch로 겹쳐 재발화**함(실측: 같은
트랙+pitch 인접쌍 88개 중 45개 겹침, 동일 시각 완전중복 15개). MIDI는 (채널,pitch)당 상태가
하나뿐인데 `MIDIScheduler`가 어떤 pitch가 울리는지 추적하지 않아 → (a) NOTE_ON 재전송으로
어택이 다시 들리고 (b) 앞 노트의 NOTE_OFF가 **뒤 노트까지 꺼버려** 한 음이 조각조각 끊김.
**수정**: `MIDIScheduler._voice[(track,pitch)] = (off_at, flag)`로 보이스 추적.
`schedule()`이 겹침을 감지하면 재타건하지 않고 **note-off만 뒤로 연장(tie)**, 완전 포함되는
노트는 버림. 옛 off 항목은 **스케줄 시점에** `flag["cancelled"]=True`로 무효화하고 `_loop()`가
pop 시 폐기 — 노트는 재생보다 수 초 앞서 스케줄되므로 발화 시점에 off 시각을 비교하는 방식은
정상 off까지 stale로 버린다(구현 시 실제로 발생). 겹치지 않는 반복은 재타건 유지.
**남은 원인**: 생성 단계에서 울리는 중인 pitch의 Note-On을 로짓 마스킹하는 처리는 아직 미적용
(`chunk_gen._allow_mask`) — 소리는 정상이지만 모델이 중복 토큰에 예산을 낭비함.

## 알려진 튜닝 포인트
- `time_scale`(노트 밀도), `temperature`/`top_p`(다양성), `lead_cap`(생성 선행 버퍼),
  `prebuffer`(시작 전 사전버퍼 초), `pitch_min`/`pitch_max`(생성 음역 마스킹) — 모두
  UI/`/api/start`에서 조정 가능.
- 문법 로짓 마스킹으로 생성이 막혀 자동정지하는 일은 없어짐(유효토큰 항상 생성).
