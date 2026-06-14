import os
import mido
import click


def play_with_default_player(input):
    """MIDI 출력 포트를 사용할 수 없을 때, OS 기본 플레이어로 파일을 엽니다."""
    print("ℹ️ [대체 재생] MIDI 출력 장치를 사용할 수 없어 시스템 기본 플레이어로 엽니다.")
    try:
        os.startfile(os.path.abspath(input))  # Windows 전용
    except AttributeError:
        # macOS / Linux 대비
        import subprocess, sys
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.run([opener, input])


@click.command()
@click.option('--input', '-i', default="output.mid", show_default=True,
              help="재생할 MIDI 파일 경로")
def main(input):
    """
    MIDI 파일을 시스템 기본 신디사이저(Windows 내장 MIDI)를 통해 오디오로 재생합니다.
    (추가적인 사운드폰트 설치가 필요 없습니다.)

    인자 없이 실행하면 기본값으로 'output.mid'를 재생합니다.
    """
    if not os.path.exists(input):
        print(f"❌ [오류] 파일을 찾을 수 없습니다: {input}")
        return

    try:
        # 사용 가능한 출력 포트 확인
        outputs = mido.get_output_names()
        if not outputs:
            print("[알림] 사용 가능한 MIDI 출력 장치를 찾을 수 없습니다.")
            play_with_default_player(input)
            return

        port_name = outputs[0]  # 보통 Windows에서는 'Microsoft GS Wavetable Synth 0'
        print(f"\n🎵 [재생 시작] 파일: {input}")
        print(f"🔊 [출력 장치] {port_name}")
        print("정지하려면 Ctrl+C를 누르세요.\n")

        mid = mido.MidiFile(input)

        with mido.open_output(port_name) as port:
            try:
                for msg in mid.play():
                    if not msg.is_meta:
                        port.send(msg)
            finally:
                # 재생이 중단되어도 울리는 음을 모두 정지시킵니다.
                port.reset()

        print("✅ [재생 완료]")

    except KeyboardInterrupt:
        print("\n⏹️ [재생 중지됨]")
    except Exception as e:
        print(f"\n❌ 재생 중 오류가 발생했습니다: {e}")
        play_with_default_player(input)


if __name__ == '__main__':
    main()
