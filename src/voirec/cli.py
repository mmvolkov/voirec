"""Command-line interface for voirec."""

import click
from pathlib import Path
from .transcribers import (
    WhisperTranscriber, GigaAmTranscriber, ParakeetTranscriber,
    transcribe_channels, diarize_and_transcribe, format_dialogue,
)


@click.command()
@click.argument('audio_file', type=click.Path(exists=True, path_type=Path))
@click.option('--output-dir', '-o', type=click.Path(path_type=Path), default=None,
              help='Директория для выходных файлов (по умолчанию: рядом с входным файлом)')
@click.option('--whisper-model', default='onnx-community/whisper-large-v3-turbo',
              help='Модель Whisper (по умолчанию: onnx-community/whisper-large-v3-turbo)')
@click.option('--gigaam-model', default='gigaam-v3-e2e-rnnt',
              help='Модель GigaAM: gigaam-v3-ctc, gigaam-v3-rnnt, gigaam-v3-e2e-ctc, gigaam-v3-e2e-rnnt (по умолчанию: gigaam-v3-e2e-rnnt)')
@click.option('--parakeet-model', default='nemo-parakeet-tdt-0.6b-v3',
              help='Модель Parakeet: nemo-parakeet-tdt-0.6b-v3, nemo-parakeet-tdt-0.6b-v2, nemo-parakeet-ctc-0.6b-v3, nemo-parakeet-rnnt-1.1b-v3 (по умолчанию: nemo-parakeet-tdt-0.6b-v3)')
@click.option('--skip-whisper', is_flag=True, help='Пропустить Whisper')
@click.option('--skip-gigaam', is_flag=True, help='Пропустить GigaAM')
@click.option('--skip-parakeet', is_flag=True, help='Пропустить Parakeet')
@click.option('--diarize', is_flag=True, help='Включить диаризацию (разделение говорящих с тайм-кодами)')
@click.option('--num-speakers', type=int, default=None, help='Точное число говорящих')
@click.option('--max-speakers', type=int, default=None, help='Максимальное число говорящих (для auto-detect)')
@click.option('--language', default=None, help='Код языка для транскрибации, например ru, en (по умолчанию: автодетект)')
def main(
    audio_file: Path,
    output_dir: Path,
    whisper_model: str,
    gigaam_model: str,
    parakeet_model: str,
    skip_whisper: bool,
    skip_gigaam: bool,
    skip_parakeet: bool,
    diarize: bool,
    num_speakers: int,
    max_speakers: int,
    language: str | None,
):
    """Транскрибация аудио файла с помощью Whisper, GigaAM и Parakeet."""

    click.echo(f"🎙️  Обработка файла: {audio_file}")

    if output_dir is None:
        output_dir = audio_file.parent
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    base_name = audio_file.stem
    transcribers = {}

    if not skip_whisper:
        try:
            click.echo(f"🔧 Загружаю Whisper ({whisper_model})...")
            transcribers['whisper'] = WhisperTranscriber(model_name=whisper_model)
        except Exception as e:
            click.echo(f"⚠️  Не удалось загрузить Whisper: {e}", err=True)

    if not skip_gigaam:
        try:
            click.echo(f"🔧 Загружаю GigaAM ({gigaam_model})...")
            transcribers['gigaam'] = GigaAmTranscriber(model_name=gigaam_model)
        except Exception as e:
            click.echo(f"⚠️  Не удалось загрузить GigaAM: {e}", err=True)

    if not skip_parakeet:
        try:
            click.echo(f"🔧 Загружаю Parakeet ({parakeet_model})...")
            transcribers['parakeet'] = ParakeetTranscriber(model_name=parakeet_model)
        except Exception as e:
            click.echo(f"⚠️  Не удалось загрузить Parakeet: {e}", err=True)

    if not transcribers:
        click.echo("❌ Ни один транскрибер не загружен. Выход.", err=True)
        return

    if diarize:
        first_name, first_transcriber = next(iter(transcribers.items()))
        output_file = output_dir / f"{base_name}_dialogue.txt"
        try:
            click.echo(f"\n🔄 Диаризация + транскрибация ({first_name.upper()})...")
            segments = diarize_and_transcribe(
                first_transcriber,
                str(audio_file),
                num_speakers=num_speakers,
                max_speakers=max_speakers,
                language=language,
            )
            dialogue = format_dialogue(segments)
            output_file.write_text(dialogue, encoding='utf-8')
            click.echo(f"✅ Диалог сохранён -> {output_file}")
        except Exception as e:
            click.echo(f"❌ Диаризация: ошибка - {e}", err=True)
    else:
        for name, transcriber in transcribers.items():
            output_file = output_dir / f"{base_name}_{name}.txt"
            try:
                click.echo(f"\n🔄 {name.upper()}: начинаю транскрибацию...")
                text = transcribe_channels(transcriber, str(audio_file), language=language)
                output_file.write_text(text, encoding='utf-8')
                click.echo(f"✅ {name.upper()}: готово -> {output_file}")
            except Exception as e:
                click.echo(f"❌ {name.upper()}: ошибка - {e}", err=True)

    click.echo(f"\n✨ Транскрибация завершена!")


if __name__ == '__main__':
    main()
