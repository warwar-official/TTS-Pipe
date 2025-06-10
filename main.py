import argparse
import json
import logging
import sys
import time
import urllib.request
import urllib.error
import wave
from pathlib import Path

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('tts_pipeline.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Константи для API
GEMINI_API_KEY = ""
GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key=' + GEMINI_API_KEY
TTS_API_URL = 'http://127.0.0.1:8000/v1/audio/speech'
PARTS_DIR = Path('parts')
MAX_TTS_TIMEOUT = 900  # 15 хвилин
RETRY_COUNT = 3
LETTERS_PER_FRADMENT = 2500


def parse_args():
    parser = argparse.ArgumentParser(description='Озвучення тексту з локальної TTS моделі')
    parser.add_argument('-f', '--file', required=True,
                        help='Вхідний TXT-файл з текстом для озвучування')
    parser.add_argument('-s', '--stressed', default=None,
                        help='Файл для збереження наголошеного тексту')
    parser.add_argument('-o', '--output', default=None,
                        help='Файл для збереження згенерованого аудіо')
    parser.add_argument('-v', '--voice', type=int, default=5,
                        help='Номер голосу TTS моделі (default=5)')
    return parser.parse_args()


def stress_text(input_path: Path, stressed_path: Path):
    """
    Відправка тексту в API Gemini для розмітки наголосів
    """
    logger.info(f"Розпочато наголошення: {input_path} → {stressed_path}")
    with input_path.open('r', encoding='utf-8') as f:
        content = f.read()

    prompt = (
            "Завдання: Розстав наголоси в наступному тексті. "
            "Ти маєш враховувати контекст використання слів та правильно наголошувати омоніми. "
            "Заміни цифри та числа на їх текстове представлення з урахуванням контексту та правил відмінювання в українській мові. "
            f"Текст: {content}\n "
            "Інструкція: Поверни результат у форматі: 'Ма́ло хто поміча́є, як ма́ло сві́тла на даха́х буди́нків.'. "
            "У відповіді не повинно бути жодних додаткових фраз, коментарів чи пояснень, лише зазначений формат."
        )

    for _ in range(2):
        payload = {
            "contents":[
                {
                    "parts":[
                        {
                            "text": prompt
                        }
                    ]
                }
            ]
        }

        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            GEMINI_API_URL,
            data=data,
            headers={'Content-Type': 'application/json'}
        )
        try:
            with urllib.request.urlopen(req, timeout=MAX_TTS_TIMEOUT) as resp:
                res_text = resp.read().decode('utf-8')
        except Exception as e:
            logger.error(f"Помилка при зверненні до Gemini API: {e}")
            return False

        json_data = json.loads(res_text)

        prompt = prompt = (
            "Завдання: Перевір розстановку наголосів у тексті та за потреби внеси виправлення. "
            "Зосередся на можливих помилках, де слово може бути наголошено порізному. "
            "Враховуй контекст та сенс речення при визначенні правильного наголошення. "
            f"Текст: {json_data}\n "
            "Інструкція: Поверни результат у форматі: 'Ма́ло хто поміча́є, як ма́ло сві́тла на даха́х буди́нків.'. "
            "У відповіді не повинно бути жодних додаткових фраз, коментарів чи пояснень, лише зазначений формат."
        )
    
    with stressed_path.open('w', encoding='utf-8') as f:
        f.write(json_data["candidates"][0]["content"]["parts"][0]["text"])
    logger.info(f"Наголошений текст збережено у {stressed_path}")
    return True


def tts_synthesize(stressed_path: Path, voice: int):
    """
    Відправка тексту на TTS модель та збереження частин
    """
    logger.info(f"Розпочато озвучування: {stressed_path}")
    if not PARTS_DIR.exists():
        PARTS_DIR.mkdir()

    # Перевірка на порожність теки
    if any(PARTS_DIR.iterdir()):
        logger.error(f"Тека '{PARTS_DIR}' не порожня. Припинення виконання.")
        print(f"Тека '{PARTS_DIR}' не порожня. Очистіть її і спробуйте знову.")
        return False

    text = stressed_path.read_text(encoding='utf-8')
    # Розбиваємо на фрагменти >= 500 символів по абзацам
    fragments = []
    for para in text.split('\n'):
        if not para.strip():
            continue
        if fragments and len(fragments[-1]) < LETTERS_PER_FRADMENT:
            fragments[-1] += ' ' + para
        else:
            fragments.append(para)

    for idx, frag in enumerate(fragments, start=1):
        success = False
        for attempt in range(1, RETRY_COUNT + 1):
            try:
                payload = {
                    'input': frag,
                    'voice': voice,
                    'speed': 0.85,
                    'verbalize': 0
                }
                data = json.dumps(payload).encode('utf-8')
                req = urllib.request.Request(
                    TTS_API_URL,
                    data=data,
                    headers={'Content-Type': 'application/json'}
                )
                with urllib.request.urlopen(req, timeout=MAX_TTS_TIMEOUT) as resp:
                    audio = resp.read()
                part_file = PARTS_DIR / f"part_{idx:03d}.wav"
                with part_file.open('wb') as f:
                    f.write(audio)
                logger.info(f"Збережено частину {idx}: {part_file}")
                success = True
                break
            except urllib.error.HTTPError as e:
                logger.error(f"HTTP помилка при фрагменті {idx}, спроба {attempt}: {e}")
            except Exception as e:
                logger.error(f"Помилка при фрагменті {idx}, спроба {attempt}: {e}")
            time.sleep(1)

        if not success:
            print(f"Не вдалося озвучити фрагмент {idx}. Повернення до меню.")
            return False

    return True


def merge_parts(output_path: Path, source_dir: Path):
    """
    Склеювання WAV-файлів у один
    """
    logger.info(f"Почато склеювання частин у {output_path}")
    part_files = sorted(source_dir.glob('part_*.wav'))
    if not part_files:
        logger.error("Не знайдено частин для склеювання.")
        print("Не знайдено частин у папці для склеювання.")
        return False

    with wave.open(str(part_files[0]), 'rb') as w:
        params = w.getparams()
        frames = w.readframes(w.getnframes())

    for pf in part_files[1:]:
        with wave.open(str(pf), 'rb') as w:
            if w.getparams() != params:
                logger.warning(f"Параметри аудіо в {pf} відрізняються.")
            frames += w.readframes(w.getnframes())

    with wave.open(str(output_path), 'wb') as out_f:
        out_f.setparams(params)
        out_f.writeframes(frames)

    duration_sec = params.nframes / params.framerate
    if duration_sec > 3600:
        print(f"Увага: результат займає більше години ({duration_sec/3600:.2f} год)")
    logger.info(f"Склеєно файл: {output_path}")
    return True


def run_full_cycle(input_file: Path, stressed_file: Path, output_file: Path, voice: int):
    if stress_text(input_file, stressed_file):
        if tts_synthesize(stressed_file, voice):
            merge_parts(output_file, PARTS_DIR)


def menu(args):
    input_file = Path(args.file)
    stressed_file = Path(args.stressed) if args.stressed else input_file.with_suffix('.stressed.txt')
    output_file = Path(args.output) if args.output else input_file.with_suffix('.wav')

    while True:
        print("\nМеню:")
        print("1 - повний цикл (наголошення + озвучування + склеювання)")
        print("2 - наголошення")
        print("3 - озвучування")
        print("4 - склеювання")
        print("0 - вихід")
        choice = input("Виберіть пункт: ")

        if choice == '1':
            run_full_cycle(input_file, stressed_file, output_file, args.voice)
        elif choice == '2':
            stress_text(input_file, stressed_file)
        elif choice == '3':
            tts_synthesize(stressed_file, args.voice)
        elif choice == '4':
            merge_parts(output_file, PARTS_DIR)
        elif choice == '0':
            print("Вихід.")
            break
        else:
            print("Невірний вибір, спробуйте ще раз.")


if __name__ == '__main__':
    args = parse_args()
    try:
        menu(args)
    except Exception as e:
        logger.exception(f"Неочікувана помилка: {e}")
        sys.exit(1)
