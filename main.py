import sys
import os
from pathlib import Path

# Определяем рабочую директорию
if getattr(sys, 'frozen', False):
    # Если запущено как .exe
    EXE_DIR = Path(sys.executable).parent
else:
    # Если запущено из PyCharm
    EXE_DIR = Path(__file__).parent

# Проверка: находится ли программа в Program Files
# Если да, то мы не можем писать туда данные
IS_PROGRAM_FILES = "Program Files" in str(EXE_DIR)

if IS_PROGRAM_FILES:
    # Используем папку пользователя (AppData), куда разрешена запись
    USER_DIR = Path(os.getenv('APPDATA')) / 'AI Drought Predictor'
    USER_DIR.mkdir(parents=True, exist_ok=True)

    DATA_DIR = USER_DIR / 'data'
    MODELS_DIR = USER_DIR / 'models'
    RESULTS_DIR = USER_DIR / 'results'
else:
    # Если программа в обычной папке (например, на диске D или в Dev), пишем туда
    DATA_DIR = EXE_DIR / 'data'
    MODELS_DIR = EXE_DIR / 'models'
    RESULTS_DIR = EXE_DIR / 'results'


# Функция создания папок
def ensure_folders():
    print("\n" + "=" * 60)
    print(" ПРОВЕРКА НЕОБХОДИМЫХ ПАПОК...")
    print("=" * 60)
    print(f"📂 Данные сохраняются в: {USER_DIR if IS_PROGRAM_FILES else EXE_DIR}")
    print()

    for folder in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
        try:
            folder.mkdir(parents=True, exist_ok=True)
            print(f"✅ Папка '{folder.name}' готова")
        except Exception as e:
            print(f"❌ Ошибка создания '{folder.name}': {e}")

    print("=" * 60 + "\n")


# Сохраняем пути в переменные окружения, чтобы другие файлы знали, куда сохранять
os.environ['AI_DP_DATA_DIR'] = str(DATA_DIR)
os.environ['AI_DP_MODELS_DIR'] = str(MODELS_DIR)
os.environ['AI_DP_RESULTS_DIR'] = str(RESULTS_DIR)

# Запуск инициализации
if __name__ == '__main__':
    ensure_folders()

    sys.path.insert(0, str(EXE_DIR))

    from src.gui import main

    print("🚀 ЗАПУСК ПРИЛОЖЕНИЯ...")
    print("=" * 60 + "\n")

    main()