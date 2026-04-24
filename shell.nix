{ pkgs ? import <nixpkgs> {}
, enableROCm ? false  # Установите true для поддержки AMD GPU
}:

pkgs.mkShell {
  buildInputs = with pkgs; [
    # Python и uv
    python312
    uv

    # Системные зависимости для аудио обработки
    ffmpeg
    portaudio

    # Для vosk
    gcc
    stdenv.cc.cc.lib

    # Для PyTorch - системные библиотеки
    zstd
    zlib
    glib
    libGL
    libGLU

    # Полезные утилиты
    git
  ] ++ pkgs.lib.optionals enableROCm (with pkgs; [
    # ROCm для AMD GPU
    rocmPackages.clr
    rocmPackages.rocm-runtime
    rocmPackages.rocm-device-libs
  ]);

  shellHook = ''
    echo "🎙️  Voirec development environment"
    echo "Python: $(python --version)"
    echo "uv: $(uv --version)"
    ${pkgs.lib.optionalString enableROCm ''
    echo "ROCm: enabled"
    ''}

    # Создаем виртуальное окружение если его нет
    if [ ! -d .venv ]; then
      echo "Создаю виртуальное окружение..."
      uv venv --python 3.12
    fi

    source .venv/bin/activate

    # Настройка переменных окружения для нативных библиотек (после активации venv)
    export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.portaudio}/lib:${pkgs.zstd.out}/lib:${pkgs.zlib}/lib:${pkgs.glib}/lib:${pkgs.libGL}/lib:${pkgs.libGLU}/lib:$LD_LIBRARY_PATH"

    ${pkgs.lib.optionalString enableROCm ''
    # Устанавливаем ROCm-сборку torch если ещё не установлена
    if python -c "import torch; assert hasattr(torch.version, 'hip') and torch.version.hip" 2>/dev/null; then
      echo "ROCm torch уже установлен"
    else
      echo "Устанавливаю torch/torchaudio с поддержкой ROCm..."
      uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm7.1 --reinstall-package torch --reinstall-package torchaudio
    fi

    # Настройка ROCm для AMD GPU
    export ROCM_PATH="${pkgs.rocmPackages.clr}"
    export LD_LIBRARY_PATH="${pkgs.rocmPackages.clr}/lib:${pkgs.rocmPackages.rocm-runtime}/lib:$LD_LIBRARY_PATH"
    export HIP_PLATFORM=amd

    # Диагностика GPU
    echo ""
    echo "🔍 Диагностика GPU:"

    # Проверка ROCm устройств на системном уровне
    if [ -c /dev/kfd ]; then
      echo "  ✓ /dev/kfd обнаружен (ROCm kernel driver активен)"
    else
      echo "  ✗ /dev/kfd не найден - ROCm драйверы не загружены"
      echo "    Убедитесь, что ROCm установлен системно и драйвер amdgpu загружен"
    fi

    # Проверка PyTorch (если уже установлен)
    if python -c "import torch" 2>/dev/null; then
      echo ""
      echo "  PyTorch установлен, проверяю конфигурацию:"
      python -c "
import torch
print(f'    Версия: {torch.__version__}')
if hasattr(torch.version, 'hip') and torch.version.hip:
    print(f'    ROCm: {torch.version.hip}')
else:
    print('    ⚠  ROCm build: НЕТ (установлена CPU/CUDA версия)')
print(f'    GPU доступен: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'    Устройств: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'      [{i}] {torch.cuda.get_device_name(i)}')
      "
    else
      echo "  ℹ  PyTorch не установлен (будет установлен при uv sync)"
    fi

    echo ""
    ''}
  '';
}
