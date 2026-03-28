# 🤖 ML Data Pipeline — Portfolio Project

> Полный воспроизводимый ML-пайплайн: от сбора данных до обученной модели,  
> с тремя точками human-in-the-loop контроля качества.

---

## Быстрый старт

```bash
# 1. Клонировать репозиторий
git clone <your-repo-url>
cd ml-data-pipeline

# 2. Установить зависимости
pip install -r requirements.txt

# 3. Настроить API ключи
cp .env.example .env
# Заполнить OPENAI_API_KEY в .env

# 4. Запустить пайплайн
python run_pipeline.py

# Или в автоматическом режиме (без HITL-пауз):
python run_pipeline.py --no-hitl

# Опции:
python run_pipeline.py --query "toxic comment classification" \
                       --threshold 0.75 \
                       --al-select 30
```

---

## Структура репозитория

```
ml-data-pipeline/
│
├── agents/
│   ├── dataset_agent.py        # Шаг 1: сбор данных (Kaggle + HuggingFace)
│   ├── data_quality_agent.py   # Шаг 2: очистка → HTML-отчёт
│   ├── annotation_agent.py     # Шаг 3: авторазметка + Label Studio export
│   └── al_agent.py             # Шаги 4–5: AL selection + обучение (entropy vs random)
│
├── notebooks/
│   └── al_experiment.ipynb     # AL эксперимент Track A (standalone)
│
├── data/
│   ├── raw/
│   │   ├── raw_catalog.csv     # Каталог найденных датасетов
│   │   └── dataset.csv         # Сырые данные
│   └── labeled/
│       ├── clean.csv           # После очистки
│       ├── auto_labeled.csv    # После авторазметки
│       ├── reviewed.csv        # После HITL-2
│       ├── al_selected.csv     # Отобрано AL агентом
│       └── labelstudio_import.json
│
├── models/
│   └── final_model.pkl         # Обученная модель (vectorizer + classifier)
│
├── reports/
│   ├── data_quality_report.html  # Интерактивный HTML-отчёт качества данных
│   ├── annotation_spec.md        # Инструкция для разметчиков
│   ├── learning_curve.png        # Кривые обучения: entropy vs random
│   ├── al_report.json            # JSON с метриками AL эксперимента
│   ├── pipeline_metrics.json     # Метрики всех шагов пайплайна
│   └── final_report.md           # Финальный отчёт (5 разделов)
│
├── review_queue.csv              # ← HITL-2: примеры для ручной проверки
├── al_review_queue.csv           # ← HITL-3: AL-отобранные для ручной разметки
│
├── run_pipeline.py             # 🚀 ТОЧКА ВХОДА — запускает весь пайплайн
├── requirements.txt
├── .env.example
└── README.md
```

---

## Архитектура пайплайна

```
[DataCollectionAgent] ──────────────────────────────────────
  Kaggle + HuggingFace search → data/raw/dataset.csv
         │
         ▼
[DataQualityAgent] ─────────────────────────────────────────
  detect issues → fix → HTML report + clean.csv
         │
         ▼
  ❗ HITL-1: Human reviews quality_report.html
             Approves/rejects cleaning strategy
         │
         ▼
[AnnotationAgent] ──────────────────────────────────────────
  GPT-4o-mini auto-labels → annotation_spec.md + labelstudio.json
         │
         ▼
  ❗ HITL-2: Human fixes labels with confidence < 0.7
             review_queue.csv → review_queue_corrected.csv
         │
         ▼
[ActiveLearningAgent — select] ─────────────────────────────
  Entropy query → picks most uncertain pool examples
         │
         ▼
  ❗ HITL-3: Human labels AL-selected (most informative) examples
             al_review_queue.csv → al_review_queue_corrected.csv
         │
         ▼
[ActiveLearningAgent — train] ──────────────────────────────
  entropy vs random comparison → learning_curve.png + model.pkl
         │
         ▼
[ReportAgent] ──────────────────────────────────────────────
  final_report.md + DATA_CARD.md + pipeline_metrics.json
```

---

## Human-in-the-Loop — инструкция

### HITL-1: Подтверждение стратегии очистки
Пайплайн остановится и попросит вас:
1. Открыть `reports/data_quality_report.html` в браузере
2. Проверить: сколько дублей найдено, какие колонки имеют пропуски, есть ли выбросы
3. Нажать **Enter** для продолжения или **q** для отмены

### HITL-2: Правка автоматических меток
Пайплайн остановится и попросит вас:
1. Открыть файл `review_queue.csv` (примеры с confidence < 0.7)
2. Исправить неверные метки в столбце `label_auto` (значения: `positive` / `negative` / `neutral`)
3. Сохранить файл как `review_queue_corrected.csv`
4. Нажать **Enter** — пайплайн подхватит исправления

### HITL-3: Ручная разметка AL-примеров
1. Открыть `al_review_queue.csv` — наиболее неопределённые примеры по мнению модели
2. Заполнить / исправить метки
3. Сохранить как `al_review_queue_corrected.csv`
4. Нажать **Enter**

> Если пропустить шаги — введите `s`. Пайплайн продолжит с авто-метками.

---

## Зависимости

```
# requirements.txt
prefect>=2.14
openai>=1.0
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
scipy>=1.10
matplotlib>=3.7
kaggle>=1.6
huggingface-hub>=0.20
datasets>=2.16
python-dotenv>=1.0
certifi
```

---

## Переменные окружения

```bash
# .env
OPENAI_API_KEY=sk-...        # обязательно (для AnnotationAgent + DataQualityAgent)
KAGGLE_USERNAME=...           # опционально (для DataCollectionAgent)
KAGGLE_KEY=...                # опционально
```

> Без OPENAI_API_KEY авторазметка и объяснения стратегии недоступны.  
> Без Kaggle credentials — используется только HuggingFace + fallback.

---

## Воспроизводимость

Пайплайн **воспроизводим на чистом окружении**:
- Если Kaggle/HuggingFace недоступны → автоматически создаётся синтетический демо-датасет (600 строк, 3 класса)
- `--no-hitl` флаг пропускает все интерактивные паузы → подходит для CI/CD
- Все промежуточные файлы сохраняются → можно перезапустить с любого шага

---

*Проект создан как ML-портфолио | Track A: Active Learning*
