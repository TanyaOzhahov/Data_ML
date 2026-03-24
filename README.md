# ML Data Pipeline with Agents

## 📌 Описание

Проект реализует end-to-end ML pipeline с использованием агентов:

- DataCollectionAgent (поиск датасетов)
- DataQualityAgent (очистка данных)
- AnnotationAgent (авторазметка)
- ActiveLearningAgent (умный отбор данных)

Pipeline
Data Collection — поиск датасетов (Kaggle + HuggingFace)
Data Cleaning — удаление пропусков, дубликатов, выбросов
Auto Labeling — автоматическая разметка
Human-in-the-loop — ручная проверка низкоуверенных примеров
Active Learning — выбор информативных данных
Training — обучение модели

Метрики
Accuracy
F1-score
