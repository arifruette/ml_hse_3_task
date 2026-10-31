# ml_hse_3_task

**[Открыть приложение](https://mlhse3-bos2kpnue2j5w7bcw25fuo.streamlit.app/)**

Домашнее задание 3: логистическая регрессия, метрики классификации, балансировка классов и приложение Streamlit для оценки вин.

- [Выполненный ноутбук](HW3_MMDA.ipynb)
- [Приложение и инструкция запуска](wine_app/README.md)
- [Исходные данные](Data/wine_quality.csv)

## Запуск приложения

Python 3.12:

```bash
python -m pip install -r wine_app/requirements.txt
python -m streamlit run wine_app/app.py
```

В Streamlit Community Cloud точка входа: `wine_app/app.py`, ветка `main`, Python 3.12.
Модель уже обучена; приложение загружает сохраненный пайплайн и не обучает его заново.

Для вёрстки веб-интерфейса приложения использовался Codex. (это не пранк)
