import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

APP_DIR = Path(__file__).resolve().parent
st.set_page_config(page_title='Винная подборка', page_icon='🍷', layout='wide')


@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / 'model.joblib')


@st.cache_data
def load_test():
    return pd.read_csv(APP_DIR / 'test.csv')


@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


def batch_predictions(frame, model, feature_names, threshold):
    missing = sorted(set(feature_names) - set(frame.columns))
    if missing:
        raise ValueError('Не хватает столбцов: ' + ', '.join(missing))
    if frame.empty:
        raise ValueError('Файл не содержит строк с данными.')
    try:
        features = frame.loc[:, feature_names].apply(pd.to_numeric, errors='raise')
    except (TypeError, ValueError) as error:
        raise ValueError('Все признаки должны содержать числа.') from error
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError('В признаках есть пропуски или бесконечные значения.')
    if not features['is_red'].isin([0, 1]).all():
        raise ValueError('Столбец is_red должен содержать 0 (белое) или 1 (красное).')
    result = frame.copy()
    result['probability'] = model.predict_proba(features)[:, 1]
    result['prediction'] = (result['probability'] >= threshold).astype(int)
    return result


model = load_model()
test = load_test()
X_test = test.drop(columns='target')
y_test = test['target']
feature_names = list(X_test.columns)
scores = test_proba(model, X_test)
config = json.loads((APP_DIR / 'config.json').read_text(encoding='utf-8'))

st.title('🍷 Винная подборка')
st.write('Оценка вина по лабораторному анализу: хорошим считается вино с экспертной оценкой от 7.')
threshold = st.sidebar.slider('Порог отбора', min_value=0.0, max_value=1.0,
                              value=float(config['threshold']), step=0.001,
                              format='%.3f', key='threshold')
st.sidebar.caption('Один порог применяется к оценке вина, метрикам и загруженному файлу.')
single_tab, metrics_tab, batch_tab, weights_tab = st.tabs(
    ['Оценка одного вина', 'Модельные метрики', 'Пакетное предсказание', 'Что выучила модель'])

labels = {
    'fixed_acidity': 'Нелетучая кислотность', 'volatile_acidity': 'Летучая кислотность',
    'citric_acid': 'Лимонная кислота', 'residual_sugar': 'Остаточный сахар',
    'chlorides': 'Хлориды', 'free_sulfur_dioxide': 'Свободный диоксид серы',
    'total_sulfur_dioxide': 'Общий диоксид серы', 'density': 'Плотность',
    'pH': 'pH', 'sulphates': 'Сульфаты', 'alcohol': 'Алкоголь, %', 'is_red': 'Красное вино'}

with single_tab:
    st.subheader('Лабораторные показатели')
    values = {}
    cols = st.columns(3)
    for index, feature in enumerate(feature_names):
        with cols[index % 3]:
            if feature == 'is_red':
                values[feature] = int(st.toggle('Красное вино', value=bool(X_test[feature].median())))
            else:
                values[feature] = st.number_input(
                    labels[feature], min_value=float(X_test[feature].min()),
                    max_value=float(X_test[feature].max()), value=float(X_test[feature].median()),
                    step=0.0001 if feature == 'density' else 0.001,
                    format='%.4f' if feature == 'density' else '%.3f', key=f'input_{feature}')
    sample = pd.DataFrame([values], columns=feature_names)
    probability = float(model.predict_proba(sample)[0, 1])
    st.metric('Вероятность хорошего вина', f'{probability:.1%}')
    if probability >= threshold:
        st.success('Добавить в подборку')
    else:
        st.info('Не добавлять в подборку при текущем пороге')
    st.caption('Диапазоны ввода и начальные значения взяты из test.csv. Это демо, а не новая настройка модели.')

with metrics_tab:
    st.subheader('Качество на отложенных данных')
    pred = scores >= threshold
    precision = metrics.precision_score(y_test, pred, zero_division=0)
    recall = metrics.recall_score(y_test, pred, zero_division=0)
    f1 = metrics.f1_score(y_test, pred, zero_division=0)
    metric_cols = st.columns(3)
    for col, name, value in zip(metric_cols, ['Precision', 'Recall', 'F1'], [precision, recall, f1]):
        col.metric(name, f'{value:.3f}')
    st.write('Если увеличить порог, подборка станет меньше: модель будет отбирать вина строже. '
             'Recall при этом не растет. Если уменьшить порог, хороших вин пропустим меньше, '
             'но в подборку могут чаще попадать посредственные. Precision не обязан меняться монотонно.')
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    metrics.ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=['Остальные', 'Хорошие'], ax=axes[0], cmap='Blues', colorbar=False)
    axes[0].set_title('Матрица ошибок')
    metrics.RocCurveDisplay.from_predictions(y_test, scores, ax=axes[1], name='Логрегрессия')
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    axes[1].scatter(fp / (fp + tn), recall, color='red', label=f'Порог {threshold:.3f}')
    axes[1].plot([0, 1], [0, 1], '--', color='gray')
    axes[1].legend()
    metrics.PrecisionRecallDisplay.from_predictions(y_test, scores, ax=axes[2], name='Логрегрессия')
    axes[2].scatter(recall, precision, color='red', label=f'Порог {threshold:.3f}')
    axes[2].axhline(y_test.mean(), ls='--', color='gray', label='Доля хороших вин')
    axes[2].legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption('Изменение порога не меняет ROC-AUC и AP: они оценивают ранжирование. '
               'Этот тест уже использован для отчета; настройка по нему не дает независимой оценки качества.')

with batch_tab:
    st.subheader('Оценить вина из CSV')
    st.write('Загрузите CSV с теми же 12 признаками, без target. Порядок столбцов может отличаться.')
    st.download_button('Скачать пример CSV', X_test.head(5).to_csv(index=False).encode('utf-8'),
                       file_name='wine_example.csv', mime='text/csv')
    uploaded = st.file_uploader('CSV с признаками вин', type=['csv'])
    if uploaded is not None:
        try:
            result = batch_predictions(pd.read_csv(uploaded), model, feature_names, threshold)
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
            st.error(f'Не удалось обработать файл. {error}')
        else:
            st.dataframe(result, width='stretch', hide_index=True)
            st.download_button('Скачать предсказания', result.to_csv(index=False).encode('utf-8'),
                               file_name='wine_predictions.csv', mime='text/csv')

with weights_tab:
    st.subheader('Какие признаки связаны с хорошим вином')
    weights = pd.Series(model[-1].coef_.ravel(), index=feature_names).sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh([labels[name] for name in weights.index], weights.values,
            color=np.where(weights.values >= 0, '#397862', '#ad5061'))
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('Коэффициент после стандартизации')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.write('Сильнее всего за хорошее вино голосуют: ' +
             ', '.join(labels[name] for name in weights.nlargest(3).index) + '.')
    st.write('Сильнее всего против: ' +
             ', '.join(labels[name] for name in weights.nsmallest(3).index) + '.')
    st.caption('Сравниваются изменения на одно стандартное отклонение при прочих равных. '
               'Это связи в модели, а не доказанные причины качества вина.')
