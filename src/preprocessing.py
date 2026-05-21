import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple


class DataPreprocessor:
    """
    Класс для предобработки климатических данных
    """

    def __init__(self, sequence_length: int = 30):
        """
        Args:
            sequence_length: Длина последовательности для LSTM (дни)
        """
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fill_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Заполнение пропущенных значений методом линейной интерполяции
        """
        data_filled = data.copy()

        # Числовые колонки для интерполяции
        numeric_cols = ['precipitation', 'temperature_avg', 'temperature_min',
                        'temperature_max', 'humidity_avg', 'pressure_avg']

        for col in numeric_cols:
            if col in data_filled.columns:
                # Линейная интерполяция пропущенных значений
                data_filled[col] = data_filled[col].interpolate(method='linear')
                # Заполнение крайних значений
                data_filled[col] = data_filled[col].bfill().ffill()

        print("Пропущенные значения заполнены")
        return data_filled

    def calculate_drought_indices(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Расчет индексов засухи
        """
        data_with_indices = data.copy()

        # Расчет скользящего среднего осадков за 30 дней
        data_with_indices['precip_30d'] = data_with_indices['precipitation'].rolling(
            window=30, min_periods=1
        ).sum()

        # Расчет скользящего среднего температуры за 7 дней
        data_with_indices['temp_7d'] = data_with_indices['temperature_avg'].rolling(
            window=7, min_periods=1
        ).mean()

        # Индекс дефицита осадков
        monthly_avg = data_with_indices.groupby(data_with_indices['date'].dt.month)['precipitation'].mean()
        data_with_indices['precip_deficit'] = data_with_indices.apply(
            lambda row: row['precipitation'] / monthly_avg[row['date'].month]
            if monthly_avg[row['date'].month] > 0 else 1,
            axis=1
        )

        print("Индексы засухи рассчитаны")
        return data_with_indices

    def normalize_data(self, data: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Нормализация данных с использованием StandardScaler
        """
        # Выбор признаков для модели
        feature_columns = ['precipitation', 'temperature_avg', 'humidity_avg',
                           'pressure_avg', 'precip_30d', 'temp_7d']

        # Проверка наличия всех колонок
        available_cols = [col for col in feature_columns if col in data.columns]

        features = data[available_cols].values

        if fit:
            # Обучение scaler на данных
            self.scaler.fit(features)
            self.is_fitted = True
            print("Scaler обучен на данных")

        # Нормализация данных
        normalized = self.scaler.transform(features)

        return normalized

    def create_sequences(self, data: np.ndarray, target_col: str,
                         data_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Создание последовательностей для LSTM
        """
        X = []
        y = []

        # Создание целевой переменной (бинарная классификация засухи)
        target_data = data_df[target_col].values

        for i in range(len(data) - self.sequence_length):
            # Последовательность входных данных
            seq = data[i:(i + self.sequence_length)]
            X.append(seq)

            # Целевое значение: засуха в следующий период
            future_precip = np.mean(target_data[i + self.sequence_length:i + self.sequence_length + 7])
            future_temp = data_df['temperature_avg'].iloc[i + self.sequence_length:i + self.sequence_length + 7].mean()

            # Засуха: осадки < 5мм и температура > 25°C
            is_drought = 1 if (future_precip < 5 and future_temp > 25) else 0
            y.append(is_drought)

        X = np.array(X)
        y = np.array(y)

        print(f"Создано {len(X)} последовательностей длиной {self.sequence_length}")
        print(f"Распределение классов: {np.bincount(y)}")

        return X, y

    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Полный пайплайн подготовки данных
        """
        # Заполнение пропусков
        data_clean = self.fill_missing_values(data)

        # Расчет индексов
        data_with_indices = self.calculate_drought_indices(data_clean)

        # Нормализация
        normalized_data = self.normalize_data(data_with_indices)

        # Создание последовательностей
        X, y = self.create_sequences(normalized_data, 'precipitation', data_with_indices)

        return X, y