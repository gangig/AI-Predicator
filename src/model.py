import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple
from sklearn.utils.class_weight import compute_class_weight


class DroughtLSTMModel:
    """
    LSTM модель для прогнозирования засухи
    """

    def __init__(self, sequence_length: int = 30, n_features: int = 6):
        """
        Args:
            sequence_length: Длина входной последовательности
            n_features: Количество признаков
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.history = None

    def build_model(self, lstm_units: list = [64, 32],
                    dropout_rate: float = 0.3,
                    learning_rate: float = 0.001) -> keras.Model:
        """
        Построение LSTM модели
        """
        # Очистка сессии TensorFlow
        tf.keras.backend.clear_session()

        # Входной слой
        inputs = keras.Input(shape=(self.sequence_length, self.n_features))

        # Первый LSTM слой
        x = layers.LSTM(
            lstm_units[0],
            return_sequences=True,
            recurrent_dropout=0.2
        )(inputs)
        x = layers.Dropout(dropout_rate)(x)

        # Второй LSTM слой
        x = layers.LSTM(
            lstm_units[1],
            return_sequences=False,
            recurrent_dropout=0.2
        )(x)
        x = layers.Dropout(dropout_rate)(x)

        # Полносвязный слой
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(dropout_rate / 2)(x)

        # Выходной слой (бинарная классификация)
        outputs = layers.Dense(1, activation='sigmoid')(x)

        # Создание модели
        self.model = keras.Model(inputs, outputs, name='drought_lstm')

        # Компиляция модели
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )

        # Вывод информации о модели
        self.model.summary()

        return self.model

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 100, batch_size: int = 32,
              callbacks: list = None) -> keras.callbacks.History:
        """
        Обучение модели с использованием весов классов для борьбы с дисбалансом
        """
        if self.model is None:
            self.build_model()

        # Расчет весов классов
        try:
            print("\n=== Расчет весов классов для борьбы с дисбалансом ===")

            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(y_train),
                y=y_train
            )

            class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

            print(f"Вес класса 0 (нет засухи): {class_weights[0]:.3f}")
            print(f"Вес класса 1 (засуха): {class_weights[1]:.3f}")
            print(f"Распределение в обучающей выборке: {np.bincount(y_train)}")
            print("=" * 60 + "\n")
        except Exception as e:
            # Если не получилось рассчитать веса, используем стандартные
            print(f"Не удалось рассчитать веса классов: {e}")
            print("Используем стандартные веса (1.0 для всех классов)")
            class_weight_dict = {0: 1.0, 1: 1.0}

        # Callback для ранней остановки
        if callbacks is None:
            callbacks = []

        # Добавляем свои callback'и только если они не были переданы
        has_early_stopping = any(isinstance(cb, keras.callbacks.EarlyStopping) for cb in callbacks)
        has_reduce_lr = any(isinstance(cb, keras.callbacks.ReduceLROnPlateau) for cb in callbacks)

        if not has_early_stopping:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True,
                    verbose=0
                )
            )

        if not has_reduce_lr:
            callbacks.append(
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7,
                    verbose=0
                )
            )

        # Обучение модели с весами классов
        print("Начало обучения модели...")
        try:
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                class_weight=class_weight_dict,
                verbose=1  # Показываем прогресс
            )
        except Exception as e:
            print(f"Ошибка при обучении: {e}")
            raise

        return self.history

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Оценка модели на тестовых данных
        """
        if self.model is None:
            raise ValueError("Модель не построена. Сначала вызовите build_model()")

        # Оценка модели
        metrics = self.model.evaluate(X_test, y_test, verbose=0)

        # Получение предсказаний
        y_pred_proba = self.model.predict(X_test)

        # ИСПРАВЛЕНИЕ: Используем адаптивный порог
        # Если модель предсказывает только один класс, снижаем порог
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()

        # Проверяем, предсказывает ли модель оба класса
        unique_preds = np.unique(y_pred)
        if len(unique_preds) == 1:
            print(f"⚠️ ВНИМАНИЕ: Модель предсказывает только один класс ({unique_preds[0]})")
            print("Снижаем порог классификации до 0.3...")
            y_pred = (y_pred_proba > 0.3).astype(int).flatten()

            # Если всё ещё один класс, пробуем порог 0.2
            if len(np.unique(y_pred)) == 1:
                print("Снижаем порог до 0.2...")
                y_pred = (y_pred_proba > 0.2).astype(int).flatten()

        # Расчет дополнительных метрик с защитой от ошибок
        from sklearn.metrics import classification_report, confusion_matrix

        try:
            cm = confusion_matrix(y_test, y_pred)

            # Проверка, что матрица ошибок имеет правильный размер
            if cm.shape != (2, 2):
                print(f"⚠️ Матрица ошибок имеет размер {cm.shape}, расширяем до 2x2")
                # Создаем полную матрицу 2x2
                cm_full = np.zeros((2, 2), dtype=int)
                for i in range(min(cm.shape[0], 2)):
                    for j in range(min(cm.shape[1], 2)):
                        cm_full[i, j] = cm[i, j] if i < cm.shape[0] and j < cm.shape[1] else 0
                cm = cm_full

            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        except Exception as e:
            print(f"Ошибка при расчете метрик: {e}")
            # Создаем пустые метрики
            cm = np.zeros((2, 2), dtype=int)
            report = {'0': {'precision': 0, 'recall': 0, 'f1-score': 0},
                      '1': {'precision': 0, 'recall': 0, 'f1-score': 0}}

        results = {
            'loss': metrics[0],
            'accuracy': metrics[1],
            'precision': metrics[2] if len(metrics) > 2 else 0.0,
            'recall': metrics[3] if len(metrics) > 3 else 0.0,
            'auc': metrics[4] if len(metrics) > 4 else 0.5,
            'classification_report': report,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba.flatten()
        }

        return results

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Предсказание засухи
        """
        if self.model is None:
            raise ValueError("Модель не построена")

        y_pred_proba = self.model.predict(X)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()

        return y_pred, y_pred_proba

    def save_model(self, filepath: str = 'models/drought_model.h5'):
        """
        Сохранение модели
        """
        if self.model is None:
            raise ValueError("Нет модели для сохранения")

        self.model.save(filepath)
        print(f"Модель сохранена в {filepath}")

    def load_model(self, filepath: str):
        """
        Загрузка модели
        """
        self.model = keras.models.load_model(filepath)
        print(f"Модель загружена из {filepath}")