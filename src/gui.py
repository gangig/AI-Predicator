import sys
import os
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QTableWidget, QTableWidgetItem, QGroupBox, QSpinBox,
                             QDoubleSpinBox, QComboBox, QTextEdit, QTabWidget,
                             QProgressBar, QMessageBox, QSplitter, QFrame)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import tensorflow as tf
from tensorflow import keras

from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.model import DroughtLSTMModel
from src.trainer import TrainingVisualizer

class TrainingThread(QThread):
    progress_signal = pyqtSignal(int, str)
    epoch_signal = pyqtSignal(dict)
    finished_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)

    def __init__(self, model, X_train, y_train, X_val, y_val, epochs, batch_size):
        super().__init__()
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.epochs = epochs
        self.batch_size = batch_size

    def run(self):
        try:
            class ProgressCallback(keras.callbacks.Callback):
                def __init__(self, thread):
                    super().__init__()
                    self.thread = thread

                def on_epoch_end(self, epoch, logs=None):
                    if logs is None:
                        logs = {}

                    progress = int((epoch + 1) / self.thread.epochs * 100)
                    msg = f"Эпоха {epoch + 1}/{self.thread.epochs} - " \
                          f"Loss: {logs.get('loss', 0):.4f} - " \
                          f"Accuracy: {logs.get('accuracy', 0):.4f}"

                    # Безопасная отправка сигнала
                    try:
                        self.thread.progress_signal.emit(progress, msg)
                    except:
                        pass

                    # Отправка метрик для графика
                    try:
                        self.thread.epoch_signal.emit({
                            'epoch': epoch + 1,
                            'loss': float(logs.get('loss', 0)),
                            'accuracy': float(logs.get('accuracy', 0)),
                            'val_loss': float(logs.get('val_loss', 0)),
                            'val_accuracy': float(logs.get('val_accuracy', 0))
                        })
                    except:
                        pass

            callback = ProgressCallback(self)

            # Создаем дополнительные callback'и с защитой
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=0  # Отключаем вывод в консоль
            )

            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=0  # Отключаем вывод в консоль
            )

            history = self.model.train(
                self.X_train, self.y_train,
                self.X_val, self.y_val,
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=[callback, early_stopping, reduce_lr]
            )
            self.finished_signal.emit(history)
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.error_signal.emit(f"{str(e)}\n\n{error_details}")

class DroughtPredictorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.data_loader = DataLoader()
        self.preprocessor = None
        self.model = None
        self.visualizer = TrainingVisualizer()
        self.data = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.training_history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        self.current_metrics = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('AI Drought Predictor - Система прогнозирования засухи')
        self.setGeometry(100, 100, 1400, 900)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        self.tabs = QTabWidget()

        self.data_preprocessing_tab = self.create_data_and_preprocessing_tab()
        self.tabs.addTab(self.data_preprocessing_tab, "1. Загрузка и предобработка")

        self.training_tab = self.create_training_tab()
        self.tabs.addTab(self.training_tab, "2. Обучение модели")

        self.results_tab = self.create_results_tab()
        self.tabs.addTab(self.results_tab, "3. Результаты")

        main_layout.addWidget(self.tabs)
        self.status_bar = self.statusBar()
        self.status_bar.showMessage('Готов к работе')

    def create_data_and_preprocessing_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        load_group = QGroupBox("Загрузка данных")
        load_layout = QHBoxLayout(load_group)

        self.btn_load_csv = QPushButton("Загрузить CSV")
        self.btn_load_csv.clicked.connect(self.load_csv)
        load_layout.addWidget(self.btn_load_csv)

        self.btn_load_json = QPushButton("Загрузить JSON")
        self.btn_load_json.clicked.connect(self.load_json)
        load_layout.addWidget(self.btn_load_json)

        self.btn_load_xml = QPushButton("Загрузить XML")
        self.btn_load_xml.clicked.connect(self.load_xml)
        load_layout.addWidget(self.btn_load_xml)

        self.btn_clear = QPushButton("Очистить данные")
        self.btn_clear.clicked.connect(self.clear_data)
        load_layout.addWidget(self.btn_clear)

        layout.addWidget(load_group)

        self.data_table = QTableWidget()
        self.data_table.setColumnCount(7)
        self.data_table.setHorizontalHeaderLabels([
            'Дата', 'Осадки (мм)', 'Темп. ср. (°C)',
            'Темп. мин (°C)', 'Темп. макс (°C)',
            'Влажность (%)', 'Давление (гПа)'
        ])
        layout.addWidget(self.data_table)

        self.data_info = QTextEdit()
        self.data_info.setMaximumHeight(80)
        self.data_info.setReadOnly(True)
        layout.addWidget(self.data_info)

        layout.addWidget(QLabel("<hr>"))

        prep_group = QGroupBox("Параметры предобработки")
        prep_layout = QHBoxLayout(prep_group)

        prep_layout.addWidget(QLabel("Длина последовательности (дни):"))
        self.seq_length_spin = QSpinBox()
        self.seq_length_spin.setRange(7, 90)
        self.seq_length_spin.setValue(30)
        prep_layout.addWidget(self.seq_length_spin)

        prep_layout.addWidget(QLabel("Доля тестовой выборки:"))
        self.test_size_spin = QDoubleSpinBox()
        self.test_size_spin.setRange(0.1, 0.4)
        self.test_size_spin.setValue(0.2)
        self.test_size_spin.setSingleStep(0.05)
        prep_layout.addWidget(self.test_size_spin)

        self.btn_preprocess = QPushButton("Выполнить предобработку")
        self.btn_preprocess.clicked.connect(self.preprocess_data)
        self.btn_preprocess.setEnabled(False)
        prep_layout.addWidget(self.btn_preprocess)

        layout.addWidget(prep_group)

        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(150)
        self.stats_text.setPlaceholderText("После предобработки здесь появится статистика...")
        layout.addWidget(self.stats_text)

        return widget

    def create_training_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        model_group = QGroupBox("Параметры модели")
        model_layout = QHBoxLayout(model_group)

        model_layout.addWidget(QLabel("LSTM слои:"))
        self.lstm1_spin = QSpinBox()
        self.lstm1_spin.setRange(16, 256)
        self.lstm1_spin.setValue(64)
        model_layout.addWidget(self.lstm1_spin)

        self.lstm2_spin = QSpinBox()
        self.lstm2_spin.setRange(16, 128)
        self.lstm2_spin.setValue(32)
        model_layout.addWidget(self.lstm2_spin)

        model_layout.addWidget(QLabel("Learning rate:"))
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 0.5)  # Диапазон от 0.0001 до 0.5
        self.lr_spin.setValue(0.001)  # Значение по умолчанию
        self.lr_spin.setSingleStep(0.001)  # Шаг при нажатии стрелок: 0.001
        self.lr_spin.setDecimals(5)  # Показываем до 5 знаков после запятой
        self.lr_spin.setPrefix("")  # Без префикса
        self.lr_spin.setSuffix("")  # Без суффикса
        self.lr_spin.setKeyboardTracking(False)  # Можно вводить с клавиатуры
        model_layout.addWidget(self.lr_spin)

        layout.addWidget(model_group)

        train_group = QGroupBox("Параметры обучения")
        train_layout = QHBoxLayout(train_group)

        train_layout.addWidget(QLabel("Эпохи:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(10, 200)
        self.epochs_spin.setValue(100)
        train_layout.addWidget(self.epochs_spin)

        train_layout.addWidget(QLabel("Batch size:"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(8, 128)
        self.batch_spin.setValue(32)
        train_layout.addWidget(self.batch_spin)

        self.btn_train = QPushButton("Начать обучение")
        self.btn_train.clicked.connect(self.start_training)
        self.btn_train.setEnabled(False)
        train_layout.addWidget(self.btn_train)

        layout.addWidget(train_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        layout.addWidget(self.log_text)

        self.training_plot = MplCanvas(self, width=10, height=6, dpi=100)
        layout.addWidget(self.training_plot)

        return widget

    def create_results_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        btn_layout = QHBoxLayout()

        self.btn_evaluate = QPushButton("Оценить модель (вручную)")
        self.btn_evaluate.clicked.connect(self.evaluate_model)
        self.btn_evaluate.setEnabled(False)
        btn_layout.addWidget(self.btn_evaluate)

        self.btn_save_model = QPushButton("Сохранить модель (.h5)")
        self.btn_save_model.clicked.connect(self.save_model)
        self.btn_save_model.setEnabled(False)
        btn_layout.addWidget(self.btn_save_model)

        self.btn_save_report = QPushButton("Сохранить отчет (.txt)")
        self.btn_save_report.clicked.connect(self.save_report)
        self.btn_save_report.setEnabled(False)
        btn_layout.addWidget(self.btn_save_report)

        layout.addLayout(btn_layout)

        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setMaximumHeight(200)
        layout.addWidget(self.metrics_text)

        self.results_plot = MplCanvas(self, width=10, height=6, dpi=100)
        layout.addWidget(self.results_plot)

        return widget

    def load_csv(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Загрузить CSV", "", "CSV Files (*.csv)")
        if filepath:
            try:
                self.data = self.data_loader.load_csv(filepath)
                self.display_data()
                self.status_bar.showMessage(f"Данные загружены: {filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка загрузки: {e}")

    def load_json(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Загрузить JSON", "", "JSON Files (*.json)")
        if filepath:
            try:
                self.data = self.data_loader.load_json(filepath)
                self.display_data()
                self.status_bar.showMessage(f"Данные загружены: {filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка загрузки: {e}")

    def load_xml(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Загрузить XML", "", "XML Files (*.xml)")
        if filepath:
            try:
                self.data = self.data_loader.load_xml(filepath)
                self.display_data()
                self.status_bar.showMessage(f"Данные загружены: {filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка загрузки: {e}")

    def display_data(self):
        if self.data is None: return
        self.data_table.setRowCount(len(self.data))
        for i, (_, row) in enumerate(self.data.iterrows()):
            self.data_table.setItem(i, 0, QTableWidgetItem(str(row['date'])))
            self.data_table.setItem(i, 1, QTableWidgetItem(str(row['precipitation'])))
            self.data_table.setItem(i, 2, QTableWidgetItem(str(row['temperature_avg'])))
            self.data_table.setItem(i, 3, QTableWidgetItem(str(row['temperature_min'])))
            self.data_table.setItem(i, 4, QTableWidgetItem(str(row['temperature_max'])))
            self.data_table.setItem(i, 5, QTableWidgetItem(str(row['humidity_avg'])))
            self.data_table.setItem(i, 6, QTableWidgetItem(str(row['pressure_avg'])))

        self.data_info.setText(f"✓ Загружено записей: {len(self.data)}\n"
                               f"✓ Период: {self.data['date'].min().date()} - {self.data['date'].max().date()}")

        self.btn_preprocess.setEnabled(True)
        self.status_bar.showMessage("Данные загружены. Нажмите 'Выполнить предобработку'")

    def clear_data(self):
        self.data = None
        self.data_table.setRowCount(0)
        self.data_info.clear()
        self.stats_text.clear()
        self.btn_preprocess.setEnabled(False)
        self.status_bar.showMessage("Данные очищены")

    def preprocess_data(self):
        if self.data is None:
            QMessageBox.warning(self, "Внимание", "Сначала загрузите данные")
            return
        try:
            seq_length = self.seq_length_spin.value()
            test_size = self.test_size_spin.value()

            self.preprocessor = DataPreprocessor(sequence_length=seq_length)
            X, y = self.preprocessor.prepare_data(self.data)

            split_idx = int(len(X) * (1 - test_size))
            self.X_train = X[:split_idx]
            self.y_train = y[:split_idx]
            self.X_test = X[split_idx:]
            self.y_test = y[split_idx:]

            # ПРОВЕРКА РАСПРЕДЕЛЕНИЯ КЛАССОВ
            train_unique, train_counts = np.unique(y[:split_idx], return_counts=True)
            test_unique, test_counts = np.unique(y[split_idx:], return_counts=True)

            print(f"\n=== РАСПРЕДЕЛЕНИЕ КЛАССОВ ===")
            print(f"Обучающая выборка: {dict(zip(train_unique, train_counts))}")
            print(f"Тестовая выборка: {dict(zip(test_unique, test_counts))}")

            if len(test_unique) < 2:
                QMessageBox.warning(self, "Внимание",
                                    f"⚠️ В тестовой выборке представлен только ОДИН класс!\n\n"
                                    f"Классы в тесте: {test_unique}\n"
                                    f"Это может привести к ошибкам при оценке.\n\n"
                                    f"Рекомендации:\n"
                                    f"1. Увеличьте объем данных\n"
                                    f"2. Измените долю тестовой выборки\n"
                                    f"3. Добавьте больше примеров засухи")

            self.stats_text.setText(f"Всего последовательностей: {len(X)}\n"
                                    f"Обучающая выборка: {len(self.X_train)}\n"
                                    f"Тестовая выборка: {len(self.X_test)}\n"
                                    f"Распределение классов (train): {dict(zip(train_unique, train_counts))}\n"
                                    f"Распределение классов (test): {dict(zip(test_unique, test_counts))}")

            self.btn_train.setEnabled(True)
            self.status_bar.showMessage("Данные подготовлены")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка предобработки: {e}")

    def start_training(self):
        if self.X_train is None:
            QMessageBox.warning(self, "Внимание", "Сначала выполните предобработку")
            return

        self.training_history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

        self.model = DroughtLSTMModel(
            sequence_length=self.seq_length_spin.value(),
            n_features=6
        )
        self.model.build_model(
            lstm_units=[self.lstm1_spin.value(), self.lstm2_spin.value()],
            dropout_rate=0.3,
            learning_rate=self.lr_spin.value()
        )

        self.training_thread = TrainingThread(
            self.model, self.X_train, self.y_train,
            self.X_test, self.y_test, self.epochs_spin.value(), self.batch_spin.value()
        )

        self.training_thread.progress_signal.connect(self.update_progress)
        self.training_thread.epoch_signal.connect(self.update_training_plot)
        self.training_thread.finished_signal.connect(self.training_finished)
        self.training_thread.error_signal.connect(self.training_error)

        self.btn_train.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.training_thread.start()

    def update_progress(self, progress, message):
        self.progress_bar.setValue(progress)
        self.log_text.append(message)
        self.log_text.ensureCursorVisible()

    def update_training_plot(self, metrics):
        self.training_history['loss'].append(metrics['loss'])
        self.training_history['accuracy'].append(metrics['accuracy'])
        self.training_history['val_loss'].append(metrics['val_loss'])
        self.training_history['val_accuracy'].append(metrics['val_accuracy'])

        self.training_plot.figure.clear()

        ax1 = self.training_plot.figure.add_subplot(121)
        ax2 = self.training_plot.figure.add_subplot(122)

        epochs = range(1, len(self.training_history['loss']) + 1)

        ax1.plot(epochs, self.training_history['loss'], 'b-', linewidth=2, label='Потери (train)')
        ax1.plot(epochs, self.training_history['val_loss'], 'r--', linewidth=2, label='Потери (val)')
        ax1.set_title('Функция потерь (Loss)', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Эпоха')
        ax1.set_ylabel('Loss')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, self.training_history['accuracy'], 'b-', linewidth=2, label='Точность (train)')
        ax2.plot(epochs, self.training_history['val_accuracy'], 'r--', linewidth=2, label='Точность (val)')
        ax2.set_title('Точность классификации (Accuracy)', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Эпоха')
        ax2.set_ylabel('Accuracy')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1.0])

        self.training_plot.figure.tight_layout()
        self.training_plot.draw()

    def training_finished(self, history):
        """Этот метод вызывается ПОСЛЕ завершения обучения. Программа НЕ должна закрываться."""
        try:
            self.progress_bar.setVisible(False)
            self.btn_train.setEnabled(True)
            self.btn_evaluate.setEnabled(True)
            self.btn_save_model.setEnabled(True)

            self.status_bar.showMessage("Обучение завершено успешно")

            # Показываем графики обучения на вкладке 2
            self.visualizer.plot_training_history(history)

            # ИСПРАВЛЕНИЕ: Заменяем фигуру целиком вместо попытки добавить оси
            fig = self.visualizer.plot_training_history(history)
            self.training_plot.figure = fig
            self.training_plot.draw()

            # Сообщаем пользователю и ПЕРЕХОДИМ на вкладку результатов
            QMessageBox.information(self, "Обучение завершено",
                                    "Модель успешно обучена!\n\n"
                                    "Сейчас программа переключится на вкладку результатов.")

            # Переключаемся на вкладку "Результаты" (индекс 2)
            self.tabs.setCurrentIndex(2)

            # Принудительно обновляем интерфейс, чтобы окно не зависло
            QApplication.processEvents()

            # Автоматически выполняем оценку
            self.evaluate_model()

        except Exception as e:
            # Если произошла ошибка, мы перехватываем её и показываем,
            # но программа останется открытой!
            import traceback
            error_details = traceback.format_exc()
            QMessageBox.critical(self, "Критическая ошибка после обучения",
                                 f"Произошла ошибка при выводе результатов, но программа останется открытой.\n\n"
                                 f"Ошибка: {str(e)}\n\n"
                                 f"Детали:\n{error_details}")

    def training_error(self, error_msg):
        self.progress_bar.setVisible(False)
        self.btn_train.setEnabled(True)
        QMessageBox.critical(self, "Ошибка обучения", error_msg)

    def evaluate_model(self):
        try:
            if self.model is None or self.X_test is None:
                QMessageBox.warning(self, "Внимание", "Сначала обучите модель")
                return

            self.status_bar.showMessage("Выполняется оценка модели...")
            QApplication.processEvents()

            metrics = self.model.evaluate(self.X_test, self.y_test)
            self.current_metrics = metrics

            # Расчет F1-score с защитой от деления на ноль
            precision = metrics['precision']
            recall = metrics['recall']
            if (precision + recall) > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0

            # Подсчет статистики из матрицы ошибок
            cm = metrics['confusion_matrix']
            tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
            total = tn + fp + fn + tp

            # ПОНЯТНЫЙ ТЕКСТОВЫЙ ОТЧЕТ
            metrics_text = "📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ МОДЕЛИ\n"
            metrics_text += "=" * 50 + "\n\n"

            metrics_text += "🎯 ОБЩАЯ ТОЧНОСТЬ:\n"
            metrics_text += f"   Правильных предсказаний: {metrics['accuracy']:.1%}\n"
            metrics_text += f"   Из {total} примеров верно: {tn + tp}\n\n"

            metrics_text += "📈 ДЕТАЛЬНАЯ СТАТИСТИКА:\n"
            metrics_text += f"   ✓ Правильно найдено 'нет засухи': {tn} ({tn / total:.1%})\n"
            metrics_text += f"   ✓ Правильно найдено 'засуха': {tp} ({tp / total:.1%})\n"
            metrics_text += f"   ✗ Ошибочно предсказана засуха: {fp}\n"
            metrics_text += f"   ✗ Пропущена настоящая засуха: {fn}\n\n"

            metrics_text += "📊 КАЧЕСТВО ПРОГНОЗИРОВАНИЯ:\n"
            metrics_text += f"   • Precision (точность класса): {precision:.1%}\n"
            metrics_text += f"     (из всех предсказанных засух, сколько реально были засухой)\n"
            metrics_text += f"   • Recall (полнота): {recall:.1%}\n"
            metrics_text += f"     (из всех реальных засух, сколько модель нашла)\n"
            metrics_text += f"   • F1-Score (баланс): {f1:.1%}\n"
            metrics_text += f"   • AUC-ROC (разделяющая способность): {metrics['auc']:.3f}\n\n"

            metrics_text += "⚠️ ИНТЕРПРЕТАЦИЯ:\n"
            if tp == 0 and fn > 0:
                metrics_text += "   ❌ МОДЕЛЬ НЕ НАХОДИТ ЗАСУХУ!\n"
                metrics_text += "   Причина: слишком мало примеров засухи в данных\n"
                metrics_text += "   Решение: увеличить данные или изменить порог классификации\n"
            elif recall < 0.5:
                metrics_text += "   ⚠️ Модель плохо находит засуху (низкий Recall)\n"
            elif precision < 0.5:
                metrics_text += "   ⚠️ Много ложных срабатываний (низкий Precision)\n"
            else:
                metrics_text += "   ✅ Модель работает хорошо!\n"

            metrics_text += "\n" + "=" * 50 + "\n"
            metrics_text += "МАТРИЦА ОШИБОК:\n"
            metrics_text += f"                Предсказано\n"
            metrics_text += f"                Нет засухи   Засуха\n"
            metrics_text += f"Реально  Нет засухи    {tn:4d}      {fp:4d}\n"
            metrics_text += f"         Засуха        {fn:4d}      {tp:4d}\n"

            self.metrics_text.setText(metrics_text)

            # УЛУЧШЕННЫЙ ГРАФИК
            self.visualizer.plot_final_metrics(metrics)
            self.results_plot.figure.clear()

            # Создаем 2 подграфика: матрица ошибок + столбчатая диаграмма
            fig = plt.figure(figsize=(12, 6))
            gs = plt.GridSpec(1, 2, width_ratios=[1, 1.2])

            # График 1: Матрица ошибок
            ax1 = fig.add_subplot(gs[0])
            im = ax1.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            fig.colorbar(im, ax=ax1, shrink=0.8)

            ax1.set(xticks=np.arange(cm.shape[1]),
                    yticks=np.arange(cm.shape[0]),
                    xticklabels=['Нет засухи', 'Засуха'],
                    yticklabels=['Нет засухи', 'Засуха'],
                    title='Матрица ошибок\n(сколько правильно/неправильно)',
                    ylabel='Реальный класс',
                    xlabel='Предсказанный класс')
            plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

            # Добавляем числа в ячейки
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax1.text(j, i, format(cm[i, j], 'd'),
                             ha="center", va="center",
                             color="white" if cm[i, j] > thresh else "black",
                             fontsize=14, fontweight='bold')

            # График 2: Столбчатая диаграмма метрик
            ax2 = fig.add_subplot(gs[1])
            metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            metric_values = [metrics['accuracy'], precision, recall, f1]
            colors = ['#2ecc71' if v > 0.7 else '#f39c12' if v > 0.4 else '#e74c3c'
                      for v in metric_values]

            bars = ax2.bar(metric_names, metric_values, color=colors,
                           edgecolor='black', linewidth=1.5, alpha=0.8)
            ax2.set_ylim([0, 1.0])
            ax2.set_title('Метрики качества модели\n(чем выше, тем лучше)',
                          fontsize=12, fontweight='bold')
            ax2.set_ylabel('Значение (0-1)')
            ax2.axhline(y=0.7, color='green', linestyle='--', alpha=0.5,
                        label='Хороший уровень (0.7)')
            ax2.legend(loc='lower right')
            ax2.grid(axis='y', alpha=0.3)

            # Добавляем значения на столбцы
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                         f'{height:.1%}', ha='center', va='bottom',
                         fontsize=11, fontweight='bold')

            plt.tight_layout()
            self.results_plot.figure = fig
            self.results_plot.draw()

            self.status_bar.showMessage("Модель оценена. Результаты готовы!")
            self.btn_save_report.setEnabled(True)

            # Сообщение с кратким итогом
            if tp == 0:
                msg = ("⚠️ ВНИМАНИЕ!\n\n"
                       "Модель НЕ НАУЧИЛАСЬ предсказывать засуху.\n\n"
                       f"Всего правильно предсказано: {tn + tp} из {total}\n"
                       f"Засух найдено: {tp} из {fn + tp}\n\n"
                       "Рекомендации:\n"
                       "1. Добавьте больше данных с засушливыми годами\n"
                       "2. Увеличьте количество эпох обучения\n"
                       "3. Измените порог классификации")
            else:
                msg = (f"✅ Оценка завершена!\n\n"
                       f"Точность: {metrics['accuracy']:.1%}\n"
                       f"Найдено засух: {tp} из {fn + tp} (Recall: {recall:.1%})\n"
                       f"F1-Score: {f1:.1%}")

            QMessageBox.information(self, "Результаты оценки", msg)

        except Exception as e:
            QMessageBox.critical(self, "Ошибка оценки", f"Не удалось оценить модель: {e}")

    def save_model(self):
        if self.model is None: return
        filepath, _ = QFileDialog.getSaveFileName(self, "Сохранить модель", "models/model.h5", "H5 Files (*.h5)")
        if filepath:
            self.model.save_model(filepath)
            QMessageBox.information(self, "Успех", f"Модель сохранена: {filepath}")

    def save_report(self):
        """Сохранение текстового отчета с метриками"""
        if self.current_metrics is None:
            QMessageBox.warning(self, "Внимание", "Сначала оцените модель")
            return

        filepath, _ = QFileDialog.getSaveFileName(self, "Сохранить отчет", "report.txt", "Text Files (*.txt)")
        if filepath:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write("ОТЧЕТ О РАБОТЕ СИСТЕМЫ ПРОГНОЗИРОВАНИЯ ЗАСУХИ\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(self.metrics_text.toPlainText())
                    f.write("\n\n" + "=" * 50 + "\n")
                    f.write("Дата генерации: " + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'))

                QMessageBox.information(self, "Успех", f"Отчет сохранен: {filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить отчет: {e}")


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.figure.add_subplot(111)
        super().__init__(self.figure)


def main():
    app = QApplication(sys.argv)
    font = QFont("Arial", 10)
    app.setFont(font)
    window = DroughtPredictorGUI()
    window.show()
    # ВАЖНО: sys.exit должен быть только здесь. Это заставляет программу ждать закрытия окна.
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()