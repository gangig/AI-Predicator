import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
import os


class TrainingVisualizer:
    def __init__(self, save_dir=None):
        # Берем путь из переменных окружения, если он задан (после сборки)
        # Иначе используем дефолтный
        if save_dir is None:
            self.save_dir = os.environ.get('AI_DP_RESULTS_DIR', 'results')
        else:
            self.save_dir = save_dir

        # Убеждаемся, что папка существует
        os.makedirs(self.save_dir, exist_ok=True)

    def plot_training_history(self, history) -> plt.Figure:
        """
        Построение графиков истории обучения с русскими подписями
        """
        # Создание фигуры с подграфиками
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Метрики обучения модели', fontsize=16, fontweight='bold', y=0.995)

        epochs = range(1, len(history.history['loss']) + 1)

        # 1. График точности
        axes[0, 0].plot(epochs, history.history['accuracy'], 'b-', linewidth=2, label='Точность (обучение)')
        if 'val_accuracy' in history.history:
            axes[0, 0].plot(epochs, history.history['val_accuracy'], 'r--', linewidth=2, label='Точность (валидация)')
        axes[0, 0].set_title('Точность классификации', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Эпоха')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, linestyle='--')
        axes[0, 0].set_ylim([0, 1.0])

        # 2. График потерь
        axes[0, 1].plot(epochs, history.history['loss'], 'b-', linewidth=2, label='Потери (обучение)')
        if 'val_loss' in history.history:
            axes[0, 1].plot(epochs, history.history['val_loss'], 'r--', linewidth=2, label='Потери (валидация)')
        axes[0, 1].set_title('Функция потерь (Loss)', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Эпоха')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, linestyle='--')

        # 3. Precision
        if 'precision' in history.history:
            axes[1, 0].plot(epochs, history.history['precision'], 'g-', linewidth=2, label='Precision (обучение)')
            if 'val_precision' in history.history:
                axes[1, 0].plot(epochs, history.history['val_precision'], 'm--', linewidth=2,
                                label='Precision (валидация)')
            axes[1, 0].set_title('Точность выявления засухи (Precision)', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('Эпоха')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3, linestyle='--')
            axes[1, 0].set_ylim([0, 1.0])

        # 4. AUC
        if 'auc' in history.history:
            axes[1, 1].plot(epochs, history.history['auc'], 'c-', linewidth=2, label='AUC-ROC (обучение)')
            if 'val_auc' in history.history:
                axes[1, 1].plot(epochs, history.history['val_auc'], 'y--', linewidth=2, label='AUC-ROC (валидация)')
            axes[1, 1].set_title('Площадь под ROC-кривой (AUC-ROC)', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Эпоха')
            axes[1, 1].set_ylabel('AUC')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3, linestyle='--')
            axes[1, 1].set_ylim([0.5, 1.0])

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = os.path.join(self.save_dir, 'training_history.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig

    def plot_final_metrics(self, metrics: dict):
        """
        Финальные метрики после обучения
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Итоговые метрики качества модели', fontsize=16, fontweight='bold', y=0.995)

        # 1. Accuracy
        accuracy = metrics.get('accuracy', 0)
        axes[0, 0].bar(['Точность (Accuracy)'], [accuracy], color=['green'], alpha=0.7, edgecolor='black')
        axes[0, 0].set_ylim([0, 1.0])
        axes[0, 0].set_title('Итоговая точность', fontsize=12, fontweight='bold')
        axes[0, 0].text(0, accuracy + 0.02, f'{accuracy:.2%}', ha='center', va='bottom', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3, axis='y', linestyle='--')

        # 2. Precision, Recall, F1
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        names = ['Precision', 'Recall', 'F1-Score']
        values = [precision, recall, f1_score]
        colors = ['blue', 'orange', 'red']
        bars = axes[0, 1].bar(names, values, color=colors, alpha=0.7, edgecolor='black')
        axes[0, 1].set_ylim([0, 1.0])
        axes[0, 1].set_title('Метрики качества', fontsize=12, fontweight='bold')
        for bar in bars:
            axes[0, 1].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
                            f'{bar.get_height():.2f}', ha='center', va='bottom', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='y', linestyle='--')

        # 3. Confusion Matrix
        cm = metrics.get('confusion_matrix', np.zeros((2, 2)))
        im = axes[1, 0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        fig.colorbar(im, ax=axes[1, 0])
        axes[1, 0].set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
                       xticklabels=['Нет засухи', 'Засуха'], yticklabels=['Нет засухи', 'Засуха'],
                       title='Матрица ошибок', ylabel='Истинный', xlabel='Предсказанный')
        plt.setp(axes[1, 0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axes[1, 0].text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                                color="white" if cm[i, j] > thresh else "black", fontweight='bold')

        # 4. AUC
        auc_score = metrics.get('auc', 0)
        axes[1, 1].bar(['AUC-ROC'], [auc_score], color=['purple'], alpha=0.7, edgecolor='black')
        axes[1, 1].set_ylim([0.5, 1.0])
        axes[1, 1].set_title('ROC-AUC', fontsize=12, fontweight='bold')
        axes[1, 1].text(0, auc_score + 0.02, f'{auc_score:.3f}', ha='center', va='bottom', fontsize=14,
                        fontweight='bold')
        axes[1, 1].axhline(y=0.5, color='r', linestyle='--', label='Случайное угадывание')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y', linestyle='--')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = os.path.join(self.save_dir, 'final_metrics.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig