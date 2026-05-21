import pandas as pd
import numpy as np
from typing import Tuple, Optional
import json
import xml.etree.ElementTree as ET


class DataLoader:
    """
    Класс для загрузки климатических данных из различных форматов:
    CSV, JSON, XML
    """

    def __init__(self):
        self.data = None
        self.required_columns = [
            'date', 'precipitation', 'temperature_avg',
            'temperature_min', 'temperature_max', 'humidity_avg', 'pressure_avg'
        ]

    def load_csv(self, filepath: str) -> pd.DataFrame:
        """
        Загрузка данных из CSV файла
        """
        try:
            self.data = pd.read_csv(filepath, encoding='utf-8')

            missing_cols = [col for col in self.required_columns if col not in self.data.columns]
            if missing_cols:
                raise ValueError(f"Отсутствуют необходимые колонки: {missing_cols}")

            self.data['date'] = pd.to_datetime(self.data['date'])
            self.data = self.data.sort_values('date').reset_index(drop=True)

            print(f"Успешно загружено {len(self.data)} записей из CSV")
            return self.data

        except Exception as e:
            print(f"Ошибка при загрузке CSV: {e}")
            raise

    def load_json(self, filepath: str) -> pd.DataFrame:
        """
        Загрузка данных из JSON файла
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            # Поддержка разных структур JSON
            if isinstance(json_data, list):
                self.data = pd.DataFrame(json_data)
            elif isinstance(json_data, dict):
                # Если данные вложены в ключ
                for key in ['data', 'records', 'weather', 'observations']:
                    if key in json_data:
                        self.data = pd.DataFrame(json_data[key])
                        break
                else:
                    self.data = pd.DataFrame(json_data)
            else:
                raise ValueError("Неподдерживаемый формат JSON")

            missing_cols = [col for col in self.required_columns if col not in self.data.columns]
            if missing_cols:
                raise ValueError(f"Отсутствуют необходимые колонки: {missing_cols}")

            self.data['date'] = pd.to_datetime(self.data['date'])
            self.data = self.data.sort_values('date').reset_index(drop=True)

            print(f"Успешно загружено {len(self.data)} записей из JSON")
            return self.data

        except Exception as e:
            print(f"Ошибка при загрузке JSON: {e}")
            raise

    def load_xml(self, filepath: str) -> pd.DataFrame:
        """
        Загрузка данных из XML файла
        """
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()

            records = []

            # Поиск записей в разных возможных тегах
            for tag in ['record', 'observation', 'data', 'weather', 'entry']:
                elements = root.findall(f'.//{tag}')
                if elements:
                    for elem in elements:
                        row = {}
                        for child in elem:
                            row[child.tag] = child.text
                        if row:
                            records.append(row)
                    break

            if not records:
                # Альтернативный парсинг - все дочерние элементы корня
                for child in root:
                    row = {}
                    for subchild in child:
                        row[subchild.tag] = subchild.text
                    if row:
                        records.append(row)

            if not records:
                raise ValueError("Не найдены данные в XML файле")

            self.data = pd.DataFrame(records)

            missing_cols = [col for col in self.required_columns if col not in self.data.columns]
            if missing_cols:
                raise ValueError(f"Отсутствуют необходимые колонки: {missing_cols}")

            self.data['date'] = pd.to_datetime(self.data['date'])
            self.data = self.data.sort_values('date').reset_index(drop=True)

            print(f"Успешно загружено {len(self.data)} записей из XML")
            return self.data

        except Exception as e:
            print(f"Ошибка при загрузке XML: {e}")
            raise

    def get_data(self) -> pd.DataFrame:
        """Возвращает загруженные данные"""
        if self.data is None:
            raise ValueError("Данные не загружены. Сначала вызовите метод загрузки.")
        return self.data