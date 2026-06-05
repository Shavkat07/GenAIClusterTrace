# src/stages/split.py
import sys
import logging
from pathlib import Path
import yaml
import pandas as pd

logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s [%(levelname)s] %(message)s",
	handlers=[logging.StreamHandler(sys.stdout)]
)


def load_params(params_path: str = "params.yaml") -> dict:
	with open(params_path, "r") as f:
		return yaml.safe_load(f)


def split_data():
	logging.info("Старт этапа разделения данных на Train/Test...")
	
	# 1. Загрузка параметров
	config = load_params()
	split_config = config["split"]
	
	input_path = Path(split_config["input_path"])
	output_dir = Path(split_config["output_dir"])
	test_size = float(split_config["test_size"])
	
	# 2. Чтение подготовленной матрицы признаков
	if not input_path.exists():
		logging.error(f"Файл признаков не найден: {input_path}")
		sys.exit(1)
	
	df = pd.read_csv(input_path)
	
	# Обеспечим правильный формат времени для точной сортировки
	df['datetime'] = pd.to_datetime(df['datetime'])
	df = df.sort_values(by='datetime').reset_index(drop=True)
	
	# 3. Хронологический расчет точки разделения
	# Находим уникальные временные метки, чтобы разделение было честным по времени
	unique_timestamps = sorted(df['datetime'].unique())
	split_idx = int(len(unique_timestamps) * (1 - test_size))
	split_datetime = unique_timestamps[split_idx]
	
	logging.info(f"Глобальная дата разделения выборки (Временной порог): {split_datetime}")
	
	# 4. Разделение на Train и Test
	df_train = df[df['datetime'] < split_datetime].copy()
	df_test = df[df['datetime'] >= split_datetime].copy()
	
	# Перемешивание внутри Train допустимо для некоторых моделей градиентного бустинга,
	# но в инфраструктурных задачах лучше сохранить порядок или группировку. Оставим сортировку.
	df_train = df_train.sort_values(by=['container_ip', 'datetime']).reset_index(drop=True)
	df_test = df_test.sort_values(by=['container_ip', 'datetime']).reset_index(drop=True)
	
	# 5. Сохранение артефактов
	train_path = output_dir / "train.csv"
	test_path = output_dir / "test.csv"
	
	df_train.to_csv(train_path, index=False)
	df_test.to_csv(test_path, index=False)
	
	logging.info(f"Разделение завершено.")
	logging.info(
		f"Train выборка: {df_train.shape} строк (период: {df_train['datetime'].min()} -> {df_train['datetime'].max()})")
	logging.info(
		f"Test выборка: {df_test.shape} строк (период: {df_test['datetime'].min()} -> {df_test['datetime'].max()})")


if __name__ == "__main__":
	split_data() 