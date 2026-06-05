# src/stages/preprocess.py
import os
import sys
import logging
from pathlib import Path
import yaml
import pandas as pd
import numpy as np

# Настройка логирования в stdout для продакшен-мониторинга
logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s [%(levelname)s] %(message)s",
	handlers=[logging.StreamHandler(sys.stdout)]
)


def load_params(params_path: str = "params.yaml") -> dict:
	"""Загрузка конфигурационных параметров эксперимента."""
	try:
		with open(params_path, "r") as f:
			return yaml.safe_load(f)
	except Exception as e:
		logging.error(f"Ошибка при загрузке {params_path}: {e}")
		raise


def preprocess_data():
	logging.info("Старт этапа предобработки данных...")
	
	# 1. Загрузка параметров
	config = load_params()
	prep_config = config["preprocess"]
	
	raw_gpu_path = Path(prep_config["raw_gpu_path"])
	raw_qps_path = Path(prep_config["raw_qps_path"])
	output_dir = Path(prep_config["output_dir"])
	window = prep_config["agg_window"]
	
	output_dir.mkdir(parents=True, exist_ok=True)
	
	# 2. Чтение сырых данных
	logging.info(f"Чтение сырых файлов: {raw_gpu_path} и {raw_qps_path}")
	df_gpu = pd.read_csv(raw_gpu_path)
	df_qps = pd.read_csv(raw_qps_path)
	
	# 3. Приведение временных шкал
	logging.info("Преобразование timestamp_anon в datetime...")
	df_gpu['datetime'] = pd.to_datetime(df_gpu['timestamp_anon'], unit='s')
	df_qps['datetime'] = pd.to_datetime(df_qps['timestamp_anon'], unit='s')
	
	# 4. Выделение глобального признака из API Requests без привязки к IP
	logging.info("Расчет глобального признака Cluster API QPS...")
	df_api_global = df_qps[df_qps['container_ip'].isna() | (df_qps['request_type'] == 'API Requests')]
	df_api_agg = (
		df_api_global.set_index('datetime')
		.resample(window)['value']
		.sum()
		.rename('global_api_qps')
		.to_frame()
	)
	
	# 5. Обработка контейнерных данных QPS (только Generative)
	logging.info("Агрегация контейнерных QPS (Generative)...")
	df_qps_nodes = df_qps[df_qps['container_ip'].notna() & (df_qps['request_type'] == 'Generative Requests')]
	df_qps_agg = (
		df_qps_nodes.set_index('datetime')
		.groupby('container_ip')
		.resample(window)['value']
		.sum()
		.rename('node_generative_qps')
		.reset_index()
	)
	
	# 6. Агрегация данных утилизации GPU
	logging.info("Агрегация утилизации GPU по контейнерам...")
	df_gpu_agg = (
		df_gpu.set_index('datetime')
		.groupby('container_ip')
		.resample(window)['value']
		.mean()
		.rename('gpu_util')
		.reset_index()
	)
	
	# 7. Объединение по сетке (Внутреннее соединение по временным окнам и контейнерам)
	logging.info("Объединение признаков и целевой переменной...")
	# Сначала объединяем локальные метрики контейнеров
	df_merged = pd.merge(
		df_gpu_agg,
		df_qps_agg,
		on=['datetime', 'container_ip'],
		how='inner'
	)
	
	# Затем добавляем глобальные системные метрики
	df_final = pd.merge(
		df_merged.set_index('datetime'),
		df_api_agg,
		left_index=True,
		right_index=True,
		how='left'
	).reset_index()
	
	# Заполнение возможных пропусков, возникших при несовпадении глобальных окон
	df_final['global_api_qps'] = df_final['global_api_qps'].fillna(0)
	
	# Сортировка для корректного разбиения по времени в будущем
	df_final = df_final.sort_values(by=['container_ip', 'datetime']).reset_index(drop=True)
	
	# 8. Сохранение обработанных данных
	output_file = output_dir / "features_target.csv"
	df_final.to_csv(output_file, index=False)
	logging.info(f"Предобработка успешно завершена. Файл сохранен: {output_file}")
	logging.info(f"Итоговый размер матрицы признаков: {df_final.shape}")


if __name__ == "__main__":
	preprocess_data()