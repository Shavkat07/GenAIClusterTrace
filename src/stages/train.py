# src/stages/train.py
import sys
import logging
import json
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Здесь logging.INFO используется верно — как целочисленная константа уровня
logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s [%(levelname)s] %(message)s",
	handlers=[logging.StreamHandler(sys.stdout)]
)


def load_params(params_path: str = "params.yaml") -> dict:
	with open(params_path, "r") as f:
		return yaml.safe_load(f)


def train_model():
	logging.info("Старт этапа обучения модели...")
	
	# 1. Загрузка конфигурации
	config = load_params()
	train_config = config["train"]
	base_config = config["base"]
	
	train_path = Path(train_config["train_path"])
	test_path = Path(train_config["test_path"])
	model_dir = Path(train_config["model_dir"])
	reports_dir = Path(train_config["reports_dir"])
	
	model_dir.mkdir(parents=True, exist_ok=True)
	reports_dir.mkdir(parents=True, exist_ok=True)
	
	# 2. Чтение выборок
	df_train = pd.read_csv(train_path)
	df_test = pd.read_csv(test_path)
	
	features = train_config["features"]
	target = train_config["target"]
	
	X_train = df_train[features]
	y_train = df_train[target]
	X_test = df_test[features]
	y_test = df_test[target]
	
	logging.info(f"Признаки для обучения: {features}")
	logging.info(f"Размерность матриц: X_train {X_train.shape}, X_test {X_test.shape}")
	
	# 3. Инициализация и обучение Baseline-модели
	model = RandomForestRegressor(
		n_estimators=train_config["n_estimators"],
		max_depth=train_config["max_depth"],
		random_state=base_config["random_state"],
		n_jobs=-1
	)
	
	logging.info("Обучение RandomForestRegressor...")
	model.fit(X_train, y_train)
	
	# 4. Валидация модели
	logging.info("Валидация модели на тестовой выборке (из будущего)...")
	predictions = model.predict(X_test)
	
	mae = mean_absolute_error(y_test, predictions)
	mse = mean_squared_error(y_test, predictions)
	rmse = np.sqrt(mse)
	r2 = r2_score(y_test, predictions)
	
	metrics = {
		"mae": float(mae),
		"rmse": float(rmse),
		"r2": float(r2)
	}
	
	logging.info(f"Результаты Baseline: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
	
	# 5. Сохранение артефактов (Модель и Метрики)
	model_path = model_dir / train_config["model_name"]
	metrics_path = reports_dir / train_config["metrics_name"]
	
	joblib.dump(model, model_path)
	logging.info(f"Модель сохранена в: {model_path}")
	
	with open(metrics_path, "w") as f:
		json.dump(metrics, f, indent=4)
	logging.info(f"Метрики сохранены в: {metrics_path}")


if __name__ == "__main__":
	train_model()