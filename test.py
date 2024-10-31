import pandas as pd
from keras import Sequential
from keras.src.layers import LSTM, Dense
from keras.src.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import FinanceDataReader as fdr
from datetime import datetime, timedelta
from keras._tf_keras.keras.models import load_model, save_model

asset_name = 379810

model_path = f'./save_models/lstm_{asset_name}.h5'
model = load_model(model_path)

print(model)