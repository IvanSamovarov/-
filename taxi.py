# импорт библиотек
import pandas as pd
import numpy as np
import streamlit as st
import joblib
from joblib import load
from sklearn.preprocessing import OneHotEncoder

# загрузка модели
model = joblib.load("taxi_fare_model.pkl")
data = pd.read_csv('taxi_trip_pricing.csv')

for col in data.columns:
    if data[col].isnull().any():
        if data[col].dtype in ['int64', 'float64']:
          median_value = data[col].median() # Замена на медиану для численных признаков
          data[col] = data[col].fillna(median_value)
        else:
            mode_value = data[col].mode()[0]  # Замена на моду для категориальных признаков
            data[col] = data[col].fillna(mode_value)

# Категориальные признаки
categorical_features = ['Time_of_Day', 'Day_of_Week', 'Traffic_Conditions', 'Weather']
numerical_features = ['Trip_Distance_km', 'Passenger_Count', 'Base_Fare', 'Per_Km_Rate', 'Per_Minute_Rate', 'Trip_Duration_Minutes']
# Создаем энкодер и обучаем его на всех данных, чтобы не было проблем с новыми значениями
encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
X = data.drop('Trip_Price', axis=1)
encoder.fit(X[categorical_features])
feature_names = encoder.get_feature_names_out(categorical_features)


all_columns = [
    'Trip_Distance_km',
    'Time_of_Day',
    'Day_of_Week',
    'Passenger_Count',
    'Traffic_Conditions',
    'Weather',
    'Base_Fare',
    'Per_Km_Rate',
    'Per_Minute_Rate',
    'Trip_Duration_Minutes'
    ]

# Указание контента сайта
st.title("Прогноз цен на такси")

trip_distance_km = st.number_input(
    "Дистанция поездки (в километрах)",
    min_value=1,
    max_value=146,
    value=30,
    step=1
)
passenger_count = st.number_input(
    "Количество пассажиров",
    min_value=1,
    max_value=4,
    value=1,
    step=1
)
base_fare = st.number_input(
    "Базовая стоимость проезда",
    min_value=2,
    max_value=5,
    value=3,
    step=1
)
per_km_rate = st.number_input(
    "Стоимость поездки за километр",
    min_value=0.5,
    max_value=2.0,
    value=1.0,
    step=0.1
)
per_minute_rate = st.number_input(
    "Стоимость поездки за минуту",
    min_value=0.1,
    max_value=0.5,
    value=0.2,
    step=0.1
)
trip_duration_minutes = st.number_input(
    "Общее время поездки",
    min_value=5,
    max_value=120,
    value=20,
    step=1
)
time_of_day = st.selectbox('Время суток', options=data['Time_of_Day'].unique())
day_of_week = st.selectbox('День недели', options=data['Day_of_Week'].unique())
traffic_conditions = st.selectbox('Условия движения', options=data['Traffic_Conditions'].unique())
weather = st.selectbox('Погода', options=data['Weather'].unique())

# Преобразование данных и прогноз
input_data = pd.DataFrame({
    'Trip_Distance_km': [trip_distance_km],
    'Passenger_Count': [passenger_count],
    'Base_Fare': [base_fare],
    'Per_Km_Rate': [per_km_rate],
    'Per_Minute_Rate': [per_minute_rate],
    'Trip_Duration_Minutes': [trip_duration_minutes],
    'Time_of_Day': [time_of_day],
    'Day_of_Week': [day_of_week],
    'Traffic_Conditions': [traffic_conditions],
    'Weather': [weather]
})
# Отделяем числовые и категориальные признаки
input_data_num = input_data[numerical_features]
input_data_cat = input_data[categorical_features]


# Кодируем категориальные признаки
input_data_encoded = encoder.transform(input_data_cat)
input_data_encoded = pd.DataFrame(input_data_encoded, columns = feature_names)


# Объединяем числовые и закодированные категориальные признаки
input_data_processed = pd.concat([input_data_num, input_data_encoded], axis = 1)

if st.button('Предсказать стоимость'):
    prediction = model.predict(input_data_processed)[0]
    st.success(f'Предсказанная стоимость поездки: {prediction:.2f}')