import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Sample code for Anomaly Detection using LSTM

# Function to prepare the dataset
def prepare_data(data_frame):
    # Scaling the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_frame)

    X, y = [], []
    time_step = 30  # Use 30 timestamps
    
    for i in range(len(data_scaled) - time_step):
        X.append(data_scaled[i:i + time_step])
        y.append(data_scaled[i + time_step])
        
    return np.array(X), np.array(y)

# Create Sequential LSTM Model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=64))
    model.add(Dropout(0.2))
    model.add(Dense(units=input_shape[1]))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def main():
    # Load example health data
    # In practice, this will be replaced by real data from device APIs
    data = pd.read_csv('data/sample_health_data.csv')  # Replace with the actual data source

    X, y = prepare_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = create_lstm_model(input_shape)
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

    # Evaluate the model
    loss = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss}')

if __name__ == "__main__":
    main()
