from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model

model = Sequential([
    Dense(10, input_shape=(7,))
])

model.save("test_model.h5")

loaded = load_model("test_model.h5")

print("MODEL WORKING")