from tensorflow.keras.models import load_model

model = load_model(r"best_model.h5")

print(model.summary())

model_params = model.get_weights()

    
