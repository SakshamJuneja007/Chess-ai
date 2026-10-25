from tensorflow.keras.models import load_policy_network

MODEL_PATH = "D:\\chess-ai\\models\\resnet_model.keras"

def load_policy_network():
    model = load_policy_network()(MODEL_PATH)
    return model

if __name__ == "__main__":
    model = load_policy_network()
    print("Model loaded successfully!")
    model.summary()
 