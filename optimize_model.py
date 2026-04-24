import tensorflow as tf
import os

def main():
    print("Carregando modelo treinado (Keras)...\n")

    model = tf.keras.models.load_model("model.h5", compile=False)

    print("Convertendo modelo para TensorFlow Lite...\n")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Otimização 
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    with open("model.tflite", "wb") as f:
        f.write(tflite_model)

    print("Modelo convertido e salvo como model.tflite")

    # Comparação de tamanho
    original = os.path.getsize("model.h5") / 1024
    otimizado = os.path.getsize("model.tflite") / 1024

    print("\nCOMPARAÇÃO:")
    print(f"Tamanho original: {original:.2f} KB")
    print(f"Tamanho otimizado: {otimizado:.2f} KB")

if __name__ == "__main__":
    main()