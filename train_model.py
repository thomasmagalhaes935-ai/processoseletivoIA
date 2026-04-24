import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time

def carregar_dados():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalização
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Ajuste de dimensão (28,28) -> (28,28,1)
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    return (x_train, y_train), (x_test, y_test)

def construir_modelo():
    model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),

        layers.Conv2D(24, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(48, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),

        layers.Flatten(),

        layers.Dense(96, activation='relu'),
        layers.Dropout(0.2),

        layers.Dense(32, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def main():
    (x_train, y_train), (x_test, y_test) = carregar_dados()

    model = construir_modelo()

    print("\nIniciando treinamento da rede neural...\n")

    inicio = time.time()

    model.fit(
        x_train,
        y_train,
        epochs=5,
        validation_data=(x_test, y_test),
        verbose=2
    )

    fim = time.time()

    loss, acc = model.evaluate(x_test, y_test, verbose=0)

    print("\n===== RESULTADOS =====")
    print(f"Loss: {loss:.4f}")
    print(f"Acurácia: {acc:.4f}")
    print(f"Tempo de treino: {fim - inicio:.2f} segundos")

    model.save("model.h5")
    print("Modelo salvo como model.h5")

if __name__ == "__main__":
    main()