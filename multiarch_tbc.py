import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Activation
from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import time
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, roc_curve, auc
import joblib
import datetime
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Variables globales
img_dir = "tbc_images"
img_size = 350
#_archs = ["Arch1", "Arch2", "Arch3", "Arch4", "Arch5", "Arch6"]
_archs = ["Arch1", "Arch2"]
#arch 1: 32, 32
#arch 2: 32, 32, 32
#arch 3: 48, 48
#arch 4: 48, 48, 48
#arch 5: 64, 64
#arch 6: 64, 64, 64
_epochs = 50

# Función para cargar imágenes
def load_img_data(path):
    image_files = glob.glob(os.path.join(path, "topes/*.jpg")) + \
                  glob.glob(os.path.join(path, "baches/*.jpg")) + \
                  glob.glob(os.path.join(path, "libre/*.jpg"))
    X, y = [], []
    for image_file in image_files:
        if "topes" in image_file:
            label = 0
        elif "baches" in image_file:
            label = 1
        else:
            label = 2 # sin topes y sin baches
        img_arr = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img_arr, (img_size, img_size))
        X.append(img_resized)
        y.append(label)
    return X, y

# Definición de arquitecturas de modelos
def create_model(arch_type):
    model = Sequential()
    if arch_type in ["Arch1", "Arch2"]:
        model.add(Conv2D(32, (3, 3), input_shape=(img_size, img_size, 1), activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPool2D(pool_size=(2, 2)))
        if arch_type == "Arch2":
            model.add(Conv2D(32, (3, 3), activation="relu"))
            model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(32, activation="relu"))
    elif arch_type in ["Arch3", "Arch4"]:
        model.add(Conv2D(48, (3, 3), input_shape=(img_size, img_size, 1), activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Conv2D(48, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPool2D(pool_size=(2, 2)))
        if arch_type == "Arch4":
            model.add(Conv2D(48, (3, 3), activation="relu"))
            model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(48, activation="relu"))
    elif arch_type in ["Arch5", "Arch6"]:
        model.add(Conv2D(64, (3, 3), input_shape=(img_size, img_size, 1), activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPool2D(pool_size=(2, 2)))
        if arch_type == "Arch6":
            model.add(Conv2D(64, (3, 3), activation="relu"))
            model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(64, activation="relu"))
    #model.add(Flatten())
    #model.add(Dense(32, activation="relu"))
    model.add(Dense(3, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

# Función para plotear el promedio de precisión por arquitectura
def plot_architecture_average(fold_avg_acc_by_arch, fold_avg_val_acc_by_arch):
    epochs_range = range(1, _epochs + 1)
    plt.figure(figsize=(12, 6))
    
    for arch_idx, arch_name in enumerate(_archs):
        plt.plot(epochs_range, fold_avg_acc_by_arch[arch_idx], label=f"{arch_name} - Train Avg Accuracy")
        plt.plot(epochs_range, fold_avg_val_acc_by_arch[arch_idx], linestyle="--", label=f"{arch_name} - Val Avg Accuracy")

    plt.title("Average Accuracy per Architecture")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("average_performance_architectures.png")
    plt.close()

def save_results_to_file(arch_name, metrics, duration, model_size_mb):
    with open("resultados.txt", "a") as f:
        f.write(f"Arquitectura: {arch_name}\n")
        f.write(f"Duración del entrenamiento: {duration:.2f} segundos\n")
        f.write(f"Tamaño del modelo: {model_size_mb:.2f} MB\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1']:.4f}\n")
        f.write("-" * 50 + "\n")

# Cargar los datos
X, y = load_img_data(img_dir)
X = np.array(X).reshape(-1, img_size, img_size, 1) / 255
y = np.array(y)
y_cat = to_categorical(y, num_classes=3)
# Validación cruzada y entrenamiento de modelos
kf = StratifiedKFold(n_splits=5)

# Almacenar precisión de entrenamiento y validación promedio para cada arquitectura en cada época
fold_avg_acc_by_arch = [np.zeros(_epochs) for _ in _archs]
fold_avg_val_acc_by_arch = [np.zeros(_epochs) for _ in _archs]
arch_metrics = {arch: {'accuracy': [], 'precision': [], 'recall': [], 'f1': []} for arch in _archs}
arch_conf_matrices = {arch: np.zeros((3, 3), dtype=int) for arch in _archs}
#arch_training_times = {arch: []}
arch_training_times = {arch: [] for arch in _archs}

model_sizes = {}

count_kfold_r = 0
count_arch_r = 1
for train_index, test_index in kf.split(X, y):
    for arch_idx, arch in enumerate(_archs):
        #cur_run = (count_arch_r * 5) + count_kfold_r
        cur_run = count_arch_r
        print("Counting: %s of %s" %(cur_run, 10))
        X_train, X_test = X[train_index], X[test_index]
        #y_train, y_test = y_cat[train_index], y_cat[test_index]
        y_train = to_categorical(y[train_index], num_classes=3)
        y_test = to_categorical(y[test_index], num_classes=3)

        model = create_model(arch)
        start_time=time.time()
        history = model.fit(X_train, y_train, batch_size=32, epochs=_epochs, validation_split=0.2, verbose=0)
        end_time=time.time()
        # Sumar las métricas de precisión y validación para luego promediar
        fold_avg_acc_by_arch[arch_idx] += np.array(history.history['accuracy'])
        fold_avg_val_acc_by_arch[arch_idx] += np.array(history.history['val_accuracy'])
        # Evaluación
        y_pred = model.predict(X_test)
        y_true = np.argmax(y_test, axis=1)
        y_pred_classes = np.argmax(y_pred, axis=1)

        accuracy = accuracy_score(y_true, y_pred_classes)
        precision = precision_score(y_true, y_pred_classes, average='macro')
        recall = recall_score(y_true, y_pred_classes, average='macro')
        f1 = f1_score(y_true, y_pred_classes, average='macro')

        # Tiempo
        duration = end_time - start_time

        # Tamaño del modelo
        model.save(f"model_{arch}.h5")
        model_size_mb = os.path.getsize(f"model_{arch}.h5") / (1024 * 1024)

        # Guardar resultados
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        #save_results_to_file(arch, metrics, duration, model_size_mb)
        # Matriz de confusión
        #cm = confusion_matrix(y_true, y_pred_classes)
        #disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2])

        # Mostrar y guardar la imagen
        #disp.plot(cmap='Blues', values_format='d')
        #plt.title(f"Matriz de Confusión - {arch}")
        #plt.savefig(f"confusion_matrix_{arch}.png")
        #plt.close()

        # Calcular métricas y matriz de confusión
        cm = confusion_matrix(y_true, y_pred_classes)
        arch_conf_matrices[arch] += cm  # sumas matriz por fold

        arch_metrics[arch]['accuracy'].append(accuracy)
        arch_metrics[arch]['precision'].append(precision)
        arch_metrics[arch]['recall'].append(recall)
        arch_metrics[arch]['f1'].append(f1)
        #arch_metrics[arch]['roc_auc'].append(roc_auc)

        arch_training_times[arch].append(duration)
        model_sizes[arch] = model_size_mb  # se sobrescribe, pero el tamaño no cambia por fold
        count_arch_r += 1
    count_kfold_r += 1

for arch in _archs:
    avg_metrics = {
        metric: np.mean(values) for metric, values in arch_metrics[arch].items()
    }
    avg_duration = np.mean(arch_training_times[arch])
    size = model_sizes[arch]
    
    save_results_to_file(arch, avg_metrics, avg_duration, size)

    # Matriz de confusión promedio/acumulada
    disp = ConfusionMatrixDisplay(confusion_matrix=arch_conf_matrices[arch],
                                  display_labels=[0, 1, 2])
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f"Confusion Matrix (Average) - {arch}")
    plt.savefig(f"confusion_matrix_{arch}.png")
    plt.close()

# Promediar las métricas de precisión para cada arquitectura
fold_avg_acc_by_arch = [acc / kf.get_n_splits() for acc in fold_avg_acc_by_arch]
fold_avg_val_acc_by_arch = [val_acc / kf.get_n_splits() for val_acc in fold_avg_val_acc_by_arch]

# Graficar la precisión promedio de cada arquitectura
plot_architecture_average(fold_avg_acc_by_arch, fold_avg_val_acc_by_arch)
