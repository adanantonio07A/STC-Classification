import cv2
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import glob
from sklearn.metrics import confusion_matrix
import time

#guide of archs:
img_dir = "tb_images"
img_size=350
_archs = ["model_Arch", "model_Arch"]
#_archs = ["32_32", "32_32_32", "48_48", "48_48_48", "64_64", "64_64_64"]

ARCH = "32_32"
in_dir = ""
img_size=350
classes = ['baches', 'topes', 'libre']
# Funcion para cargar y preprocesar imagenes
def load_and_preprocess_image(filepath, img_size):
    img_arr = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # Leer en escala de grises
    img_arr = cv2.resize(img_arr, (img_size, img_size))  # Redimensionar
    img_arr = img_arr / 255.0  # Escalar los valores de pixeles entre 0 y 1
    img_arr = np.expand_dims(img_arr, axis=2)  # Agregar la dimension para canales (grayscale)
    return img_arr

def evaluate_multiple_elements_from_directory(directory_path, model, output_csv="predictions.csv", image_size=(350, 350), bins=256):
        """
        Evalúa todas las imágenes en un directorio y sus subdirectorios.
        Args:
            directory_path: Ruta del directorio que contiene las imágenes.
            model: El modelo previamente entrenado.
            output_csv: Nombre del archivo CSV donde se guardarán los resultados.
            image_size: Tamaño de la imagen de entrada para la CNN.
        """
        predictions = []  # Lista para almacenar los nombres de los archivos y sus predicciones
        acc_times = []
        # Recorre todas las subcarpetas y archivos de imágenes
        for root, _, files in os.walk(directory_path):
            for filename in files:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):  # Filtrar solo imágenes
                    image_path = os.path.join(root, filename)
                    print(f"Evaluando la imagen: {image_path}")
                    
                    # 1. Cargar la imagen
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"Error al cargar la imagen: {image_path}")
                        predictions.append([filename, "Error"])
                        continue
                    
                    image_ = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    image_resized = cv2.resize(image_,  (img_size, img_size)) / 255.0  # Normalización
                    image_resized = np.expand_dims(image_resized, axis=0)  # Expandir dimensiones para entrada de la CNN
                    
                    # 2. Predecir con el modelo
                    start_time = time.time()  # Iniciar medición
                    prediction = model.predict(image_resized)
                    end_time = time.time()  # Finalizar medición
                    #print(prediction)
                    #print(type(prediction[0][0]))
                    # Calcular el tiempo total y promedio por imagen
                    cur_total_time = end_time - start_time
                    #print(cur_total_time)
                    acc_times.append(cur_total_time)
                    #cur_pred = "baches" if prediction[0][0] > 0.5 else "topes" 
                    #predicted_class = classes[np.argmax(prediction[0][0])]  # Obtener la clase con mayor probabilidad
                    #predictions.append([filename, cur_pred])
                    predicted_class = classes[np.argmax(prediction[0])]
                    predictions.append([filename, predicted_class])
                    #print(cur_pred)
        # Guardar los resultados en un archivo CSV
        with open('tiempos_modelos.txt', 'a') as file:
            avg = sum(acc_times) / len(acc_times)
            file.write(f"Modelo {model_path}: Tiempo promedio por imagen: {avg:.6f} segundos\n")
            #print(f"Modelo {model_path}: Tiempo promedio por imagen: {avg:.6f} segundos")
            #print(f"Max: {max(acc_times)}")
        with open(output_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Archivo", "Clase Predicha"])  # Encabezados del archivo CSV
            writer.writerows(predictions)
        return predictions

# Cargar todas las imagenes del directorio
def load_images_from_directory(directory, img_size):
    images = []
    labels = []
    for label in ['baches', 'topes']:
        img_paths = glob.glob(os.path.join(directory, label, '*.jpg'))
        for img_path in img_paths:
            img_arr = load_and_preprocess_image(img_path, img_size)
            images.append(img_arr)
            labels.append(0 if label == 'baches' else 1)  # Etiquetado: 0 para baches, 1 para topes
    return np.array(images), np.array(labels)

# Cargar imagenes y etiquetas
X_test, y_test = load_images_from_directory('tb_images', img_size)

# Cambiar la forma de los datos para que se ajusten a las dimensiones de la red neuronal
X_test = X_test.reshape(-1, img_size, img_size, 1)

for i in range(0, 1):
  cur_in_dir = ""
  for j in range(0,2):
    file2save_data = _archs[i] +"_" + str(j+1)
    model_path = cur_in_dir + _archs[i] + str(j+1) + ".h5"
    print(model_path)
    #exit()
    
    model = load_model(model_path)
    directory_path = "tbc_images/baches"  # Cambia esta ruta por la carpeta que contiene las imágenes
    output_csv = "pred_baches" + str(i) + "_" + str(j) + ".csv"
    
    # Evaluar todas las imágenes del directorio de la clase 1
    evaluate_multiple_elements_from_directory(directory_path, model, output_csv)
    #Para clase 2
    # Directorio donde se encuentran las imágenes
    directory_path = "tbc_images/topes"  # Cambia esta ruta por la carpeta que contiene las imágenes
    
    # Nombre del archivo CSV de salida, Clase2, malignant
    output_csv = "pred_topes" + str(i) + "_" + str(j) + ".csv"
    
    # Evaluar todas las imágenes del directorio de la clase 2
    evaluate_multiple_elements_from_directory(directory_path, model, output_csv)
    #evaluate_multiple_elements_from_directory()
    directory_path = "tbc_images/libre"
    output_csv = "pred_libre" + str(i) + "_" + str(j) + ".csv"
    evaluate_multiple_elements_from_directory(directory_path, model, output_csv)

    """
    predictions = None
    with open('tiempos_modelos.txt', 'a') as file:
      start_time = time.time()  # Iniciar medición
      predictions = model.predict(X_test)
      end_time = time.time()  # Finalizar medición
      # Calcular el tiempo total y promedio por imagen
      total_time = end_time - start_time
      avg_time_per_image = total_time / len(X_test)
      
      # Guardar resultados en el archivo
      file.write(f"Modelo {model_path}: Tiempo total: {total_time:.4f} segundos, "
                  f"Tiempo promedio por imagen: {avg_time_per_image:.6f} segundos\n")
      print(f"Modelo {model_path}: Tiempo total: {total_time:.4f} segundos, "
            f"Tiempo promedio por imagen: {avg_time_per_image:.6f} segundos")

    # Mostrar resultados
    #for k, prediction in enumerate(predictions):
    #    print(f"Imagen {k+1} - Real: {'baches' if y_test[k] == 0 else 'topes'}, Prediccion: {prediction[0]:.4f}")
        #plt.imshow(X_test[i].reshape(img_size, img_size), cmap='gray')
        #plt.title(f"Real: {'Parasitized' if y_test[i] == 0 else 'Uninfected'}, Prediccion: {prediction[0]:.4f}")
        #plt.show()

    # Convertir las predicciones a etiquetas binarias (1 si la prediccion es >= 0.5, 0 si es < 0.5)
    predicted_labels = [0 if pred >= 0.5 else 1 for pred in predictions]

    # Obtener la matriz de confusion para calcular TP, TN, FP, FN
    tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels).ravel()

    print(f"Verdaderos Positivos (TP): {tp}")
    print(f"Verdaderos Negativos (TN): {tn}")
    print(f"Falsos Positivos (FP): {fp}")
    print(f"Falsos Negativos (FN): {fn}")

    # Evaluar el desempenio en todas las imagenes
    #loss, accuracy = model.evaluate(X_test, y_test)
    #print(f"Perdida: {loss}, Precision: {accuracy}")
    # Evaluar el desempeño en todas las imagenes
    #loss, accuracy = model.evaluate(X_test, y_test)
    #print(f"Perdida: {loss}, Precision: {accuracy}")
    """
# --------- USO DEL SCRIPT ---------
exit()
if __name__ == "__main__":
        # Para clase 1
        # SOB_B_F-14-9133-400
        # SOB_B_F-14-14134-400
        # Directorio donde se encuentran las imágenes
        
        directory_path = "tb_images/baches"  # Cambia esta ruta por la carpeta que contiene las imágenes
        output_csv = "time_results/pred_baches" + str(i) + ".csv"
        evaluate_multiple_elements_from_directory(directory_path, combined_model, output_csv)
        
        # SOB_M_DC-14-2523-400-004
        # SOB_M_DC-14-2773-400-035
        #Para clase 2
        # Directorio donde se encuentran las imágenes
        directory_path = "tb_images/topes"  # Cambia esta ruta por la carpeta que contiene las imágenes
        
        # Nombre del archivo CSV de salida, Clase2, malignant
        output_csv = "time_results/pred_topes" + str(i) + ".csv"
        
        # Evaluar todas las imágenes del directorio de la clase 2
        evaluate_multiple_elements_from_directory(directory_path, combined_model, output_csv)
