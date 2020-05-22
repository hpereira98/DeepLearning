import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import model_from_yaml
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

# Utils para gravar modelos e pesos utilizá-los posteriormente
'''
Gravar um modelo num ficheiro utilizando o formato json. O nome do ficheiro deve ter a
extensão .json
'''
def save_model_json(model,fich):
    model_json = model.to_json()
    with open(fich, "w") as json_file:
        json_file.write(model_json)

'''
Gravar um modelo num ficheiro utilizando o formato yaml. O nome do ficheiro deve ter a
extensão .yaml
'''
def save_model_yaml(model,fich):
    model_yaml = model.to_yaml()
    with open(fich, "w") as yaml_file:
        yaml_file.write(model_yaml)

'''
Gravar os pesos de um modelo treinado num ficheiro utilizando o formato HDF5. O nome do
ficheiro deve ter a extensão .h5
'''
def save_weights_hdf5(model,fich):
    model.save_weights(fich)
    print("Saved model to disk")

'''
Ler um modelo de um ficheiro no formato json e criar o respetivo modelo em memória.
'''
def load_model_json(fich):
    json_file = open(fich, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    return loaded_model

'''
Ler um modelo de um ficheiro no formato yaml e criar o respetivo modelo em memória.
'''
def load_model_yaml(fich):
    yaml_file = open(fich, 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    return model_from_yaml(loaded_model_yaml)

'''
Ler os pesos um modelo treinado de um ficheiro no formato hdf5 para o respectivo
modelo.
'''
def load_weights_hdf5(model,fich):
    model.load_weights(fich)
    print("Loaded model from disk")