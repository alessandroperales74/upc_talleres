#importar librerias
import streamlit as st
import pickle
import pandas as pd

#Extrar los archivos pickle
with open('modelo_decision_tree.pkl', 'rb') as archivo:
    d_tree = pickle.load(archivo)

with open('modelo_random_forest.pkl', 'rb') as archivo:
    r_forest = pickle.load(archivo)

with open('modelo_bagging.pkl', 'rb') as archivo:
    bagging = pickle.load(archivo)

with open('modelo_lightgbm.pkl', 'rb') as archivo:
    light_gbm = pickle.load(archivo)

with open('modelo_xgboost.pkl', 'rb') as archivo:
    xgb = pickle.load(archivo)

#funcion para clasificar las plantas 
def classify(num):
    if num == 0:
        return 'Sin riesgo de infarto'
    elif num == 1:
        return 'Riesgo de infarto'

def main():
    #titulo
    st.title('Modelamiento de Heart Disease')
    #titulo de sidebar
    st.sidebar.header('User Input Parameters')

    #funcion para poner los parametrso en el sidebar
    def user_input_parameters():
        edad = st.sidebar.slider('Edad', 15,99)
        colesterol = st.sidebar.slider('Nivel de Colesterol', 100,500)
        data = {'Age': edad,
                'Cholesterol': colesterol
                }
        features = pd.DataFrame(data, index=[0])
        return features

    df = user_input_parameters()

    #escoger el modelo preferido
    option = ['Decision Tree', 'Random Forest', 'Bagging Classifier', 'LightGBM','XGBoost']
    model = st.sidebar.selectbox('Qué modelo deseas usar?', option)

    st.subheader('Parámetros')
    st.subheader(model)
    st.write(df)

    if st.button('RUN'):
        if model == 'Decision Tree':
            st.success(classify(d_tree.predict(df)))
        elif model == 'Random Forest':
            st.success(classify(r_forest.predict(df)))
        elif model == 'Bagging Classifier':
            st.success(classify(bagging.predict(df)))
        elif model == 'LightGBM':
            st.success(classify(light_gbm.predict(df)))
        else:
            st.success(classify(xgb.predict(df)))

if __name__ == '__main__':
    main()
    
