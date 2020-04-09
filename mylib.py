class mylib: 
    
    import string
    def dqr(data):
        
        import pandas as pd
        import numpy as np
    
        #%% Determinar el nombre de las variables
        columns = pd.DataFrame(list(data.columns.values), 
                               columns=['Nombres'],
                               index = list(data.columns.values))
        
        
        #%% Determinar el tipo de variable
        d_types = pd.DataFrame(data.dtypes,
                               columns=['Tipo'])
        
        #%% Determinar valores perdidos
        missing_values = pd.DataFrame(data.isnull().sum(axis=0),
                                      columns=['Valores perdidos'])
        
        #%% Determinar valores presentes
        present_values = pd.DataFrame(data.count(),
                                      columns=['Valores presentes'])
        
        #%% Determinar valores únicos
        unique_values = pd.DataFrame(data.nunique(),
                                      columns=['Valores unicos'])
        
        #%% Valores minimos y maximos de cada variable
        min_values = pd.DataFrame(columns=['Min'])
        max_values = pd.DataFrame(columns=['Max'])
        for col in list(data.columns.values):
            try:
                min_values.loc[col]=[data[col].min()]
                max_values.loc[col]=[data[col].max()]
            except: 
                pass
        
        #%% Juntar todas las tablas DATA QUALITY REPORT
        dqr = columns.join(d_types).join(missing_values).join(
                present_values).join(unique_values).join(min_values).join(max_values)
        return dqr

    #%% Funcion para retirar signos de puntuacion
    def remove_punctuation(x):
        try: 
            x = ''.join(ch for ch in x if ch not in string.punctuation)
        except:
            pass
        return x
    
    #tmp = string.punctuation + '`' y se pone, not in tmp
    #%% Remover digitos
    def remove_digits(x):
        try:
            x = ''.join(ch for ch in x if ch not in string.digits)
        except:
            pass
        return x
    
    #%% Remover espacios en blanco
    def remove_whitespaces(x):
        try:
            x = ''.join(x.split())
        except:
            pass
        return x
    
    #%% Remplazar texto
    def replace_text(x, to_replace, replacement):
        try:
            x = x.replace(to_replace, replacement)
        except:
            pass
        return x
    
    #%% Convertir a mayusculas
    def uppercase_text(x):
        try:
            x = x.upper()
        except:
            pass
        return x
    
    #%% Convertir a minusculas
    def lowercase_text(x):
        try:
            x = x.lower()
        except:
            pass
        return x
    
    #%% Dejar solo digitos
    def only_digits(x):
        try:
            x = ''.join(ch for ch in x if ch in string.digits)
        except:
            pass
        return x