import pandas as pd
from itertools import combinations
import time
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

# --- Carga dataset
data = pd.read_csv('data_secretariado.csv')

# Manejo de fechas para sacar edad y mes
data['FECHA_NACIMIENTO'] = pd.to_datetime(data['FECHA_NACIMIENTO'], errors='coerce')
data['FECHA_DESAPARICION'] = pd.to_datetime(data['FECHA_DESAPARICION'], errors='coerce')

data['edad_calculada'] = 2026 - data['FECHA_NACIMIENTO'].dt.year
data['mes_desparicion'] = data['FECHA_DESAPARICION'].dt.month_name()

def categorizar_por_edad(e):
    if pd.isna(e) or e < 0: 
        return 'EDAD_DESCONOCIDA'
    if e <= 11: return 'NIÑEZ'
    if e <= 17: return '12-17'
    if e <= 59: return 'ADULTO'
    return 'ADULTO_MAYOR'

data['GRUPO_EDAD'] = data['edad_calculada'].apply(categorizar_por_edad)
data = data.fillna('SIN_DATO')

# Columnas para las reglas de asociacion
cols = ['SEXO', 'GRUPO_EDAD', 'ENTIDAD', 'mes_desparicion', 'ESTATUS_VICTIMA']

lista_transacciones = []
for _, fila in data[cols].iterrows():
    temp = [f"{col}_{val}" for col, val in fila.items()]
    lista_transacciones.append(temp)

#=======================APRIORI==========================

def apriori_manual(dataset, min_sup):
    n = len(dataset)
    
    # Paso 1: Sacar items frecuentes de tamaño 1
    counts = {}
    for t in dataset:
        for item in t:
            it = frozenset([item])
            counts[it] = counts.get(it, 0) + 1
            
    frecuentes_act = {it: c/n for it, c in counts.items() if c/n >= min_sup}
    todos_frecuentes = frecuentes_act.copy()
    
    k = 2
    while frecuentes_act:
        # FASE DE UNION: Generar candidatos Ck
        items = list(frecuentes_act.keys())
        candidatos = []
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                union = items[i] | items[j]
                if len(union) == k and union not in candidatos:
                    # FASE DE PODA
                    subs = [union - frozenset([x]) for x in union]
                    if all(s in frecuentes_act for s in subs):
                        candidatos.append(union)
        
        # Conteo de soporte para candidatos Ck
        counts_k = {c: 0 for c in candidatos}
        for t in dataset:
            t_set = set(t)
            for c in candidatos:
                if c.issubset(t_set):
                    counts_k[c] += 1
        
        # Filtrar candidatos que si pasaron el soporte (Lk)
        frecuentes_act = {c: val/n for c, val in counts_k.items() if val/n >= min_sup}
        todos_frecuentes.update(frecuentes_act)
        k += 1
        
    return todos_frecuentes

def sacando_reglas(dict_frecuentes, min_conf):
    res = []
    for itemset, sup in dict_frecuentes.items():
        if len(itemset) > 1:
            for i in range(1, len(itemset)):
                for ant in combinations(itemset, i):
                    ant = frozenset(ant)
                    cons = itemset - ant
                    
                    conf = sup / dict_frecuentes[ant]
                    if conf >= min_conf:
                        lift = conf / dict_frecuentes[cons]
                        res.append({
                            'antecedente': list(ant),
                            'consecuente': list(cons),
                            'Soporte': sup,
                            'Confianza': conf,
                            'Lift': lift
                        })
    return pd.DataFrame(res)

# Prueba
frecuentes = apriori_manual(lista_transacciones, 0.05)
df_reglas = sacando_reglas(frecuentes, 0.4)

print("--- REGLAS ENCONTRADAS Apriori---")
print(df_reglas.sort_values('Lift', ascending=False).head(10))

#===============COMPARACION===================
# Preparación para librerías
te = TransactionEncoder()
te_ary = te.fit(lista_transacciones).transform(lista_transacciones)
df_binario = pd.DataFrame(te_ary, columns=te.columns_)

# # 1. Ejecución Apriori Manual
inicio = time.time()
frecuentes_manual = apriori_manual(lista_transacciones, 0.05)
reglas_manual = sacando_reglas(frecuentes_manual, 0.4)
tiempo_manual = time.time() - inicio


# 2. Ejecución Apriori de Librería
inicio = time.time()
frecuentes_lib = apriori(df_binario, min_support=0.05, use_colnames=True)
reglas_lib = association_rules(frecuentes_lib, metric="confidence", min_threshold=0.4)
tiempo_lib = time.time() - inicio
print("\n ------REGLAS ENCONTRADAS Apriori (Libreria)---")
# Mostrar algunas reglas para comparar visualmente
print(reglas_lib[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(5))

# 3. Ejecución FP-Growth
inicio = time.time()
frecuentes_fp = fpgrowth(df_binario, min_support=0.05, use_colnames=True)
reglas_fp = association_rules(frecuentes_fp, metric="confidence", min_threshold=0.4)
tiempo_fp = time.time() - inicio
print("\n -----REGLAS ENCONTRADAS FP-Growth (Libreria)---")
# Mostrar algunas reglas para comparar visualmente
print(reglas_fp[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(5))

#Comparacion de tiempo y cantidad de reglas
print(f"\n1. Tiempo Apriori Manual: {tiempo_manual:.4f} segundos")
print(f"Reglas generadas (Manual): {len(reglas_manual)}")
print(f"\n2. Tiempo Apriori Librería: {tiempo_lib:.4f} segundos")
print(f"Reglas generadas (Librería): {len(reglas_lib)}")
print(f"\n3. Tiempo FP-Growth: {tiempo_fp:.4f} segundos")
print(f"Reglas generadas (FP-Growth): {len(reglas_fp)}")
