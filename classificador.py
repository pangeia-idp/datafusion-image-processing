import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. CARREGAMENTO DOS DADOS
print("Carregando arquivos...")
df_labels = pd.read_csv('classificacao_final_corrigida.csv')
df_features = pd.read_csv('data_contest.csv')

# 2. UNIÃO DAS BASES (MERGE)
df_final = pd.merge(df_features, df_labels[['stac_id', 'super_classe', 'stac_browser_link']], on='stac_id')

# 3. DIVISÃO DOS DATASETS (70% Treino, 20% Validação, 10% Teste)
print("Dividindo os dados e gerando ficheiros CSV...")

# Primeiro Split: Separa 10% para o Teste Final
df_temp, df_teste = train_test_split(
    df_final, test_size=0.10, random_state=42, stratify=df_final['super_classe']
)

# Segundo Split: Dos 90% restantes, separa 22.2% para Validação (equivale a 20% do total)
df_treino, df_validacao = train_test_split(
    df_temp, test_size=0.222, random_state=42, stratify=df_temp['super_classe']
)

# Guardar os CSVs (úteis para o processo de FS e conferência de alvos)
df_treino.to_csv('treino.csv', index=False)
df_validacao.to_csv('validacao.csv', index=False)
df_teste.to_csv('teste.csv', index=False)

# Relatório de Separação no Terminal
def relatorio_estatistico(nome, df):
    print(f"\n--- Estrutura: {nome} ({len(df)} linhas) ---")
    contagem = df['super_classe'].value_counts()
    percent = df['super_classe'].value_counts(normalize=True) * 100
    for cl in contagem.index:
        print(f"{cl}: {contagem[cl]} amostras ({percent[cl]:.1f}%)")

relatorio_estatistico("TREINO", df_treino)
relatorio_estatistico("VALIDAÇÃO", df_validacao)
relatorio_estatistico("TESTE", df_teste)

# 4. PRÉ-PROCESSAMENTO PARA O MODELO
def preparar_pipeline(df):
    # Colunas que são IDs ou textos longos devem ser removidas para o treino
    colunas_drop = [
        'stac_id', 'collect_id', 'datetime', 'start_datetime', 'end_datetime', 
        'stac_browser_link', 'platform', 'constellation', 'frequency_band'
    ]
    X_aux = df.drop(columns=colunas_drop + ['super_classe'], errors='ignore')
    # Converte variáveis categóricas (texto) em colunas numéricas
    return pd.get_dummies(X_aux)

X_train = preparar_pipeline(df_treino)
X_val = preparar_pipeline(df_validacao)
X_test = preparar_pipeline(df_teste)

# Alinhamento: Garante que todos os conjuntos tenham as mesmas colunas após o dummies
X_train, X_val = X_train.align(X_val, join='left', axis=1, fill_value=0)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# Encoder para as classes (nao_urbano, urbano, costa)
le = LabelEncoder()
y_train = le.fit_transform(df_treino['super_classe'])
y_val = le.transform(df_validacao['super_classe'])
y_test = le.transform(df_teste['super_classe'])

# 5. TREINAMENTO (CONFIGURAÇÃO ANTI-OVERFITTING)
print("\nIniciando treino com parâmetros de regularização...")
modelo = xgb.XGBClassifier(
    n_estimators=1500,
    learning_rate=0.01,          # Passo lento para maior precisão
    max_depth=3,                 # Profundidade baixa evita que o modelo "decore" ruído
    min_child_weight=5,          # Requisito mínimo de amostras por folha
    subsample=0.8,               # Treina com 80% das linhas em cada árvore
    colsample_bytree=0.8,        # Treina com 80% das colunas em cada árvore
    reg_lambda=2,                # Penalidade L2
    objective='multi:softprob',
    num_class=len(le.classes_),
    random_state=42,
    early_stopping_rounds=50     # Para o treino se a validação parar de melhorar
)

# Treinamos monitorando simultaneamente Treino e Validação para o gráfico
modelo.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=False
)

# 6. DIAGNÓSTICO DE OVERFITTING (LOG LOSS)
resultados = modelo.evals_result()
plt.figure(figsize=(10, 6))
plt.plot(resultados['validation_0']['mlogloss'], label='Erro Treino')
plt.plot(resultados['validation_1']['mlogloss'], label='Erro Validação')
plt.title('Curva de Aprendizado: Diagnóstico de Overfitting')
plt.xlabel('Iterações (Árvores)')
plt.ylabel('Log Loss')
plt.legend()
plt.savefig('curva_aprendizado.png')

# 7. AVALIAÇÃO FINAL E GRÁFICOS
y_pred_train = modelo.predict(X_train)
y_pred_test = modelo.predict(X_test)

print("\n" + "="*45)
print(f"ACURÁCIA TREINO: {accuracy_score(y_train, y_pred_train):.2%}")
print(f"ACURÁCIA TESTE:  {accuracy_score(y_test, y_pred_test):.2%}")
print("="*45)

# Matriz de Confusão (Dados de Teste)
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Matriz de Confusão')
plt.savefig('matriz_confusao_classificacao.png')

# Importância das Features (As 15 principais)
plt.figure(figsize=(12, 8))
importances = pd.Series(modelo.feature_importances_, index=X_train.columns)
importances.nlargest(15).sort_values().plot(kind='barh', color='teal')
plt.title('Top 15 Características Decisivas')
plt.savefig('importancia_features.png')

print("\nProcesso concluído! Gráficos e CSVs gerados na pasta atual.")