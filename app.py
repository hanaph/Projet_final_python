import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.title("Analyse Exploratoire des Transactions")

@st.cache_data
def load_data():
    df = pd.read_csv("Transactions_data_complet.csv")
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    return df

df = load_data()

st.header("Exploration des Données")
st.write("Aperçu des données:")
st.dataframe(df.head())

st.write("Informations sur les données:")
st.text(df.info())

st.write("Valeurs manquantes:")
st.dataframe(df.isnull().sum())

st.header("Description des Données")
st.write("Statistiques descriptives des colonnes numériques:")
st.dataframe(df.describe().T)

numerical_cols = ['Amount', 'Value', 'PricingStrategy', 'FraudResult']
df_numerical = df[numerical_cols]
# Matrice de corrélation
correlation_matrix = df_numerical.corr()
# Affichage dans Streamlit
st.header("Corrélation entre les variables numériques")
fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
ax_corr.set_title('Carte thermique de corrélation des variables numériques')
st.pyplot(fig_corr)
st.markdown("**Conclusion:** Cette carte thermique indique une forte corrélation positive entre le montant (`Amount`) et la valeur (`Value`) des transactions, ce qui est attendu. On observe également une corrélation positive modérée entre le résultat de fraude (`FraudResult`) et `Amount` et `Value`, suggérant que les transactions de plus grande valeur ou montant pourraient être légèrement plus susceptibles d'être frauduleuses.")


st.header("Distribution des transactions dans le temps")

# Extraire les mois et année
df['YearMonth'] = df['TransactionStartTime'].dt.to_period('M')
year_months = sorted(df['YearMonth'].unique())
year_months_str = [str(ym) for ym in year_months]

# Add a selectbox for month selection
selected_year_month = st.selectbox("Sélectionnez un mois et une année", year_months_str)

# Filter data par mois
if selected_year_month:
    filtered_df = df[df['YearMonth'] == selected_year_month]
else:
    filtered_df = df

transactions_over_time = filtered_df.set_index('TransactionStartTime').resample('D').size()

fig, ax = plt.subplots(figsize=(15, 7))
transactions_over_time.plot(ax=ax)
ax.set_title(f'Nombre de transactions au fil du temps - {selected_year_month}')
ax.set_xlabel('Date')
ax.set_ylabel('Nombre de transactions')
ax.grid(True)

# Ajouter les couleurs dynamiquement
if not filtered_df.empty:
    weekend_days = transactions_over_time.index[(transactions_over_time.index.dayofweek == 5) | (transactions_over_time.index.dayofweek == 6)]
    for weekend in weekend_days:
        ax.axvspan(weekend, weekend + pd.Timedelta(days=1), color='yellow', alpha=0.3)


st.pyplot(fig)
st.markdown("**Conclusion:** Ce graphe nous montre clairement qu'il y a une baisse d'activité les weekends, visible par les creux dans le nombre de transactions pendant ces périodes (zones jaunes).")


# product category frequence
st.header("Fréquence des catégories de transactions")
fig, ax = plt.subplots(figsize=(12, 6))
sns.countplot(y='ProductCategory', data=df, order = df['ProductCategory'].value_counts().index, ax=ax)
ax.set_title('Fréquence des catégories de produits')
ax.set_xlabel('Nombre')
ax.set_ylabel('Catégorie de produit')
st.pyplot(fig)
st.markdown("**Conclusion:** Ce graphique montre que les catégories de produits 'airtime' et 'financial_services' sont de loin les plus fréquentes dans les transactions.")


# channel frequence
st.header("La fréquence de chaque canal.")
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(y='ChannelId', data=df, order = df['ChannelId'].value_counts().index, ax=ax)
ax.set_title('Fréquence des canaux')
ax.set_xlabel('Nombre')
ax.set_ylabel('ID du canal')
st.pyplot(fig)
st.markdown("**Conclusion:** Ce graphique indique que 'ChannelId_3' est le canal de transaction le plus fréquemment utilisé, suivi par 'ChannelId_2'.")


# Visualize average amount by product category
st.header("La moyenne des montants par Category de produits")
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x='Amount', y='ProductCategory', data=df, estimator=np.mean, order = df.groupby('ProductCategory')['Amount'].mean().sort_values(ascending=False).index, ax=ax)
ax.set_title('Montant moyen des transactions par catégorie de produit')
ax.set_xlabel('Montant moyen')
ax.set_ylabel('Catégorie de produit')
st.pyplot(fig)
st.markdown("**Conclusion:** Ce graphique montre que la catégorie de produit 'utility_bill' a le montant moyen de transaction le plus élevé, tandis que 'financial_services' a un montant moyen négatif (probablement dû à des remboursements ou des ajustements).")


# valeur total par pricing strategy
st.header("Valeur totale des transactions par stratégie de tarification")
strategy_performance = df.groupby('PricingStrategy')['Value'].sum().reset_index()
most_performing_strategy = strategy_performance.sort_values(by='Value', ascending=False)
st.write("Performance de chaque stratégie de tarification par valeur totale de transaction :")
st.dataframe(most_performing_strategy)
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='PricingStrategy', y='Value', data=most_performing_strategy, palette='viridis', ax=ax)
ax.set_title('Valeur totale des transactions par stratégie de tarification')
ax.set_xlabel('Stratégie de tarification')
ax.set_ylabel('Valeur totale de la transaction')
ax.tick_params(axis='x', rotation=0)
st.pyplot(fig)
st.markdown("**Conclusion:** Ce graphique démontre que la stratégie de tarification '2' génère la valeur totale de transactions la plus élevée, suivie par la stratégie '4'.")


# total fraud par pricing strategy
st.header("Nombre total de fraudes par stratégie de tarification")
fraud_by_strategy = df.groupby('PricingStrategy')['FraudResult'].sum().reset_index()
strategy_with_most_fraud = fraud_by_strategy.sort_values(by='FraudResult', ascending=False)
st.write("Nombre de fraudes par stratégie de tarification :")
st.dataframe(strategy_with_most_fraud)
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='PricingStrategy', y='FraudResult', data=strategy_with_most_fraud, palette='viridis', ax=ax)
ax.set_title('Nombre total de fraudes par stratégie de tarification')
ax.set_xlabel('Stratégie de tarification')
ax.set_ylabel('Nombre total de fraudes')
ax.tick_params(axis='x', rotation=0)
st.pyplot(fig)
st.markdown("**Conclusion:** Ce graphique indique que la stratégie de tarification '2', bien qu'elle génère le plus de valeur, est également associée au plus grand nombre de fraudes. Cependant, il est important de considérer le volume total de transactions pour chaque stratégie pour une analyse complète de la proportion de fraude.")


# top accounts by value
st.header("Les comptes avec le plus de transactions")
account_performance = df.groupby('AccountId')['Value'].sum().reset_index()
most_performing_accounts = account_performance.sort_values(by='Value', ascending=False)
st.write("Top 10 des comptes avec le plus de transactions:")
st.dataframe(most_performing_accounts.head(10))
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x='Value', y='AccountId', data=most_performing_accounts.head(10), palette='viridis', ax=ax)
ax.set_title('Top 10 des comptes ayant effectués le plus de transactions')
ax.set_xlabel('Valeur totale de la transaction')
ax.set_ylabel('ID du compte')
st.pyplot(fig)
st.markdown("**Conclusion:** Ce graphique met en évidence les 10 comptes qui ont généré la plus grande valeur totale de transactions. `AccountId_4249` se démarque nettement comme le compte ayant la plus haute valeur transactionnelle.")


# top accounts by fraud
st.header("Top 10 des comptes associés au plus grand nombre de fraudes :")
fraud_by_account = df.groupby('AccountId')['FraudResult'].sum().reset_index()
accounts_with_most_fraud = fraud_by_account.sort_values(by='FraudResult', ascending=False)
st.write("Top 10 des comptes associés au plus grand nombre de fraudes :")
st.dataframe(accounts_with_most_fraud.head(10))
st.markdown("**Conclusion:** Ce graphique identifie les 10 comptes associés au plus grand nombre de transactions frauduleuses. `AccountId_572` et `AccountId_4421` présentent un nombre de fraudes considérablement plus élevé que les autres comptes.")


# total vs transactions frauduleuses pour les top fraud accounts
st.header("Total vs Transactions frauduleuse pour les comptes les plus frauduleux")
total_transactions_per_account = df.groupby('AccountId').size().reset_index(name='TotalTransactions')
accounts_fraud_and_total = pd.merge(
    accounts_with_most_fraud.head(10),
    total_transactions_per_account,
    on='AccountId',
    how='left'
).fillna(0)

accounts_fraud_and_total_melted = accounts_fraud_and_total.melt(
    id_vars='AccountId',
    value_vars=['TotalTransactions', 'FraudResult'],
    var_name='TransactionType',
    value_name='Count'
)
accounts_fraud_and_total_melted['TransactionType'] = accounts_fraud_and_total_melted['TransactionType'].replace({'FraudResult': 'Fraudulent Transactions'})

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(
    x='Count',
    y='AccountId',
    hue='TransactionType',
    data=accounts_fraud_and_total_melted,
    palette={'TotalTransactions': 'skyblue', 'Fraudulent Transactions': 'salmon'},
    dodge=False,
    ax=ax
)
ax.set_title('Total vs Transactions frauduleuse')
ax.set_xlabel('Nombre de transactions')
ax.set_ylabel('Account ID')
ax.legend(title='Transaction Type')
st.pyplot(fig)
st.markdown("**Conclusion:** Ce graphique compare le nombre total de transactions et le nombre de transactions frauduleuses pour les comptes les plus frauduleux. Bien que certains comptes aient un nombre total élevé de transactions, la proportion de transactions frauduleuses varie, indiquant que le nombre total de transactions n'est pas le seul facteur de fraude pour ces comptes.")
