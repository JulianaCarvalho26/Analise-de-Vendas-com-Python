#!/usr/bin/env python
# coding: utf-8

# # Análise de Dados de Vendas
# 
# Para se identificar padrões e tendências a fim de gerar insights e possíveis estratégias a serem colocadas em prática em uma empresa no ramo de Varejo, a análise de dados de vendas é essencial. Perguntas como "Quais categorias de produtos tem mais saída?", "Quais os produtos mais vendidos?" e "Quais são as principais características dos consumidores?" são algumas das perguntas de negócio que podem gerar bons resultados para esse tipo de análise.
# 
# Neste Jupyter Notebook trago um exemplo sucinto de análise de vendas com linguagem Python e suas principais bibliotecas para análise de dados. O dataset utilizado foi "Retail Sales Dataset. Unveiling Retail Trends: A Dive into Sales Patterns and Customer Profiles" disponível no link: https://www.kaggle.com/datasets/mohammadtalib786/retail-sales-dataset/data

# In[28]:


# Versão Python utilizada

from platform import python_version
print("Versão Python utilizada neste Jupyter Notebook:", python_version())


# In[29]:


# Importando bibliotecas necessárias

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[30]:


# Carregando o dataset "shopping_trends" em um dataframe para ser analisado

df_shop = pd.read_csv("./shopping_trends.csv")


# In[31]:


# Verificando quantidade de linhas e colunas

df_shop.shape


# In[32]:


# Verificando amostra do dataframe

df_shop.head(10)


# In[33]:


# Verificando o tipo de dado de cada coluna

df_shop.dtypes


# In[34]:


# Verificando se há registros duplicados

df_shop[df_shop.duplicated()]


# In[35]:


# Verificando se há valores ausentes

df_shop.isnull().sum()


# In[36]:


# Verificando valor total de vendas

sales = df_shop["Purchase Amount (USD)"].sum()
print("O valor total de vendas foi de: U$", sales)


# In[37]:


# Verificando valor total de itens vendidos em cada categoria e adicionando em outro dataframe

df_totals = df_shop.groupby("Category")["Purchase Amount (USD)"].sum().reset_index()

# Renomeando as colunas do novo df após fazer o agrupamento

df_totals_items = df_totals.rename(columns={"Category": "Categoria", "Purchase Amount (USD)": "Valor Total de Vendas"})

# Calculando as porcentagens em relação ao valor total de vendas

df_totals_items["Porcentagem"] = round((df_totals_items["Valor Total de Vendas"] / sales) * 100, 2)
df_totals_items


# In[38]:


# Verificando quantidade de itens vendidos por categoria e adicionando em outro dataframe

df_cat_items = df_shop.groupby("Category")["Item Purchased"].count().reset_index()

df_cat = df_cat_items.rename(columns={"Category": "Categoria", "Item Purchased": "Total de Vendas"})

df_cat


# In[39]:


# Plotando o resultado em um gráfico de barras

df_cat.plot.bar(x = "Categoria", y = "Total de Vendas", color = ['lightblue', 'yellow', 'lightpink', 'salmon' ])
plt.xticks(rotation = 45)
plt.xlabel("Categoria")
plt.ylabel("Total de itens vendidos")
plt.title("Total de itens vendidos por categoria", fontsize = 15)
plt.show()


# ## Conclusão 1
# 
# Roupas(Clothing) é a categoria mais vendida com o total de 1737 itens vendidos, representando 44,73% do valor total de vendas da empresa.
# Em seguida, vem as categorias de Acessórios(Accessories), Calçados(Footwear) e Agasalhos(Outerwear).

# In[40]:


# Verificando quais itens foram os mais vendidos

df_itens1 = df_shop.groupby("Item Purchased")["Category"].count()
df_itens1.sort_values(ascending = False)


# In[41]:


# Plotando o Top 5 itens mais vendidos em um gráfico de barras

df_itens_top5 = df_itens1.sort_values(ascending = False).head(5)

df_itens_top5.plot.barh(color = ['palegreen', 'lightblue', 'mistyrose', 'lightpink', 'salmon' ])
plt.xlabel("Quantidade")
plt.ylabel("Item")
plt.title("Top 5 Itens Mais Vendidos", fontsize = 15)
plt.show()


# In[42]:


# Verificando quais itens foram os mais vendidos

df_itens2 = df_shop.groupby("Item Purchased")["Category"].count()
df_itens2.sort_values(ascending = True)


# In[43]:


# Plotando os 5 itens menos vendidos em um gráfico de barras

df_itens2_5 = df_itens2.sort_values(ascending = True).head(5)

df_itens2_5.plot.barh(color =  ['palegreen', 'lightblue', 'mistyrose', 'lightpink', 'salmon' ])
plt.xlabel("Quantidade")
plt.ylabel("Item")
plt.title("Os 5 Itens Menos Vendidos", fontsize = 15)
plt.show()


# ## Conclusão 2
# 
# Os 5 itens mais vendidos são Vestidos(Dress), Camisetas(Shirt), Calças(Pants), Blusas(Blouse) e Jóias(Jewelry). Enquanto os 5 itens menos vendidos são Tênis(Sneakers), Botas(Boots), Mochilas(Backpack), Luvas(Gloves) e Jeans.

# In[44]:


# Verificando gênero da maior parte dos clientes

df_gen = df_shop.groupby("Gender")["Gender"].count()
df_gen


# In[71]:


# Definindo os valores e labels do gráfico de pizza

label = ["Feminino", "Masculino"]
values = [1248, 2652]

# Definindo uma função para transformar os valores em porcentagens

def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.1f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct

# Plotando um gráfico de pizza para apresentar a porcentagem de compradores de cada gênero

plt.pie(values,
       labels = label,
       colors = ['lightblue', 'salmon'],
       autopct = make_autopct(values),
       shadow = True,
       startangle = 90)
plt.title("Clientes por Gênero")
plt.show()


# In[46]:


# Verificando a idade dos clientes por gênero

df_age = df_shop[["Gender", "Age"]]
df_age


# In[69]:


# Verificando dados estastísticos dos clientes masculinos

df_age_male_filter = df_age["Gender"] == "Male"
df_age_male = df_age[df_age_male_filter]
df_age_male.describe()


# In[70]:


# Verificando dados estastísticos dos clientes femininos

df_age_female_filter = df_age["Gender"] == "Female"
df_age_female = df_age[df_age_female_filter]
df_age_female.describe()


# In[51]:


# Plotando um gráfico boxplot para mostrar as idades dos clientes em cada gênero

sns.boxplot(data = df_age, x="Gender", y="Age", hue="Gender", palette = "pastel")


# ## Conclusão 3
# 
# A maior parte dos clientes são do gênero masculino, representando 68% do total de clientes.
# 
# A faixa etária onde se concentram a maior parte dos clientes para ambos os gêneros está entre 31 e 57 anos.

# In[74]:


# Verificando o valor total de vendas e a quantidade de vendas em cada estado

# Agrupando por estado o valor total e a quantidade de vendas em cada
df_st = df_shop.groupby("Location").agg({"Purchase Amount (USD)":"sum", 
                                             "Customer ID": "count"}).reset_index().sort_values("Purchase Amount (USD)", 
                                                                                                 ascending = False)
# Renomeando as colunas do novo df após fazer o agrupamento
df_states = df_st.rename(columns={'Location': 'Estado', 'Purchase Amount (USD)': 'Valor Total de Vendas', 'Customer ID': 'Quantidade de Vendas'})

# Salvando apenas os 10 primeiros estados com maior valor total e quantidade de vendas
df_states_top10 = df_states.head(10)
df_states_top10


# In[75]:


# Plotando um gráfico de dispersão para mostrar os 10 primeiros estados com maior valor total e quantidade de vendas

sns.relplot(x="Valor Total de Vendas", y="Quantidade de Vendas", hue="Estado", size="Valor Total de Vendas",
            sizes=(80, 400), alpha=.8, palette="muted",
            height=5, data=df_states_top10 )


# ## Conclusão 4
# 
# Os estados com maior número de vendas e maior valor total de vendas são, em ordem decrescente, Montana, Illinois, California, Idaho, Nevada, Alabama, New York, North Dakota, West Virginia e Nebraska.
