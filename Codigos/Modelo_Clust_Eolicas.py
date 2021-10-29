##########################################################
##########################################################
# Modelos de clusterização de usinas eólicas
##########################################################


# %%
# Pacotes necessários a ser instalados
# !pip install "pyproj-3.1.0-cp38-cp38-win_amd64.whl"
# !pip install "Rtree-0.9.7-cp38-cp38-win_amd64.whl"
# !pip install "Shapely-1.7.1-cp38-cp38-win_amd64.whl"
# !pip install "GDAL-3.3.0-cp38-cp38-win_amd64.whl"
# !pip install "Fiona-1.8.20-cp38-cp38-win_amd64.whl"
# !pip install geopandas

# %%
# # Leitura bibliotecas
from typing import Dict
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import os

np.random.seed(42)

# %%
# # Entrada de Dados
directory = os.getcwd()
directory=directory.replace('Codigos', '')


ARQ_USINAS = directory+"Dados/Usinas_Fic.csv"
ARQ_DADOS = directory+"Dados/Dados_Fic.csv"

df_usinas = pd.read_csv(ARQ_USINAS)
df_usinas = df_usinas.dropna()
df_usinas.set_index("cod", inplace=True)
df_dados = pd.read_csv(ARQ_DADOS)
df_dados = df_dados.dropna()
# Lê os dados de formas dos estados
fp = directory+"Dados/shapes/bra_admbnda_adm1_ibge_2020.shp"
df_estados = gpd.read_file(fp)


# %% 
# # Preparação de dados

def normaliza(df: pd.DataFrame):
    atributos = df.describe().T
    return (df - atributos["mean"]) 
    # return (df - atributos["mean"]) / atributos["std"]

df_dados_norm = normaliza(df_dados)
df_dados_norm = df_dados_norm.T
df_usinas_NE = df_usinas[df_usinas.index.isin(df_dados_norm.index)]
df_dados_norm


# %% 
# # Visualiza correlação entre usinas

COLORMAP = "tab20"

def amostra_colormap(elementos: int):
    # cmap = matplotlib.cm.get_cmap(COLORMAP)
    # pontos = np.linspace(0, 1, elementos)
    # return [cmap(p) for p in pontos]
    import matplotlib._color_data as mcd
    colors = {name for name in mcd.CSS4_COLORS
               if "xkcd:" + name in mcd.XKCD_COLORS}
    colors = pd.DataFrame(colors)[3:len(colors)]
    colors = list(colors[0])
    colors = [*colors, *colors]
    # print(colors[:elementos])
    colors[int(np.where(np.array(colors)=='lavender')[0][0])]='black'
    colors[int(np.where(np.array(colors)=='tan')[0][0])]='green'    
    colors[0]='blue'
    colors[1]='red'
    colors[26]='blue'
    colors[27]='red'
    return colors[:elementos]


def gera_ticks_labels_estados(df_usinas: pd.DataFrame):
    estados = df_usinas["Estado"].unique()
    usinas_por_estado = df_usinas.groupby("Estado").count()["Nome"]
    ticks_por_estado = []
    estados_visitados = []
    for e in estados:
        anteriores = sum([usinas_por_estado[ev] for ev in estados_visitados])
        tick = anteriores + int(usinas_por_estado[e] / 2)
        ticks_por_estado.append(tick)
        estados_visitados.append(e)
    return ticks_por_estado, estados_visitados

def gera_cores_estados(df_usinas: pd.DataFrame):
    estados = df_usinas["Estado"].unique()
    cores = amostra_colormap(len(estados))
    lut = dict(zip(estados, cores))
    return df_usinas["Estado"].map(lut).to_numpy()

def cm_saturado(valor: float):
    return LinearSegmentedColormap.from_list("cm_saturado",
                                             list(zip([0.0, valor, 1.0],
                                                      ["white", "white", "darkblue"])))

def visualiza_correlacao_usinas_estados(corr: pd.DataFrame, df_usinas: pd.DataFrame):
    cores_mapa = gera_cores_estados(df_usinas)
    ticks, labels = gera_ticks_labels_estados(df_usinas)
    g = sns.clustermap(corr,
                    row_colors=cores_mapa,col_colors=cores_mapa,
                    row_cluster=False,col_cluster=False,
                    cbar_pos=None,
                    dendrogram_ratio=0.0,
                    colors_ratio=0.01,
                    cmap=cm_saturado(0.5),
                    figsize=(9,9),
                    xticklabels=False,
                    yticklabels=False)
    g.ax_row_colors.set_yticks(ticks)
    g.ax_row_colors.set_yticklabels(labels)
    g.ax_col_colors.set_xticks(ticks)
    g.ax_col_colors.set_xticklabels(labels)
    return g


def visualiza_usinas_espacial_estados(df_usinas: pd.DataFrame,
                                      df_estados: gpd.GeoDataFrame,
                                      subsis: str):
    fig, ax = plt.subplots(figsize=(9, 9))
    estados = df_usinas["Estado"].unique()
    cores = amostra_colormap(len(estados))

    for i, e in enumerate(estados):
        df_usinas_estado = df_usinas.loc[df_usinas["Estado"] == e, :]
        df_usinas_estado.plot(ax=ax, kind="scatter", x="lat", y="long",
                              color=cores[i], zorder=1, label=e,s=80)
    if subsis == "NE":
        for e in ["Alagoas", "Bahia", "Ceará", "Maranhão",
                  "Paraíba", "Piauí", "Pernambuco",
                  "Rio Grande do Norte", "Sergipe"]:
            df_est = df_estados[df_estados["ADM1_PT"] == e]
            df_est.boundary.plot(ax=ax, zorder=0)
    elif subsis == "S":
        for e in ["Paraná", "Rio Grande do Sul", "Santa Catarina"]:
            df_est = df_estados[df_estados["ADM1_PT"] == e]
            df_est.boundary.plot(ax=ax, zorder=0)

    plt.legend()
    plt.xlabel("Long")
    plt.ylabel("Lat")
    # plt.tight_layout()


# %%
# Plota correlação por estados


corr_ventos = df_dados_norm.T.corr().abs()

matplotlib.rcParams.update(matplotlib.rcParamsDefault)
size=20
params = {'legend.fontsize':16,
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size*0.8,
          'ytick.labelsize': size*0.8,
          'axes.titlepad': 25,
          'legend.loc': 'upper right'}
plt.rcParams.update(params)
visualiza_correlacao_usinas_estados(corr_ventos, df_usinas_NE)
plt.savefig(directory+'Saidas/Cor_Usi_Est_NE.pdf', bbox_inches="tight")
plt.show()

# %%
# Plota cluster por estados
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
size=20
params = {'legend.fontsize': 16,
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size*0.8,
          'ytick.labelsize': size*0.8,
          'axes.titlepad': 25,
          'legend.loc': 'upper right'}
plt.rcParams.update(params)
visualiza_usinas_espacial_estados(df_usinas_NE, df_estados, "NE")
plt.savefig(directory+'Saidas/Clust_Est_NE.pdf')
plt.show()

# %% 
# ## Redução de dimensionalidade

# %%
pca = PCA(n_components=df_dados_norm.shape[0])
pca.fit(df_dados_norm)

matplotlib.rcParams.update(matplotlib.rcParamsDefault)
size=25
params = {'legend.fontsize': 20,
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size*0.8,
          'ytick.labelsize': size*0.8,
          'axes.titlepad': 25,
          'legend.loc': 'upper right'}
plt.rcParams.update(params)
fig = plt.figure(figsize=(10, 9))
pca_acum = pca.explained_variance_ratio_.cumsum()
plt.plot(pca_acum*100,  'ro-', linewidth=2)
plt.title('PCA')
plt.xlabel('Número de Componentes')
plt.ylabel('Variância Acumulada (%)')
plt.savefig(directory+'Saidas/PCA1_NE.pdf')
plt.show()


# %%
# Define parâmentros da 1ª clusterização
LIMIAR_VARIANCIA = 0.7
N_COMPONENTES = next(i for i, x in enumerate(pca_acum) if x >= LIMIAR_VARIANCIA)


# %%
# Aplica o PCA
pca = PCA(n_components=N_COMPONENTES)
pca.fit(df_dados_norm)
df_dados_reduzidos = pd.DataFrame(pca.transform(df_dados_norm),
                                  index=df_dados_norm.index)
df_dados_reduzidos

# %% 
# # Clusterização de Primeiro Nível

def gera_ticks_labels_clusters(df_usinas: pd.DataFrame):
    clusters = df_usinas["Cluster"].unique()
    usinas_por_cluster = df_usinas.groupby("Cluster").count()["Nome"]
    ticks_por_cluster = []
    clusters_visitados = []
    for e in clusters:
        anteriores = sum([usinas_por_cluster[ev] for ev in clusters_visitados])
        tick = anteriores + int(usinas_por_cluster[e] / 2)
        ticks_por_cluster.append(tick)
        clusters_visitados.append(e)
    return ticks_por_cluster, clusters_visitados

def gera_cores_clusters(df_usinas: pd.DataFrame):
    clusters = df_usinas["Cluster"].unique()
    cores = amostra_colormap(len(clusters))
    lut = dict(zip(clusters, cores))
    return df_usinas["Cluster"].map(lut).to_numpy()

def cm_saturado(valor: float):
    return LinearSegmentedColormap.from_list("cm_saturado",
#                                             list(zip([0.0, valor-0.01,valor, 1.0],
#                                                      ["white", "white", "deepskyblue", "darkblue"])))
                                             list(zip([0.0, valor, 1.0],
                                                      ["white", "white", "darkblue"])))

def visualiza_correlacao_usinas_clusters(corr: pd.DataFrame,
                                         df_usinas: pd.DataFrame,
                                         clusters: np.ndarray):
    # Prepara os dados, ordenando segundo os clusters
    corr_ordenado = corr.copy()
    df_usinas_ordenado = df_usinas.copy()
    df_usinas_ordenado["Cluster"] = clusters
    df_usinas_ordenado = df_usinas_ordenado.sort_values(by=["Cluster"])
    ordem = df_usinas_ordenado.index.tolist()
    corr_ordenado = corr_ordenado.reindex(ordem)
    corr_ordenado = corr_ordenado[ordem]
    # Gera o plot
    cores_mapa = gera_cores_clusters(df_usinas_ordenado)
    ticks, labels = gera_ticks_labels_clusters(df_usinas_ordenado)
    g = sns.clustermap(corr_ordenado,
                       row_colors=cores_mapa,col_colors=cores_mapa,
                       row_cluster=False,col_cluster=False,
                       cbar_pos=None,
                       dendrogram_ratio=0.0,
                       colors_ratio=0.01,
                       cmap=cm_saturado(0.7),
                       figsize=(10,10),
                       xticklabels=False,
                       yticklabels=False)
    g.ax_row_colors.set_yticks(ticks)
    g.ax_row_colors.set_yticklabels(labels)
    g.ax_col_colors.set_xticks(ticks)
    g.ax_col_colors.set_xticklabels(labels)
    return g

def visualiza_usinas_espacial_clusters(df_usinas: pd.DataFrame,
                                       clusters: list,
                                       df_estados: gpd.GeoDataFrame,
                                       subsis: str,
                                       plt_geo: bool):
    fig, ax = plt.subplots(figsize=(9, 9))
    df_usinas_copia = df_usinas.copy()
    df_usinas_copia["Cluster"] = clusters
    clusters_dif = df_usinas_copia["Cluster"].unique()
    clusters_dif.sort()
    cores = amostra_colormap(len(clusters_dif))
    for i, e in enumerate(clusters_dif):
        df_usinas_copia_estado = df_usinas_copia.loc[df_usinas_copia["Cluster"] == e, :]
        df_usinas_copia_estado.plot(ax=ax, kind="scatter", x="lat", y="long",
                                    color=cores[i], zorder=1, label=e,s=80,alpha=1,marker='o')
    plt.legend()
    plt.xlabel("Long")
    plt.ylabel("Lat")
    
    if(plt_geo):
        if subsis == "NE":
            for e in ["Alagoas", "Bahia", "Ceará", "Maranhão",
                      "Paraíba", "Piauí", "Pernambuco",
                      "Rio Grande do Norte", "Sergipe"]:
                df_est = df_estados[df_estados["ADM1_PT"] == e]
                df_est.boundary.plot(ax=ax, zorder=0)
        elif subsis == "S":
            for e in ["Paraná", "Rio Grande do Sul", "Santa Catarina"]:
                df_est = df_estados[df_estados["ADM1_PT"] == e]
                df_est.boundary.plot(ax=ax, zorder=0)

    # plt.tight_layout()
    # return fig

# %% 
# Aplica GMM do 1º nível

N_CLUSTERS = 3

gmm = GaussianMixture(n_components=N_CLUSTERS, random_state=0,init_params='kmeans')
gmm.fit(df_dados_reduzidos)
clusters = gmm.predict(df_dados_reduzidos)
clusters = [str(c + 1) for c in clusters]

# %%
# Plota correlação da clusterização de 1º nível
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
size=25
params = {'legend.fontsize': 20,
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size*0.8,
          'ytick.labelsize': size*0.8,
          'axes.titlepad': 25,
          'legend.loc': 'upper right'}
plt.rcParams.update(params)
visualiza_correlacao_usinas_clusters(corr_ventos, df_usinas_NE, clusters)
plt.savefig(directory+'Saidas/Cor_clust_nivel1_NE.pdf', bbox_inches="tight")
plt.show()

# %%
# Plota clusterização da clusterização de 1º nível
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
size=20
params = {'legend.fontsize': 16,
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size*0.8,
          'ytick.labelsize': size*0.8,
          'axes.titlepad': 25,
          'legend.loc': 'upper right'}
plt.rcParams.update(params)
visualiza_usinas_espacial_clusters(df_usinas_NE, clusters, df_estados, "NE",True)
plt.savefig(directory+'Saidas/Clust_nivel1_NE.pdf')
plt.show()



# %%
# Define parâmentros da 2ª clusterização
LIMIAR_CORR_MEDIA_RECLUSTER_1 = 0.7

# %% 
# # Reclusterização

def filtra_usinas_cluster(df_usinas: pd.DataFrame,
                          df_dados: pd.DataFrame,
                          clusters: np.ndarray,
                          cluster_desejado: int):
    df_usinas_cluster = df_usinas.copy()
    df_usinas_cluster["Cluster"] = clusters
    filtro = df_usinas_cluster["Cluster"] == cluster_desejado
    df_usinas_cluster = df_usinas_cluster.loc[filtro, :]

    indices_usinas_cluster = list(df_usinas_cluster.index)
    df_usinas_filtrado = df_usinas.loc[df_usinas.index.isin(indices_usinas_cluster), :]
    df_dados_filtrado = df_dados.loc[df_dados.index.isin(indices_usinas_cluster), :]
    return df_usinas_filtrado, df_dados_filtrado


def pca_cluster(df_usinas: pd.DataFrame,
                df_dados: pd.DataFrame,
                clusters: np.ndarray,
                clusters_desejado: int):

    df_usinas_cluster, df_dados_cluster = filtra_usinas_cluster(df_usinas,
                                                                df_dados,
                                                                clusters,
                                                                clusters_desejado)
    pca = PCA(n_components=df_dados_cluster.shape[0])
    pca.fit(df_dados_cluster)
    pca_acum = pca.explained_variance_ratio_.cumsum()
    n_componentes = next(i for i, x in enumerate(pca_acum) if x >= LIMIAR_VARIANCIA)
    n_componentes = max([2, n_componentes])
    pca = PCA(n_components=n_componentes)
    pca.fit(df_dados_cluster)
    df_dados_reduzidos = pd.DataFrame(pca.transform(df_dados_cluster),
                                      index=df_dados_cluster.index)
    return df_usinas_cluster, df_dados_reduzidos

def avalia_clusters_recluster(df_usinas: pd.DataFrame,
                              df_dados: pd.DataFrame,
                              clusters: np.ndarray):
    clusters_unicos = list(set(list(clusters)))
    clusters_recluster = []
    df_usinas_cluster = df_usinas.copy()
    df_usinas_cluster["Cluster"] = clusters
    for c in clusters_unicos:
        dfu, dfd = filtra_usinas_cluster(df_usinas, df_dados, clusters, c)
        corr = dfd.T.corr().abs()
        corr[corr < LIMIAR_VARIANCIA] = 0
        corr[corr >= LIMIAR_VARIANCIA] = 1
        if np.mean(corr.to_numpy()) < LIMIAR_CORR_MEDIA_RECLUSTER_1:
            clusters_recluster.append(c)

    return clusters_recluster

def recluster(dfs_dados_reduzidos: Dict[int, pd.DataFrame],
              num_clusters: np.ndarray) -> Dict[int, list]:
    reclusters = {}
    clusters = list(dfs_dados_reduzidos.keys())
    for c, n in zip(clusters, num_clusters):
        gmm = GaussianMixture(n_components=n, random_state=0)
        dados = dfs_dados_reduzidos[c]
        gmm.fit(dados)
        clusters_rec = gmm.predict(dados)
        clusters_rec += 1
        rec = [f"{c}.{i}" for i in clusters_rec]
        reclusters[c] = rec
    return reclusters

def atualiza_clusters_com_recluster(df_usinas: pd.DataFrame,
                                    clusters_antigos: np.ndarray,
                                    usinas_recluster: Dict[str, pd.DataFrame],
                                    reclusters: Dict[str, np.ndarray]):
    df_usinas_copia = df_usinas.copy()
    df_usinas_copia["Cluster"] = clusters_antigos
    clusters_recluster = list(usinas_recluster.keys())
    for c in clusters_recluster:
        usinas = usinas_recluster[c]
        clusters_novos = reclusters[c]
        for idx, cn in zip(list(usinas.index), clusters_novos):
            df_usinas_copia.loc[idx, "Cluster"] = cn
    return df_usinas_copia["Cluster"].to_numpy()

def recluster_recursivo(df_usinas: pd.DataFrame,
                        df_dados: pd.DataFrame,
                        clusters_iniciais: np.ndarray):
    clusters_recluster = avalia_clusters_recluster(df_usinas_NE,
                                                   df_dados_norm,
                                                   clusters_iniciais)
    # Se já passa nos critérios, não precisa de outra clusterização
    if len(clusters_recluster) == 0:
        return clusters_iniciais

    # Senão, começa a reclusterizar
    dfs_usinas_recluster = {}
    dfs_dados_reduzidos_recluster = {}
    for c in clusters_recluster:
        df_u, df_d = pca_cluster(df_usinas, df_dados, clusters_iniciais, c)
        dfs_usinas_recluster[c] = df_u
        dfs_dados_reduzidos_recluster[c] = df_d
    n_clusters_recluster = {c: 2 for c in clusters_recluster}
    while True:
        # Reclusteriza
        reclusters = recluster(dfs_dados_reduzidos_recluster,
                               list(n_clusters_recluster.values()))
        clusters_com_recluster = atualiza_clusters_com_recluster(df_usinas,
                                                                 clusters_iniciais,
                                                                 dfs_usinas_recluster,
                                                                 reclusters)
        # Testa novamente
        clusters_recluster = avalia_clusters_recluster(df_usinas,
                                                       df_dados,
                                                       clusters_com_recluster)
        # Se passou, termina e retorna. Senão, incrementa em 1 os que não passaram
        if len(clusters_recluster) == 0:
            return clusters_com_recluster
        else:
            for c in clusters_recluster:
                cluster_original = c.split(".")[0]
                n_clusters_recluster[cluster_original] += 1


# %%
# Aplica a reclusterização
clusters_com_recluster = recluster_recursivo(df_usinas_NE, df_dados_norm, clusters)

# %%
# Plota correlação da clusterização de 2º nível
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
size=25
params = {'legend.fontsize': 20,
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size*0.5,
          'ytick.labelsize': size*0.5,
          'axes.titlepad': 25,
          'legend.loc': 'upper right'}
plt.rcParams.update(params)
visualiza_correlacao_usinas_clusters(corr_ventos, df_usinas_NE, clusters_com_recluster)
plt.savefig(directory+'Saidas/Cor_clust_nivel2_NE.pdf', bbox_inches="tight")
plt.show()

# %%
# Plota clusterização da clusterização de 2º nível
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
size=20
params = {'legend.fontsize': 11,
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size*0.8,
          'ytick.labelsize': size*0.8,
          'axes.titlepad': size,
          'legend.loc': 'upper right'}
plt.rcParams.update(params)
visualiza_usinas_espacial_clusters(df_usinas_NE, clusters_com_recluster, df_estados, "NE",True)
plt.savefig(directory+'Saidas/Clust_nivel2_NE.pdf')
plt.show()



# %%
# Define parâmentros da 3ª clusterização
MIN_CORR_MEDIA = 0.7
LIMIAR_CORR_MEDIA_QUANTIL = 0.85
QUANTIL = 0.15

# %% 
# # Clusterização Hierárquica

def correlacao_media_interna_clusters(df_usinas: pd.DataFrame,
                                      df_dados: pd.DataFrame,
                                      clusters_com_recluster: np.ndarray):
    df_usinas_copia = df_usinas.copy()
    df_usinas_copia["Cluster"] = clusters_com_recluster
    clusters_unicos = list(set(list(clusters_com_recluster)))
    df_correlacao_media = None
    for c in clusters_unicos:
        usinas_cluster = df_usinas_copia.loc[df_usinas_copia["Cluster"] == c, :]
        codigos_usinas_cluster = list(usinas_cluster.index)
        dados_usinas_cluster = df_dados.loc[codigos_usinas_cluster, :]
        media_interna = np.mean(dados_usinas_cluster.to_numpy(), axis=0)
        correlacoes = {}
        for idx, linha in dados_usinas_cluster.iterrows():
            correlacoes[idx] = [np.corrcoef(linha.to_numpy(), media_interna)[1, 0]]
        df = pd.DataFrame(data=correlacoes).T
        df.columns = ["Correlação"]
        df["Cluster"] = c
        if df_correlacao_media is None:
            df_correlacao_media = df
        else:
            df_correlacao_media = pd.concat([df_correlacao_media, df])
    return df_correlacao_media

def visualiza_correlacao_media_interna(df_correlacao: pd.DataFrame):
    clusters_unicos = list(set(df_correlacao["Cluster"].tolist()))
    clusters_unicos.sort()
    # Descobre o número de plots a serem exibidos
    n_plots = int(np.ceil(np.sqrt(len(clusters_unicos))))
    fig, axs = plt.subplots(n_plots, n_plots, figsize=(22, 10))
    for i, c in enumerate(clusters_unicos):
        # Extrai os dados do cluster
        dfc = df_correlacao.loc[df_correlacao["Cluster"] == c, :]
        n_usinas = dfc.shape[0]
        # Plota
        ix = int(i / n_plots)
        iy = i % n_plots
        dfc.plot.bar(ax=axs[ix, iy], y="Correlação", color="gray", legend=False)
        axs[ix, iy].hlines(MIN_CORR_MEDIA, -1, n_usinas + 1, linestyle="dashed", color="black")
        axs[ix, iy].set_title(c)
    # Deleta os eixos extras
    for i in range(len(clusters_unicos), int(n_plots**2)):
        ix = int(i / n_plots)
        iy = i % n_plots
        fig.delaxes(axs[ix][iy]) 
    # Retorna a figura
    plt.tight_layout()


def avalia_correlacao_media_interna(df_correlacao: pd.DataFrame):
    clusters_unicos = list(set(df_correlacao["Cluster"].tolist()))
    clusters_unicos.sort()
    clusters_recluster_hierarquica = []
    for i, c in enumerate(clusters_unicos):
        # Extrai os dados do cluster
        dfc = df_correlacao.loc[df_correlacao["Cluster"] == c, :]
        # Confere os critérios
        if any([dfc["Correlação"].min() < MIN_CORR_MEDIA,
                dfc["Correlação"].quantile(QUANTIL) < LIMIAR_CORR_MEDIA_QUANTIL]):
            clusters_recluster_hierarquica.append(c)
    return clusters_recluster_hierarquica


def recluster_hierarquico(dfs_dados_reduzidos: Dict[str, pd.DataFrame],
                          dfs_correl: Dict[str, pd.DataFrame],
                          num_clusters: np.ndarray) -> Dict[str, list]:
    reclusters = {}
    clusters = list(dfs_dados_reduzidos.keys())
    for c, n in zip(clusters, num_clusters):
        # Calcula a matriz de distâncias
        dados = dfs_correl[c]
        dist = dados.to_numpy()
        dist = 1 - np.square(dist)
        # Faz a clusterização
        hier = AgglomerativeClustering(n_clusters=n,
                                       affinity="precomputed",
                                       linkage="average",
                                       distance_threshold=None)
        hier.fit(dist)
        clusters_rec = hier.labels_
        clusters_rec += 1
        rec = [f"{c}-{i}" for i in clusters_rec]
        reclusters[c] = rec
    return reclusters

def recluster_hierarquico_recursivo(df_usinas: pd.DataFrame,
                                    df_dados: pd.DataFrame,
                                    clusters_iniciais: np.ndarray):
    correl_media = correlacao_media_interna_clusters(df_usinas, df_dados, clusters_iniciais)
    clusters_recluster = avalia_correlacao_media_interna(correl_media)
    # Se já passa nos critérios, não precisa de outra clusterização
    if len(clusters_recluster) == 0:
        return clusters_iniciais
    # Senão, começa a reclusterizar
    dfs_usinas_recluster = {}
    dfs_dados_recluster = {}
    dfs_correl_dados_recluster = {}
    for c in clusters_recluster:
        df_u, df_d = filtra_usinas_cluster(df_usinas,
                                           df_dados,
                                           clusters_iniciais,
                                           c)
        dfs_usinas_recluster[c] = df_u
        dfs_dados_recluster[c] = df_d
        correl = df_d.T.corr().abs()
        dfs_correl_dados_recluster[c] = correl
    n_clusters_recluster = {c: 2 for c in clusters_recluster}
    while True:
        # Reclusteriza
        reclusters = recluster_hierarquico(dfs_dados_recluster,
                                           dfs_correl_dados_recluster,
                                           list(n_clusters_recluster.values()))
        clusters_com_recluster = atualiza_clusters_com_recluster(df_usinas,
                                                                 clusters_iniciais,
                                                                 dfs_usinas_recluster,
                                                                 reclusters)
        # Testa novamente
        correl_media = correlacao_media_interna_clusters(df_usinas,
                                                         df_dados,
                                                         clusters_com_recluster)
        clusters_recluster = avalia_correlacao_media_interna(correl_media)
        # Se passou, termina e retorna. Senão, incrementa em 1 os que não passaram
        if len(clusters_recluster) == 0:
            return clusters_com_recluster
        else:
            for c in clusters_recluster:
                cluster_original = c.split("-")[0]
                n_clusters_recluster[cluster_original] += 1


# %%
# Gera a correlação com a média interna
correl_media = correlacao_media_interna_clusters(df_usinas_NE,
                                                 df_dados_norm,
                                                 clusters_com_recluster)

# %%
# Plota correlação dcom a média interna até o 2º nível
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
size=7
params = {'legend.fontsize': 10,
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size*0.8,
          'ytick.labelsize': size*0.8,
          'axes.titlepad': size}
plt.rcParams.update(params)
visualiza_correlacao_media_interna(correl_media)
plt.savefig(directory+'Saidas/Cor_Med_Int_ini_NE.pdf')
plt.show()

# %%
# Aplica o 3º nível da reclusterização
clusters_finais = recluster_hierarquico_recursivo(df_usinas_NE, df_dados_norm, clusters_com_recluster)
correl_media = correlacao_media_interna_clusters(df_usinas_NE, df_dados_norm, clusters_finais)
avalia_correlacao_media_interna(correl_media)  # Confere se é vazio
n_clusters_finais=len(np.unique(clusters_finais))

# %%
# Plota correlação dcom a média interna depois o 3º nível
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
size=7
params = {'legend.fontsize': 10,
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size*0.8,
          'ytick.labelsize': size*0.8,
          'axes.titlepad': size}
plt.rcParams.update(params)
visualiza_correlacao_media_interna(correl_media)
plt.savefig(directory+'Saidas/Cor_Med_Int_fim_NE.pdf')
plt.show()



# %%
# Plota correlação da clusterização de 3º nível
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
size=20
params = {'legend.fontsize': 10,
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size*0.6,
          'ytick.labelsize': size*0.6,
          'axes.titlepad': 25}
plt.rcParams.update(params)
visualiza_correlacao_usinas_clusters(corr_ventos, df_usinas_NE, clusters_finais)
plt.savefig(directory+'Saidas/Cor_clust_nivel3_NE.pdf', bbox_inches="tight")
plt.show()

# %%
# Plota clusterização de 3º nível
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
size=20
params = {'legend.fontsize': 10,
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size*0.8,
          'ytick.labelsize': size*0.8,
          'axes.titlepad': 25}
plt.rcParams.update(params)
visualiza_usinas_espacial_clusters(df_usinas_NE, clusters_finais, df_estados, "NE",True)
plt.savefig(directory+'Saidas/Clust_nivel3_NE.pdf')
plt.show()

