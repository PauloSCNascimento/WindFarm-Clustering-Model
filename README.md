# Descrição
Esse modelo de clusterização de usinas eólicas foi desenvolvido para atendimento dos modelos de previsão do ONS. Os métodos utilizados neste modelo foram operacionalizados através da programação utilizando a linguagem livre Python e dados reais da região Nordeste e Sul do Brasil de posse do ONS.
Resumidamente, as principais etapas desse modelo são: (i) Aplicação do método GMM (Gaussian Mixture Models) juntamente com o K-means para clusterização de parques eólicos; (ii) Utilização de dados reais dos parques para aplicação do método, fornecendo insumos aos modelos preditivos.

# Documentação
O modelo permite a obtenção de uma clusterização de usinas eólicas com a utilização do método (GMM) e de modelos hierárquicos, baseado no perfil de ventos e localização geográfica de parques eólicos. O intuito do agrupamento gerado é fornecer insumos para os modelos de previsão de geração eólica. O sucesso dessa aplicação permite a criação de séries de geração e vento de forma mais assertiva dentro de cada cluster.

__Parte 01: Descrição teórica__ 
* Um vídeo com a descrição teórica do modelo pode ser encontrado no link:
https://www.youtube.com/
* A apresentação em pdf do vídeo é encontrada na raiz do diretório “Wind-Clustering-Model”.

**Parte 02: Descrição dos algoritmos**
* Todos os códigos relevantes ao modelo estão contidos no seguinte diretório: ~\Codigos.
* A execução do código “Modelo_Clust_Eolicas.py” já captura automaticamente o diretório raiz após o download”, desde que mantida a estrutura de pasta do modelo.
* Será necessário realizar o download de todas as pastas e seus arquivos.
* Será necessário instalar algumas bibliotecas para visualização dos gráficos geográficos.
* Pode ser necessário instalar algumas ferramentas de Build do Visual Studio.

**Links necessários**
* Ferramentas de Build do Visual Studio:
https://visualstudio.microsoft.com/pt-br/visual-cpp-build-tools/
* Visualização geográfica:
https://github.com/arthursgonzaga/DataProject/tree/main/Ep.01%20-%20Brazil%20Work%20Accidents/GeoVisualization/Dados%20SHX
