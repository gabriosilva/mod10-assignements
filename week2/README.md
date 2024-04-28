# Pipeline de Processamento de Imagens

Este projeto é um pipeline de processamento de imagens, projetado para automatizar o processamento de imagens através de uma série de etapas, incluindo redimensionamento, normalização, desfoque Gaussiano, limiarização adaptativa média, inversão e detecção de bordas de Canny. O pipeline também suporta aumento de dados como rotação e espelhamento.

## Primeiros Passos

Para executar este projeto, você precisa ter o Python instalado no seu sistema, juntamente com as bibliotecas necessárias listadas em `requirements.txt`.

### Pré-requisitos

- Python 3
- OpenCV
- NumPy
- pandas

Você pode instalar todas as bibliotecas necessárias usando o seguinte comando: `pip install -r requirements.txt`


### Executando o Pipeline

1. Coloque suas imagens na pasta `pictures`.
2. Execute o script principal: `python main.py`


### Resultados

As imagens processadas serão salvas no diretório `out`, e um arquivo CSV chamado `image_dataframe.csv` contendo as informações de processamento será gerado no diretório raiz do projeto [Dataframe]

Para ver os resultados, basta navegar até a pasta `out` ou abrir o arquivo `image_dataframe.csv`.