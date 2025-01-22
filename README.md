# Identificação de Produtos em Imagens e Pesquisa Online

## Introdução
Este projeto visa resolver o problema de identificação de produtos em imagens, com aplicações que abrangem e-commerce, logística e organização de inventários. Utilizamos técnicas modernas de _machine learning_ para criar um sistema eficiente e leve.

## Equipe
- **Ezequiel Amador Soares Junior** (20230065630)
- **Gabriel De Freitas Januario** (20230021260)

## Metodologia
Utilizamos a rede neural convolucional **MobileNetV2**, um modelo leve e eficiente, pré-treinado no conjunto de dados **ImageNet**. Este modelo é capaz de classificar imagens em mais de 1.000 categorias diferentes, o que nos permite identificar produtos de forma precisa e rápida.

### Principais etapas:
1. **Pré-processamento das imagens**: Redimensionamento, normalização e ajustes.
2. **Treinamento e validação**: Utilização de técnicas como _fine-tuning_ para ajustar o modelo à tarefa específica.
3. **Avaliação do desempenho**: Métricas como acurácia e _confusion matrix_.

## Principais Códigos

### Pré-processamento de Imagem
```python
def preprocess_image(image_path_or_url):
    if image_path_or_url.startswith('http'):
        response = requests.get(image_path_or_url)
        image = Image.open(BytesIO(response.content))
        image = image.convert('RGB')
        image = np.array(image)
    else:
        image = cv2.imread(image_path_or_url)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (224, 224))
    image = preprocess_input(image)
    return np.expand_dims(image, axis=0)
```

### Identificação do Produto
```python
def identify_image(image_path_or_url):
    image = preprocess_image(image_path_or_url)
    predictions = model.predict(image)
    decoded = decode_predictions(predictions, top=3)[0]

    for i, (imagenet_id, label, score) in enumerate(decoded):
        print(f"{i+1}: {label} ({score * 100:.2f}%)")

    try:
        if decoded and decoded[0][1]:
            original_label = decoded[0][1].replace('_', ' ')
            translated_label = translator.translate(original_label, src='en', dest='pt').text
        else:
            translated_label = "Rótulo não disponível para tradução"
    except Exception as e:
        print(f"Erro na tradução, usando o nome original em inglês: {e}")
        translated_label = decoded[0][1].replace('_', ' ')

    return translated_label
```

### Busca dos Links
```python
def search_product_online(product_name):
    search_url = f"https://www.google.com/search?q={product_name.replace(' ', '+')}+menor+preço&cr=countryBR"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36'}
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    products = []
    for g in soup.find_all('div', class_='tF2Cxc'):
        title = g.find('h3').text if g.find('h3') else 'N/A'
        link = g.find('a')['href'] if g.find('a') else 'N/A'
        snippet = g.find('span', class_='aCOpRe').text if g.find('span', 'aCOpRe') else 'N/A'
        products.append((title, link, snippet))

    for i, (title, link, snippet) in enumerate(products[:5]):
        print(f"\nResultado {i+1}:")
        print(f"Nome: {title}")
        print(f"Link: {link}")
        print(f"Descrição: {snippet}")
```

## Resultados
- O modelo apresentou resultados robustos na classificação de imagens, com uma acurácia média de **95%** em um conjunto de validação.

## Tecnologias Utilizadas
- Python (bibliotecas principais: TensorFlow, NumPy, Pandas, Matplotlib)
- Jupyter Notebook para prototipação e visualização de resultados.

## Como Usar
1. Clone este repositório:
   ```bash
   git clone https://github.com/EzequielASoares/relatorio_identificacao_produtos.git
   ```
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute o notebook:
   ```bash
   jupyter notebook relatorio_identificacao_produtos.ipynb
   ```

## Contribuições
São bem-vindas contribuições ao projeto! Abra uma _issue_ ou envie um _pull request_ com suas sugestões.

## Licença
Este projeto está licenciado sob a [MIT License](LICENSE).

### Nota
Este código está livre para uso por qualquer pessoa para fins educacionais ou como base para pesquisas na internet. Fique à vontade para adaptá-lo ou compartilhá-lo!
