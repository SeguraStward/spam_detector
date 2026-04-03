# Spam Detector

Clasificador de mensajes SMS que detecta spam usando Machine Learning. El usuario puede elegir entre Naive Bayes y Regresion Logistica.

## Tecnologias

- Python 3.11+
- scikit-learn
- pandas
- numpy

## Dataset

Enron-Spam Dataset — ~33,700 correos electronicos reales etiquetados (spam / ham)

```bash
python src/download_data.py
```

## Instalacion

```bash
pip install -r requirements.txt
```

## Uso

```bash
# Clasificar un mensaje
python src/main.py --algorithm naive_bayes --message "WIN a free iPhone now!!!"
python src/main.py --algorithm logistic    --message "Hey, are you coming tonight?"

# Evaluar modelo sobre el test set
python src/main.py --algorithm naive_bayes --evaluate
python src/main.py --algorithm logistic    --evaluate
```

## Estructura

```
spam_detection/
├── data/
│   └── SMSSpamCollection.tsv
└── src/
    ├── main.py
    ├── preprocessor.py
    ├── vectorizer.py
    ├── naive_bayes.py
    ├── logistic_regression.py
    ├── evaluator.py
    └── download_data.py
```
