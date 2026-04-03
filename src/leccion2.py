import re
from collections import Counter


texto = "¡Hola! ¿Cómo estás? Este es un mensaje de prueba. ¡Muy bien!"


class TextProcessor:

    def __init__(self):
        self.vocabulary = set()
        self.word_frecuencies ={}


    def lowercase(self, text:str)->str:
        return text.lower()

    def remove_punctuation(self, text:str)->str:
        return re.sub(r'[^\w\s-]', '', text)

    def remove_extra_whitespaces(self,text:str)->str:
        return ' '.join(text.split())


    def tokenize(self,text:str)->list:
        text = self.lowercase(text)
        text = self.remove_punctuation(text)
        text = self.remove_extra_whitespaces(text)
        tokenes = text.split()
        return tokenes

    def process(self,text:str)->list:
        return self.tokenize(text)


processor = TextProcessor()

textos_ejemplo = [
    "¡Hola! ¿Cómo estás?",
    "Este es un mensaje...   con espacios múltiples!!!",
    "COMPRA AHORA!!! Es URGENTE!!!",
]

print("\nPipeline de limpieza:")

for texto in textos_ejemplo:
    tokens = processor.process(texto)
    print(f" Procesado {tokens}")





STOPWORDS_ES = {
    'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'ser', 'se',
    'no', 'haber', 'por', 'con', 'su', 'para', 'es', 'o', 'como',
    'este', 'está', 'están', 'he', 'has', 'han', 'soy', 'eres',
    'somos', 'sois', 'estoy', 'estás', 'estamos', 'estáis',
}

class TextProcessorStopWords(TextProcessor):

    def __init__(self,stopwords:set = None):
        super().__init__()
        self.stopwords = stopwords or STOPWORDS_ES

    def remove_stopwords(self, tokens:list)-> list:
        return [word for word in tokens if word not in self.stopwords]



    def process(self, text:str) -> list:
        tokens = super().tokenize(text)
        tokens = self.remove_stopwords(tokens)
        return tokens


processor_advanced = TextProcessorStopWords()

texto_largo = "El mensaje es muy importante. Este es un spam que está diseñado para engañar"
 
print(f"\nTexto: {texto_largo}")
 
tokens_sin_stopwords = processor.process(texto_largo)
print(f"\nSin remover stopwords ({len(tokens_sin_stopwords)} palabras):")
print(tokens_sin_stopwords)
 
tokens_con_stopwords = processor_advanced.process(texto_largo)
print(f"\nRemoviendo stopwords ({len(tokens_con_stopwords)} palabras):")
print(tokens_con_stopwords)




class TextProcessorCompleto(TextProcessorStopWords):


    def __init__(self, stopwords: set = None):
        super().__init__(stopwords)
        self.word_frecuencies = {}
        self.vocabulary = set()


    def calculate_frequencies(self,tokens: list) -> dict:


        frecuency_dict = {}

        for token in tokens:
            if token not in frecuency_dict:
                frecuency_dict[token] = 0
            frecuency_dict[token] += 1

        return frecuency_dict



    def process_with_analysis(self, text: str) -> dict:

        tokens = self.process(text)
        frequencies = self.calculate_frequencies(tokens)

        self.vocabulary.update(frequencies.keys())

        return {
            'tokens':tokens ,
            'frequencies': frequencies,
            'vocab_size': len(frequencies),
            'total_words': len(tokens),
        }

processor_completo = TextProcessorCompleto()
 
textos_spam = [
    "Compra ahora! Oferta limitada. ¡Haz clic!",
    "Gana dinero rápido. Dinero fácil. ¡Compra hoy!",
]
 
print("\nAnálisis de textos spam:")
for i, texto in enumerate(textos_spam, 1):
    print(f"\n--- Texto {i} ---")
    print(f"Original: {texto}")
    
    analisis = processor_completo.process_with_analysis(texto)
    
    print(f"Tokens: {analisis['tokens']}")
    print(f"Frecuencias: {analisis['frequencies']}")
    print(f"Palabras únicas: {analisis['vocab_size']}")
    print(f"Total de palabras: {analisis['total_words']}")
 
# Mostrar vocabulario global
print(f"\nVocabulario global (todas las palabras vistas): {processor_completo.vocabulary}")





class BagOfWordsProcessor:
    def __init__(self, stopwords:set = None):
        self.processor = TextProcessorCompleto(stopwords)
        self.vocabulary_list = []
        self.document_frequencies = []

    def build_vocabulary(self, documents:list) -> list:

        for doc in documents:
            self.processor.process_with_analysis(doc)

        self.vocabulary_list = sorted(list(self.processor.vocabulary))
        return self.vocabulary_list



    def vectorize_document(self, text:str)-> dict:
        analisis = self.processor.process_with_analysis(text)
        frequencies = analisis['frequencies']

        vector = {word: frequencies.get(word,0) for word in self.vocabulary_list}
        
        return vector


    def vectorize_documents(self, documents:list)->list:
        return [self.vectorize_document(doc) for doc in documents]

    def matrix_view(self,documents:list) -> None:
        self.build_vocabulary(documents)

        vocab_display = self.vocabulary_list[:10]


        print(f"\n {'Documento': <30}", end='')
        for word in vocab_display:
            print(f"{word[:6]:<8}", end="")

        print("\n" + "-" * (30 + 8*10))

        for i, doc in enumerate(documents):
            vector = self.vectorize_document(doc)
            print(f"Doc {i+1}: {doc[:25]:<25}", end='')

            for word in vocab_display:
                count = vector.get(word, 0)
                print(f"{count:<8}", end='')
            print()



bow = BagOfWordsProcessor()
 
mensajes = [
    "Compra Compra ahora dinero! Oferta compra  limitada.",
    "¿Cómo como compra como estás? oferta Te amo.",
    "Gana dinero amo amo amo amo . Dinero fácil facil.",
]
 
print("\nMatriz de Bag of Words:")
bow.matrix_view(mensajes)
 
# Vectorizar específicamente
print("\n\nVectorización del primer documento:")
vector = bow.vectorize_document(mensajes[0])
print(f"Documento: {mensajes[0]}")
print(f"Vector (primeras 10 palabras):")
vocab_display = sorted(bow.vocabulary_list)[:10]
for word in vocab_display:
    count = vector.get(word, 0)
    if count > 0:
        print(f"  {word}: {count}")

 
def ejercicio2(documents:list) -> Counter:

    processor =  TextProcessorCompleto(STOPWORDS_ES)
    
    words = []
    for doc in documents:
        doc = processor.process(doc)
        words.extend(doc)

    return Counter(words)


print("\n Mi ejerciciosss sojfsljflsjfskljfslkjf")

print(ejercicio2(mensajes))

