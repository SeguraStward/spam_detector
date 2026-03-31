"""
LECCIÓN 3: NAIVE BAYES CLASSIFIER
==================================
 
En esta lección aprenderás:
1. Probabilidades bayesianas básicas
2. Cómo funciona Naive Bayes
3. Entrenamiento del modelo
4. Predicción y confianza
5. Evaluación del clasificador
 
Ejecuta: python leccion_3_naive_bayes.py
"""
import re
import math
from collections import Counter
  
""" 
Teorema de bayes: P(A|B) = P(B|A) * P(A) / P(B)
  P(spam | palabras) = P(palabras | spam) * P(spam) / P(palabras)

  En detector de spam se hace esta pregunta: ¿Cuál es la probabilidad de que un mensaje sea spam dado las palabras que contiene? 

  ejemplo: dataser con 10 mensajes de spam y 90 de no spam

"""
print("\n" + "=" * 70)
print("PARTE 2: IMPLEMENTAR NAIVE BAYES")
print("=" * 70)
 
class NaiveBayesClassifier:
        """
    Clasificador Naive Bayes para spam detection.
    
    CONCEPTOS CLAVE:
    - self.class_counts: cuántos documentos por clase
    - self.word_counts: frecuencia de palabra en cada clase
    - self.class_probabilities: P(clase)
    - self.word_probabilities: P(palabra | clase)
    """
        def __init__(self, alpha: float = 1.0):
            """alpha para evitar que P(palabra | spam) sea 0"""
            
            self.alpha = alpha
            self.class_counts = {}
            self.word_counts = {}
            self.class_probabilities = {}
            self.vocabulary = set()
            self.total_documents = 0
        
        def tokenize(self, text: str) -> list:
         
              text = text.lower()
              text = re.sub(r'[^a-z0-9áéíóúñ ]','', text)
              tokens = text.split()
              return tokens
        
        def train(self, documents: list, labels: list)-> None:
              
              self.total_documents = len(documents)
              self.class_counts = {}
              self.word_counts = {}
              self.vocabulary = set()

              for label in labels:
                    self.class_counts[label] = labels.count(label)
                    self.word_counts[label] = {}        
              
              for label, count in self.class_counts.items():
                    print(f"  {label}: {count} ({count/self.total_documents:.1%})")

              ## Para cada documento, contar palabras por clase
              for doc, label in zip(documents,labels):
                    tokens = self.tokenize(doc)

                    ##contar cada palabra en ese documento
                    for word in tokens:
                          self.vocabulary.add(word)
                          
                          if word not in self.word_counts[label]:
                                self.word_counts[label][word] = 0
                        
                          self.word_counts[label][word] += 1

                    # calcular probabilidades de clase P(clase)
                    self.class_probabilities = {
                          label: count / self.total_documents
                          for label, count in self.class_counts.items()
                    }
                    
              print(f"Vocabulario unico: {len(self.vocabulary)} palabras")
              print(f"priori probabilidades P(Clase): ")
              for label, prob in self.class_probabilities.items():
                    print(f"  P({label}) = {prob:.4f}")
                    
              print("\n Model entrenadoo ")

        def predict(self, text: str) -> dict:
              """
              Predice el texto entrante 
                
                Prediccion= spam o no spam
                Confianza= probabilidad de esa prediccion
                probabilidad = 0.60 spam
                detalles
              
              """
              tokens = self.tokenize(text)
              class_scores = {} #Calcular P(clase | palabras) Probabilidad de clase dado que o sabiendo las palabras
              for label in self.class_counts.keys():
                    
                    ##empezar con P(clase) probabilidad de que un mensaje sea spam o no spam
                    score = math.log(self.class_probabilities[label])

                    ## Para cada palabra, multiplicar P(palabra | clase)
                    ##usamos log para evitar underflow numerico. WHAT?

                    for word in tokens:
                          if word in self.word_counts[label]:
                                #frecuencia de esta palabra en esta clase
                                word_count = self.word_counts[label][word]
                          else:
                                word_count = 0
                            
                          #contar total de palabras en esta clase
                          total_words = sum(self.word_counts[label].values())

                          #calcular P(palabra | clase) con smoothing
                          # word_count * alpha / (total_words + alpha * len(self.vocabulary))
                          word_probability = (word_count + self.alpha) / (total_words + self.alpha * len(self.vocabulary))

                          score += math.log(word_probability)

                          class_scores[label] = score
                    
                #Convertir log probabilidades en probabilidades normales
                #usar sofmax para normalizacion

              max_score = max(class_scores.values())
              normalized_scores = {
                    label: math.exp(score - max_score)
                    for label, score in class_scores.items()
              }

              total = sum(normalized_scores.values())
              probabilities = {
                    label: score / total
                    for label, score in normalized_scores.items()
              }

              ## Prediccion 
              prediction = max(probabilities, key = probabilities.get)
              confidence = probabilities[prediction]

              return {
                    'prediction': prediction,
                    'confidence': confidence,
                    'probabilities': probabilities,
                    'tokens': tokens,
                    'tokens_count': len(tokens)
              }
        
        def predict_batch(self, texts: list)-> list:
              """Predice multiples textos"""
              return [self.predict(text) for text in texts]
        



# ============================================================================
# PARTE 3: EJEMPLO PRÁCTICO - DATOS DE ENTRENAMIENTO
# ============================================================================
 
print("\n" + "=" * 70)
print("PARTE 3: ENTRENAR CON DATOS REALES")
print("=" * 70)
 
# Dataset de entrenamiento
spam_messages = [
    "Compra ahora! Oferta limitada",
    "Gana dinero rápido y fácil",
    "Haz clic aquí para ganar premios",
    "Dinero fácil, dinero rápido, compra ya",
    "Oferta exclusiva solo para ti, compra ahora",
    "¡¡¡Dinero dinero dinero!!! Gana hoy",
    "Clica para ganar lotería",
    "Compra productos baratos baratos",
]
 
not_spam_messages = [
    "Hola, ¿cómo estás?",
    "Te amo, nos vemos mañana",
    "La reunión es a las 3pm",
    "¿Qué tal tu día?",
    "Nos vemos en el café",
    "Buenos días, ¿dormiste bien?",
    "El proyecto está terminado",
    "Gracias por tu ayuda",
    "¿Vienes al cine esta noche?",
    "Perfecto, nos hablamos luego",
]
 
# Combinar y etiquetar
all_messages = spam_messages + not_spam_messages
all_labels = ['spam'] * len(spam_messages) + ['no_spam'] * len(not_spam_messages)
 
print(f"Dataset total: {len(all_messages)} mensajes")
print(f"  Spam: {len(spam_messages)}")
print(f"  No-spam: {len(not_spam_messages)}")
 
# Entrenar
classifier = NaiveBayesClassifier(alpha=1.0)
classifier.train(all_messages, all_labels)
 