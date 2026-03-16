class Message:
    # Constructor (se ejecuta cuando creas un objeto)
    def __init__(self, text: str, sender: str):
        self.text = text              # Atributo de instancia
        self.sender = sender          # Atributo de instancia
        self.is_spam = None           # Será asignado después

    def __str__(self) -> str:
        return f"Message from {self.sender}: {self.text[:30]}..."

    
    def __repr__(self)->str:
        return f"Message(text='{self.text}', sender='{self.sender}')"

    def text_length(self)->int:
        return len(self.text)

    def get_word_count(self)->int:
        return len(self.text.split())

    def contains_keyword(self, keyword:str)->bool:
        """
            Verifica si el mensaje contiene una palabra clave.

            Conceptos:
            - .lower(): convierte strings en lowercase
            - 'in' : operador de busqueda en strings (como find() en c++)
        """

        return keyword.lower() in self.text.lower()


