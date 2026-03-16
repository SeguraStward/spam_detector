


class User: 

    def __init__(self, name:str, email:str,spam_messages_received:int):

        self.name = name
        self.email = email
        self.spam_messages_received = spam_messages_received


    def add_spam(self, spam):
        self.spam_messages_received += spam



    def __str__(self) ->str:

        return f"USER(name={self.name}, emails= {self.spam_messages_received})"



user = User("lucas","lucar@gmail.com",5)
user2 = User("batman", "bruce@gmail.com", 34)


user.add_spam(1)
user2.add_spam(1)


print(f" user1: {user} \n user2: {user2}")
