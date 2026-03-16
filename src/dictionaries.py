from oop_basico import Message
from main import messages,msg1,msg2,msg3  

print("\n\n dictionariesssss")
sender_messages = {}

for msg in messages:
    if msg.sender not in sender_messages:
        sender_messages[msg.sender] = msg.text


for sender, msgs in sender_messages.items():
    print(f" {sender} : {msgs} mensajes")


    
    confidence_scores = {

        "lucaslee@gmail.com": 0.89,
        "batman@gmail.com":0.43,
        "ryah@gmail.com":0.54

    }
    

for email, score in confidence_scores.items():
    print(f"{email} score: {score}")
