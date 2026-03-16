from oop_basico import Message


print("=" * 7)
print("CREANDO INSTANCIAS DE MESSAGE")
print("=" * 70)


msg1 = Message("Buy now! Click here for free money!!!", "spammer@evil.com")
msg2 = Message("Buy now! Click here for free money!!", "johnpork@evil.com")

print(f"\nMensaje 1 {msg1.text}\n")
print(f"sender {msg1.sender}")

print(f"cantidad de palabras: {msg1.get_word_count()}")

print(f"\ncontiene la clave: {msg1.contains_keyword("batman")}")

print(f"\nLongitud del texto {msg1.text_length()}")

print(f"\n objeto: {msg1}")


#LISTAS

messages = [msg1,msg2]

msg3 = Message("hoy estamos tomorrow we dont know", "stwardsegura@gamil.com")

messages.append(msg3)

print(f"\ntotal de mensages: {len(messages)}")
print(f"\ntodos los mensajes")

for msg in messages:
    print(f" 1. {msg}")


#list comprehension

word_counts = [msg.get_word_count() for msg in messages]
print(f"\n Palabras por mensaje (list comprehension): {word_counts}")


long_messages = [msg for msg in messages if msg.text_length() > 5]

print(f"\n Mensajes con mas de 5 palabras: {long_messages}")

print(f"\n Mensajes con mas de 5 palabras tuning {[str(m) for m in long_messages]}")




















