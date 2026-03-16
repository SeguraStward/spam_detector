from main import messages



print("\n=========================================================")
print("\n SETS")

print("\n==========================================================")


all_words = set()


for msg in messages:
    words = msg.text.lower().split()
    all_words.update(words)


print(f"\npalabras unicas en todos los mensajes: {len(all_words)}")

print(f"primeras 10 palabras: {sorted(list(all_words))[:10]}")



spam_words = set("buy click offer limited free".split())
non_spam_words = set("hey how tomorrow catch".split())


#operaciones con sets
print("\n Set operations " + "==" * 30 )
print(f"Intersection: {spam_words & non_spam_words}")
print(f"Union: {spam_words | non_spam_words }")
print(f"diferencia: {spam_words - non_spam_words}")


def calculate_spam_score(text:str) -> float:

    spam_indicator = "buy click now free offer limited".split()

    text_lower = text.lower()
    score = 0.0

    for indicator in spam_indicator:
        if indicator in text_lower:
            score += 0.15 

    return min(score,1.0)





for msg in messages:
    score = calculate_spam_score(msg.text)
    print(f"{msg.sender}: {score} (spam_probability)")
