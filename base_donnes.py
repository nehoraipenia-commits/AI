# base_donnes.py

"""base_donnes_fr = [
    ["la", "voiture", "roule", "vite"],
    ["la", "voiture", "est", "petite", "par", "rapport", "au", "camion"],
    ["la", "voiture", "est", "un", "vehicule"],
    ["le", "camion", "est", "un", "vehicule"],
    ["les", "vehicules", "transportent", "des", "personnes"],
    ["les", "gens", "habitent", "dans", "une", "maison"]
]"""

# Ta base de données
base_donnes_fr = [
    ["la", "voiture", "roule", "vite"],
    ["la", "voiture", "est", "petite", "par", "rapport", "au", "camion"],
    ["la", "voiture", "est", "un", "vehicule"],
    ["le", "camion", "est", "un", "vehicule"],
    ["les", "vehicules", "transportent", "des", "personnes"],
    ["les", "gens", "habitent", "dans", "une", "maison"]
]

def chatbot_reponse(entree_utilisateur):
    # On transforme l'entrée en minuscules pour ne pas rater de mots
    entree = entree_utilisateur.lower()
    reponses = []

    # On parcourt chaque "phrase" de ta base
    for phrase in base_donnes_fr:
        # On vérifie si le mot de l'utilisateur est présent dans la phrase
        if entree in phrase:
            # On transforme la liste de mots en une phrase lisible (string)
            reponses.append(" ".join(phrase))

    # Gestion de la réponse
    if reponses:
        print("Chatbot : Voici ce que je sais à ce sujet :")
        for r in reponses:
            print(f"- {r.capitalize()}.")
    else:
        print("Chatbot : Désolé, je n'ai aucune information sur '" + entree + "'.")

# --- Boucle principale pour discuter ---
print("--- Bienvenue dans le chatbot (tape 'quitter' pour arrêter) ---")

while True:
    user_input = input("Vous : ")
    
    if user_input.lower() == "quitter":
        print("Chatbot : Au revoir !")
        break
    
    chatbot_reponse(user_input)