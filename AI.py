import memoire as mmr
from memoire import data, data_number 
import convertisseur as cv
from convertisseur import *
import random
from Userinformations import informations_USER as infos
from Dictionary import mots_francais as mf 
import main
from main import discussion
import time


dire_est_quoi()
history = []
infos_etre = [] 


def TransformToNum(words):
    listeWord = []
    words2 = words.lower().split() 
    for word in words2:
        if word in data: 
            listeWord.append(data[word])
        else:
            listeWord.append(0)
    resultat = sum(listeWord)
    print(f"Phrase : '{words}' -> Score total : {resultat}")
    return resultat


print("Quelques informations (facultatif) :")
information_name = input("Votre nom : ")
information_sujet_prefere = input("Sujet préféré : ")
information_sauvgarder_discussion = input("Voulez vous sauvgarder cette discussion : ")

if information_name: 
    infos["name"] = information_name
if information_sujet_prefere: 
    infos["sujet_prefere"] = information_sujet_prefere

def sauvgarder_discussion():
    if information_sauvgarder_discussion == "yes" or "oui":
    
        with open("historique.txt", "w", encoding='utf8') as f:
            f.write("\nvous : ")
            f.write(question)
            f.close()
    else:
        print()

while True:
    question = input("\nquestion > ").lower()
    sauvgarder_discussion()
    history.append(question)

    time.sleep(0.5)

    if not question.strip():
        continue
    
    if question.startswith("je suis un"):
        partie = question.split("je suis un ")
        if len(partie) > 1:
            quoi = partie[1]
            infos_etre.append(quoi)
            informations_USER["etre_qq_chose"] = quoi
            print(f"IA : C'est noté, vous êtes un {quoi}.")
            continue 
    if question.startswith("je suis une"):
        partie2 = question.split("je suis une ")
        if len(partie2) > 1:
            quoi2 = partie2[1]
            informations_USER["etre_qq_chose"] = quoi2
            print(f"IA : C'est noté, vous êtes une {quoi2}.")
            continue 
    
    if "je suis" in question or "je appelle" or "je m\'appel" in question:
        if extraire_et_sauver_nom(question, mf): 
            continue 
        
    #if question.startswith("apprendre"):
       # q = input("Voules vous ouvrir le centre d'apprentissage ? : ")
        #if q.startswith("oui"):
            
    score = TransformToNum(question)
        
    if score in data_number:
        data_number[score]() 
    else:
        reponses_erreur = [
            "AI : Je ne reconnais pas ces mots.",
            "AI : Peux-tu reformuler ?",
            "AI : Désolé, je n'ai pas compris."
        ]
        print(random.choice(reponses_erreur))