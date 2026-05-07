import answer_neho_AI_fr as neho_answer
from answer_neho_AI_fr import *
import Userinformations
from Userinformations import informations_USER
import random
from datetime import datetime
import json
import os


lst_sujet_fr = ["continuer", "avancer", "apprendre"]
choix_mot_fr = random.choice(lst_sujet_fr)


def saluer1_fr():
    if informations_USER["name"] is not None:
        saluer_fr(informations_USER["name"])

    else:
        informations_USER["name"] = "Inconnu"
        saluer_fr("Inconnu")


def au_revoir1_fr():
    if informations_USER["name"] is not None:
        au_revoir_fr(informations_USER["name"])

    else:
        informations_USER["name"] = "Inconnu"
        au_revoir_fr("Inconnu")

def sujet1_fr():
    sujet_fr(informations_USER["name"], informations_USER["sujet_prefere"], choix_mot_fr)
    
def dire_meteo1_fr():
    dire_meteo_fr()

def faire_blague1_fr():
    faire_blague_fr()

def calculer_addidtion1_fr():
    num1 = input("Que voulez vous additionner (nombre 1) > ")
    num2 = input("Que voylez vous additionner (numbre 2) > ")
    calculer_addidtion_fr(num1, num2)

def dataNumber(question):
    data_number[question]()
def proposer_choses_fr():
    proposer_choses_a_faire_fr(sujet_prefere)
    
    

def repondre_cv():
    repondre_comment_caVA_fr()

def say_hour():
    maintenant = datetime.now()
    heure_formattee = maintenant.strftime("%H:%M")
    dire_heure_fr(heure_formattee)

def extraire_et_sauver_nom(phrase, liste_mots):
    mots = phrase.lower().split()
    nom_trouve = None

    if "appelle" in mots:
        index = mots.index("appelle")
        if index + 1 < len(mots):
            nom_trouve = mots[index + 1]
    elif "suis" in mots:
        index = mots.index("suis")
        if index + 1 < len(mots):
            nom_trouve = mots[index + 1]

    if nom_trouve:

        informations_USER["name"] = nom_trouve.capitalize()
        print(f"IA : Enchanté {nom_trouve.capitalize()} ! C'est enregistré.")
        return True
    return False
    
def parler_de_sujet():
    texte_a_completer_fr(informations_USER["sujet_prefere"])

def dire_est_quoi():
    print("Vous etes un(e)", informations_USER["etre_qq_chose"]) 

def dire_quoi_chatbot():
    dire_je_suis_un_chatbot()

def arrete():
    se_taire_fr()


def apprendre_phrase():


    FICHIER_MEMOIRE = "memoire_ia.json"

    def charger_memoire():
    
        if os.path.exists(FICHIER_MEMOIRE):
            with open(FICHIER_MEMOIRE, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def sauvegarder_memoire(memoire):
    
        with open(FICHIER_MEMOIRE, "w", encoding="utf-8") as f:
            json.dump(memoire, f, indent=4, ensure_ascii=False)
    
    
    