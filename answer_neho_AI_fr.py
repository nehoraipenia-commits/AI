import random
import lst_codes
from lst_codes import codes

def saluer_fr(nom):
    emoji = ["😄", "😄", "😀"]
    emoji_choix = random.choice(emoji)
    salutation_fr2 = f"""Bonjour {nom}, Comment ça va aujourd'hui ?

Je suis ravi de discuter avec toi. Dis-moi, qu'est-ce qui t'amène ?
Que ce soit pour un coup de main sur un projet, une question qui te
trotte dans la tête ou juste pour échanger deux mots, je suis là {emoji_choix} !"""
    print(salutation_fr2)

def au_revoir_fr(nom):
    au_revoir_fr2 = f"Au revoir {nom} !"
    print(au_revoir_fr2)

def sujet_fr(nom, sujet, mot):
    sujet_fr2 = f"Tu veux faire quoi {nom} ? tu veux peut-etre {mot} avec {sujet}, ou autre chose ?"
    print(sujet_fr2)

#def dire_meteo_c_fr(degres, vent, temps):
  #  dire_meteo_c_fr2 = f"Il fait en ce moment {degres} c, le vent en vers le {vent}, le temps est {temps}"
   # print(dire_meteo_c_fr2)

def dire_meteo_fr():
    print("Desole, je ne peux pas encore vous dire la meteo, mais cette fonctionnalite sera bientot ajoutee")

def faire_blague_fr():
    liste_blagues = ["Comment appel-t-on un chat qui vit dans l'espace ?"]
    blague_au_choix_fr = random.choice(liste_blagues)
    print(blague_au_choix_fr)

def calculer_addidtion_fr(num1, num2):
    parler_de_sujet_fr2 = f"En fait si tu fais {num1} + {num2} le resultat est {num1 + num2}.                                               veux tu faire un autre calcul ?"

def text_simple_reponse_fr(nom, sujet_prefere, data, salutation):
    sujet_fr23 = f"""Ok, je vois tu {data} faire {sujet_prefere}, si tu veux je peux t'aider a:                                             - avancer ton projet,                                                                                                                      - t'aider avec des nouvelles idees pour ton projet {sujet_prefere},                                                                        - ou t'aider avec autre chose                                                                                                                                                                                                                                                         dis le moi."""
    print(sujet_fr23)

def texte_a_completer_fr(sujet):
    text_fr = f"""Ok, on continue sur {sujet}, veux tu qu'on:                                                                               - fasse avancer ton projet                                                                                                                 - le publier                                                                                                                               dis le moi et on continue"""

def se_taire_fr():
    print("D'accord, je me tait.")

def proposer_choses_a_faire_fr(sujet_prefere):
    text3 = f"""Je peux te proposer plusieurs choses qu'on peut faire, voici quelques propositions:                                         - te faire une blague                                                                                                                      - avancer un projet                                                                                                                        - t'aider a faire des calculs                                                                                                              - t'aider avec {sujet_prefere}                                                                                                             - t'aider dans ton job ou te donner des conseils                                                                                                                                                                                                                                      dis moi et on continue."""
    print(text3)

def dire_heure_fr(heure):
    print(f"Il est {heure}, besoin de savoir la date du jour aussi ?")

def dire_date_fr(date):
    print(f"On est le {date}, besoin de savoir l'heure exacte ?")

def repondre_comment_caVA_fr():
    print("Je vais bien merci, et... a part ca, que voulez vous faire ?")

def dire_je_suis_un_chatbot():
    text = "Je suis un chatbot cree en 2026, par Nehoray penia"
    print(text)

def si_coder_en_python(code):
    
    if code in lst_codes.codes:
        text = f"""Non {choix}, je ne peux pas coder en {code}, 
mais je peux comprendre
quelques script"""
    else:
        text = "Desole, je ne reconnais pas ce code."
    lst_emoji = ["😅", ""]
    choix = random.choice(lst_emoji)
    print(text)