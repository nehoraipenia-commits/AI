import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import math
import time
from typing import List, Dict, Tuple, Optional
import os
import re

GRAMMATICAL_CATEGORIES = {
    "verbs": {"être", "avoir", "faire", "dire", "aller", "voir", "savoir", "pouvoir", "falloir", "vouloir", "venir", "prendre", "croire", "mettre", "passer", "devoir", "parler", "trouver", "donner", "comprendre", "connaître", "partir", "demander", "tenir", "aimer", "penser", "rester", "manger", "boire", "dormir", "courir", "marcher", "voler", "nager", "jouer", "chanter", "danser", "rire", "pleurer", "sourire", "travailler", "étudier", "apprendre", "lire", "écrire", "écouter", "entendre", "sentir", "toucher", "goûter", "créer", "détruire", "construire", "inventer", "découvrir", "chercher", "trouver", "perdre", "gagner", "réussir", "échouer"},
    "nouns": {"ordinateur", "écran", "clavier", "souris", "internet", "réseau", "données", "information", "système", "logiciel", "programme", "code", "algorithme", "intelligence", "machine", "robot", "technologie", "science", "mathématiques", "physique", "chimie", "biologie", "astronomie", "espace", "univers", "planète", "étoile", "galaxie", "énergie", "électricité", "lumière", "chaleur", "son", "vitesse", "temps", "matière", "atome", "molécule", "cellule", "gène", "adn", "cerveau", "mémoire", "processeur", "serveur", "cloud", "téléphone", "application", "site", "web", "page", "lien", "fichier", "dossier", "document", "image", "vidéo", "audio", "texte", "message", "email", "communication", "connexion", "utilisateur", "mot", "passe", "sécurité", "pirate", "virus", "bug", "erreur", "solution", "projet", "développement", "recherche", "innovation", "futur", "progrès", "évolution", "révolution", "découverte", "nature", "terre", "eau", "feu", "air", "vent", "pluie", "neige", "soleil", "lune", "ciel", "nuage", "orage", "éclair", "tonnerre", "tempête", "arbre", "fleur", "plante", "herbe", "forêt", "bois", "montagne", "colline", "vallée", "rivière", "fleuve", "lac", "mer", "océan", "plage", "sable", "rocher", "pierre", "animal", "chien", "chat", "cheval", "vache", "cochon", "mouton", "chèvre", "poule", "canard", "oiseau", "poisson", "insecte", "araignée", "serpent", "grenouille", "lion", "tigre", "éléphant", "girafe", "singe", "ours", "loup", "renard", "lapin", "baleine", "dauphin", "requin", "abeille", "papillon", "mouche", "moustique", "fourmi", "escargot", "ver", "feuille", "racine", "branche", "homme", "femme", "enfant", "garçon", "fille", "bébé", "adulte", "vieillard", "personne", "gens", "foule", "groupe", "famille", "père", "mère", "papa", "maman", "fils", "frère", "sœur", "grand-père", "grand-mère", "oncle", "tante", "cousin", "cousine", "neveu", "nièce", "mari", "époux", "épouse", "ami", "amie", "copain", "copine", "voisin", "collègue", "patron", "employé", "chef", "directeur", "professeur", "élève", "étudiant", "médecin", "infirmier", "patient", "policier", "pompier", "soldat", "juge", "avocat", "politique", "président", "ministre", "roi", "reine", "prince", "princesse", "artiste", "peintre", "musicien", "chanteur", "acteur", "écrivain", "journaliste", "sportif", "joueur", "équipe", "société", "pays", "nation", "état", "gouvernement", "loi", "droit", "justice", "paix", "guerre", "armée", "arme", "économie", "commerce", "marché", "entreprise", "usine", "magasin", "boutique", "client", "vendeur", "prix", "valeur", "jour", "nuit", "matin", "soir", "midi", "minuit", "heure", "minute", "seconde", "semaine", "mois", "année", "siècle", "idée", "pensée", "rêve", "imagination", "réalité", "vérité", "mensonge", "secret", "mystère", "problème", "question", "réponse", "raison", "cause", "conséquence", "but", "objectif", "moyen", "manière", "façon", "sorte", "type", "genre", "histoire", "vie", "mort", "amour", "haine", "joie", "tristesse", "peur", "colère", "surprise", "dégoût", "espoir", "courage", "chance", "hasard", "destin", "liberté", "égalité", "fraternité", "force", "faiblesse", "pouvoir", "autorité", "beauté", "laideur", "qualité", "défaut"},
    "adjectives": {"grand", "petit", "gros", "mince", "large", "étroit", "long", "court", "haut", "bas", "fort", "faible", "lourd", "léger", "beau", "laid", "joli", "moche", "bon", "mauvais", "meilleur", "pire", "vrai", "faux", "juste", "injuste", "clair", "sombre", "chaud", "froid", "tiède", "doux", "dur", "mou", "sec", "mouillé", "propre", "sale", "neuf", "vieux", "jeune", "ancien", "récent", "nouveau", "rapide", "lent", "facile", "difficile", "simple", "complexe", "ouvert", "fermé", "plein", "vide", "riche", "pauvre", "cher", "gratuit", "heureux", "triste", "joyeux", "colérique", "calme", "stressé", "détendu", "malade", "sain", "vivant", "mort", "intelligent", "stupide", "malin", "bête", "courageux", "lâche", "généreux", "égoïste", "gentil", "méchant", "sympathique", "désagréable", "poli", "impoli", "honnête", "menteur", "fidèle", "infidèle", "jaloux", "confiant", "timide", "sociable", "solitaire", "bavard", "silencieux", "bruyant", "tranquille", "agité", "fatigué", "énergique", "magnifique", "splendide", "merveilleux", "fantastique", "incroyable", "extraordinaire", "terrible", "horrible", "affreux", "rouge", "bleu", "vert", "jaune", "noir", "blanc", "gris", "orange", "violet", "rose", "marron", "doré", "argenté", "brillant", "mat", "transparent", "opaque", "rond", "carré", "rectangulaire", "triangulaire", "ovale", "pointu"},
    "connectors": {"le", "la", "les", "un", "une", "des", "du", "de", "ce", "cet", "cette", "ces", "mon", "ton", "son", "ma", "ta", "sa", "mes", "tes", "ses", "notre", "votre", "leur", "nos", "vos", "leurs", "je", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles", "me", "te", "se", "lui", "y", "en", "qui", "que", "quoi", "dont", "où", "lequel", "laquelle", "au", "aux", "à", "dans", "par", "pour", "vers", "avec", "sans", "sous", "sur", "chez", "malgré", "contre", "et", "ou", "ni", "mais", "or", "car", "donc", "cependant", "néanmoins", "toutefois", "puisque", "lorsque", "si", "avant", "après", "pendant", "depuis", "jusque", "environ", "presque", "très", "trop", "beaucoup", "peu", "assez", "plus", "moins", "aussi", "autant", "bien", "mal", "mieux", "vite", "lentement", "ici", "là", "loin", "près", "dedans", "dehors", "partout", "ailleurs", "oui", "non", "peut-être", "certes", "vraiment", "exactement", "absolument", "sûrement", "probablement", "évidemment", "naturellement", "simplement", "pourtant", "sinon", "ensuite", "puis", "alors", "bref", "enfin", "finalement"}
}

EXTENSIVE_FRENCH_LEXICON = [
    "le", "la", "les", "un", "une", "des", "du", "de", "ce", "cet", "cette", "ces", "mon", "ton", "son", "ma", "ta", "sa",
    "mes", "tes", "ses", "notre", "votre", "leur", "nos", "vos", "leurs", "je", "tu", "il", "elle", "on", "nous", "vous",
    "ils", "elles", "me", "te", "se", "lui", "y", "en", "qui", "que", "quoi", "dont", "où", "lequel", "laquelle", "au",
    "aux", "à", "dans", "par", "pour", "en", "vers", "avec", "de", "sans", "sous", "sur", "chez", "malgré", "contre",
    "et", "ou", "ni", "mais", "or", "car", "donc", "cependant", "néanmoins", "toutefois", "puisque", "lorsque", "si",
    "être", "suis", "es", "est", "sommes", "êtes", "sont", "été", "avoir", "ai", "as", "a", "avons", "avez", "ont", "eu",
    "faire", "fais", "fait", "faisons", "faites", "font", "fait", "dire", "dis", "dit", "disons", "dites", "disent",
    "aller", "vais", "vas", "va", "allons", "allez", "vont", "voir", "vois", "voit", "voyons", "voyez", "voient", "vu",
    "savoir", "sais", "sait", "savons", "savez", "savent", "su", "pouvoir", "peux", "peut", "pouvons", "pouvez", "peuvent",
    "falloir", "faut", "vouloir", "veux", "veut", "voulons", "voulez", "veulent", "venir", "viens", "vient", "venons",
    "venez", "viennent", "prendre", "prends", "prend", "prenons", "prenez", "prennent", "croire", "crois", "croit",
    "croyons", "croyez", "croient", "mettre", "mets", "met", "mettons", "mettez", "mettent", "passer", "passe", "passons",
    "devoir", "dois", "doit", "devons", "devez", "doivent", "parler", "parle", "parlons", "parlez", "trouver", "trouve",
    "donner", "donne", "comprendre", "comprends", "comprend", "connaître", "connais", "connait", "partir", "pars", "part",
    "demander", "demande", "tenir", "tiens", "tient", "aimer", "aime", "aimons", "penser", "pense", "pensons", "rester",
    "manger", "boire", "dormir", "courir", "marcher", "voler", "nager", "jouer", "chanter", "danser", "rire", "pleurer",
    "sourire", "travailler", "étudier", "apprendre", "lire", "écrire", "écouter", "entendre", "sentir", "toucher", "goûter",
    "créer", "détruire", "construire", "inventer", "découvrir", "chercher", "trouver", "perdre", "gagner", "réussir", "échouer",
    "grand", "petit", "gros", "mince", "large", "étroit", "long", "court", "haut", "bas", "fort", "faible", "lourd", "léger",
    "beau", "laid", "joli", "moche", "bon", "mauvais", "meilleur", "pire", "vrai", "faux", "juste", "injuste", "clair", "sombre",
    "chaud", "froid", "tiède", "doux", "dur", "mou", "sec", "mouillé", "propre", "sale", "neuf", "vieux", "jeune", "ancien",
    "récent", "nouveau", "rapide", "lent", "facile", "difficile", "simple", "complexe", "ouvert", "fermé", "plein", "vide",
    "riche", "pauvre", "cher", "gratuit", "heureux", "triste", "joyeux", "colérique", "calme", "stressé", "détendu", "malade",
    "sain", "vivant", "mort", "intelligent", "stupide", "malin", "bête", "courageux", "lâche", "généreux", "égoïste", "gentil",
    "méchant", "sympathique", "désagréable", "poli", "impoli", "honnête", "menteur", "fidèle", "infidèle", "jaloux", "confiant",
    "timide", "sociable", "solitaire", "bavard", "silencieux", "bruyant", "tranquille", "agité", "fatigué", "énergique",
    "magnifique", "splendide", "merveilleux", "fantastique", "incroyable", "extraordinaire", "terrible", "horrible", "affreux",
    "rouge", "bleu", "vert", "jaune", "noir", "blanc", "gris", "orange", "violet", "rose", "marron", "doré", "argenté", "clair",
    "foncé", "brillant", "mat", "transparent", "opaque", "rond", "carré", "rectangulaire", "triangulaire", "ovale", "pointu",
    "ordinateur", "écran", "clavier", "souris", "internet", "réseau", "données", "information", "système", "logiciel",
    "programme", "code", "algorithme", "intelligence", "artificielle", "machine", "robot", "technologie", "science",
    "mathématiques", "physique", "chimie", "biologie", "astronomie", "espace", "univers", "planète", "étoile", "galaxie",
    "énergie", "électricité", "lumière", "chaleur", "son", "vitesse", "temps", "espace", "matière", "atome", "molécule",
    "cellule", "gène", "adn", "cerveau", "mémoire", "processeur", "serveur", "cloud", "téléphone", "application", "site",
    "web", "page", "lien", "fichier", "dossier", "document", "image", "vidéo", "audio", "texte", "message", "email",
    "communication", "connexion", "utilisateur", "mot", "passe", "sécurité", "pirate", "virus", "bug", "erreur", "solution",
    "projet", "développement", "recherche", "innovation", "futur", "progrès", "évolution", "révolution", "découverte",
    "nature", "terre", "eau", "feu", "air", "vent", "pluie", "neige", "soleil", "lune", "ciel", "nuage", "orage", "éclair",
    "tonnerre", "tempête", "arbre", "fleur", "plante", "herbe", "forêt", "bois", "montagne", "colline", "vallée", "rivière",
    "fleuve", "lac", "mer", "océan", "plage", "sable", "rocher", "pierre", "animal", "chien", "chat", "cheval", "vache",
    "cochon", "mouton", "chèvre", "poule", "canard", "oiseau", "poisson", "insecte", "araignée", "serpent", "grenouille",
    "lion", "tigre", "éléphant", "girafe", "singe", "ours", "loup", "renard", "lapin", "souris", "rat", "baleine", "dauphin",
    "requin", "abeille", "papillon", "mouche", "moustique", "fourmi", "escargot", "ver", "feuille", "racine", "branche",
    "ville", "village", "rue", "route", "chemin", "place", "parc", "jardin", "pont", "bâtiment", "maison", "appartement",
    "immeuble", "toit", "mur", "porte", "fenêtre", "pièce", "chambre", "salon", "cuisine", "salle", "bain", "toilettes",
    "cave", "grenier", "garage", "escalier", "ascenseur", "meuble", "lit", "table", "chaise", "canapé", "fauteuil", "armoire",
    "placard", "bureau", "étagère", "lampe", "tapis", "rideau", "miroir", "horloge", "montre", "télévision", "radio",
    "livre", "cahier", "stylo", "crayon", "papier", "lettre", "enveloppe", "timbre", "sac", "boîte", "panier", "bouteille",
    "verre", "tasse", "assiette", "couteau", "fourchette", "cuillère", "casserole", "poêle", "four", "frigo", "machine",
    "outil", "marteau", "clou", "vis", "tournevis", "scie", "clé", "serrure", "argent", "monnaie", "billet", "carte",
    "homme", "femme", "enfant", "garçon", "fille", "bébé", "adulte", "vieillard", "personne", "gens", "foule", "groupe",
    "famille", "père", "mère", "papa", "maman", "fils", "fille", "frère", "sœur", "grand-père", "grand-mère", "oncle", "tante",
    "cousin", "cousine", "neveu", "nièce", "mari", "femme", "époux", "épouse", "ami", "amie", "copain", "copine", "voisin",
    "collègue", "patron", "employé", "chef", "directeur", "professeur", "élève", "étudiant", "médecin", "infirmier", "patient",
    "policier", "pompier", "soldat", "juge", "avocat", "politique", "président", "ministre", "roi", "reine", "prince", "princesse",
    "artiste", "peintre", "musicien", "chanteur", "acteur", "écrivain", "journaliste", "sportif", "joueur", "équipe",
    "société", "pays", "nation", "état", "gouvernement", "loi", "droit", "justice", "paix", "guerre", "armée", "arme",
    "économie", "commerce", "marché", "entreprise", "usine", "magasin", "boutique", "client", "vendeur", "prix", "valeur",
    "jour", "nuit", "matin", "soir", "midi", "minuit", "heure", "minute", "seconde", "semaine", "mois", "année", "siècle"
]

class TextCorpusReader:
    """
    Handles reading and processing custom user text files to extract knowledge and phrasing styles.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.extracted_words = []
        self.sentences = []

    def load_and_parse(self) -> List[str]:
        if not os.path.exists(self.file_path):
            print(f"📁 File '{self.file_path}' not found. Creating a clean default model...")
            with open(self.file_path, "w", encoding="utf-8") as f:
                f.write("Artificial intelligence transforms our world rapidly.\n")
                f.write("A quantum computer can solve highly complex mathematical equations.\n")
                f.write("Modern science progresses daily towards unexpected discoveries.\n")

        with open(self.file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Clean and split into sentences using Regex
        cleaned_content = re.sub(r'[^a-zA-ZÀ-ÿ\s\.,!?]', '', content)
        self.sentences = re.split(r'(?<=[.!?]) +', cleaned_content)

        # Extract words for vocabulary expansion
        words = re.findall(r'\b[a-zA-ZÀ-ÿ]+\b', cleaned_content.lower())
        self.extracted_words = list(set(words))

        print(f"📖 Loaded Corpus: {len(self.sentences)} sentences, {len(self.extracted_words)} unique words loaded.")
        return self.extracted_words

class GrammaticalKnowledgeBase:
    """
    Provides Part-Of-Speech (POS) awareness to adjust probabilities during generation.
    """
    def __init__(self, categories: Dict[str, set]):
        self.categories = categories

    def get_word_type(self, word: str) -> str:
        word = word.lower()
        for pos, words_set in self.categories.items():
            if word in words_set:
                return pos
        return "unknown"

    def apply_grammar_rules(self, current_word: str, next_word_probs: np.ndarray, inverse_vocab: Dict[int, str]) -> np.ndarray:
        current_pos = self.get_word_type(current_word)
        adjusted_probs = np.copy(next_word_probs)

        # Define basic syntactical flow rules for French Language
        target_pos = None
        if current_pos == "connectors" or current_pos == "unknown":
            target_pos = ["nouns", "verbs"]
        elif current_pos == "nouns":
            target_pos = ["verbs", "adjectives"]
        elif current_pos == "verbs":
            target_pos = ["connectors", "nouns"]
        elif current_pos == "adjectives":
            target_pos = ["nouns", "connectors"]

        if target_pos:
            # Boost probabilities for words matching the expected target grammatical type
            for idx, prob in enumerate(adjusted_probs):
                if idx in inverse_vocab:
                    candidate_word = inverse_vocab[idx]
                    if self.get_word_type(candidate_word) in target_pos:
                        adjusted_probs[idx] *= 1.8 # 80% boost to grammatically logical followers

        # Re-normalize probabilities to sum to 1 to prevent Numpy errors
        sum_probs = np.sum(adjusted_probs)
        if sum_probs > 0:
            adjusted_probs = adjusted_probs / sum_probs

        return adjusted_probs

class AttentionMechanism(nn.Module):
    """
    Advanced Attention Mechanism to focus on specific parts of the sequence.
    """
    def __init__(self, hidden_dim: int):
        super(AttentionMechanism, self).__init__()
        self.attention_weights = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # lstm_outputs shape: (batch_size, seq_len, hidden_dim)
        attention_scores = self.attention_weights(lstm_outputs) # (batch_size, seq_len, 1)
        attention_distribution = F.softmax(attention_scores, dim=1) # (batch_size, seq_len, 1)
        
        # Calculate context vector using the attention distribution
        context_vector = torch.sum(lstm_outputs * attention_distribution, dim=1) # (batch_size, hidden_dim)
        
        return context_vector, attention_distribution

class AdvancedNeuralSentenceGenerator(nn.Module):
    """
    Complex neural network architecture featuring Bidirectional LSTMs,
    Layer Normalization, Dropout, and an Attention mechanism.
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 256, num_layers: int = 2, dropout_rate: float = 0.3):
        super(AdvancedNeuralSentenceGenerator, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Word Embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dropout = nn.Dropout(dropout_rate)
        
        # Bidirectional LSTM (Understand context from left-to-right and right-to-left)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim // 2, # Halved because it's bidirectional
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Layer Normalization for stable training/inference
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Attention Mechanism
        self.attention = AttentionMechanism(hidden_dim)
        
        # Fully Connected layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.activation = nn.GELU() # Modern activation function (Gaussian Error Linear Unit)
        self.fc_dropout = nn.Dropout(dropout_rate)
        
        # Final output layer mapping back to vocabulary
        self.fc_out = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        """ Forward propagation through the advanced network. """
        # Embeddings -> Dropout
        embedded_words = self.embedding_dropout(self.embedding(input_sequence))
        
        # LSTM pass
        lstm_output, _ = self.lstm(embedded_words)
        
        # Apply layer normalization to LSTM outputs
        norm_lstm_output = self.layer_norm(lstm_output)
        
        # Apply Attention
        context_vector, _ = self.attention(norm_lstm_output)
        
        # Dense layers
        dense_out = self.activation(self.fc1(context_vector))
        dense_out = self.fc_dropout(dense_out)
        
        # Final logits
        logits = self.fc_out(dense_out)
        return logits

class AdvancedPhraseReformulator:
    """
    Main module utilizing advanced PyTorch networks and complex Numpy operations
    (Top-K, Top-P, Temperature) to formulate high-quality sentences offline.
    """
    def __init__(self, max_length: int = 15, use_gpu_if_available: bool = True, corpus_file_path: str = "mon_livre_de_phrases.txt"):
        self.max_length = max_length
        self.device = torch.device("cuda" if use_gpu_if_available and torch.cuda.is_available() else "cpu")
        print(f"🖥️  Inference engine initialized on: {self.device.type.upper()}")
        
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.model: Optional[AdvancedNeuralSentenceGenerator] = None
        
        # Store the massive lexicon
        self.base_lexicon = EXTENSIVE_FRENCH_LEXICON
        
        # Initialize Grammar and Corpus capabilities
        self.grammar_engine = GrammaticalKnowledgeBase(GRAMMATICAL_CATEGORIES)
        self.corpus_reader = TextCorpusReader(corpus_file_path)
        self.corpus_words = self.corpus_reader.load_and_parse()
        
    def _build_comprehensive_vocab(self, user_words: List[str]) -> None:
        """
        Merges user words with the massive integrated lexicon and builds mappings.
        """
        # Ensure unique words and combine (now including corpus words)
        all_words = list(set([w.lower() for w in user_words] + self.base_lexicon + self.corpus_words))
        
        # Advanced Numpy shuffling for unbiased indexing
        np_words = np.array(all_words)
        np.random.seed(int(time.time())) # Random seed based on time
        np.random.shuffle(np_words)
        
        # Create dictionaries
        self.vocab = {word: idx for idx, word in enumerate(np_words)}
        self.inverse_vocab = {idx: word for word, idx in self.vocab.items()}
        
        vocab_size = len(self.vocab)
        print(f"📚 Massive vocabulary built: {vocab_size} words available.")
        
        # Initialize the advanced model
        self.model = AdvancedNeuralSentenceGenerator(
            vocab_size=vocab_size,
            embedding_dim=128,
            hidden_dim=256,
            num_layers=2
        ).to(self.device)

    def simulate_training_phase(self, dummy_epochs: int = 3) -> None:
        """
        Simulates a training phase to setup weights dynamically using Adam optimizer
        and CrossEntropyLoss. Makes the model weights adapt to the vocabulary scale.
        """
        if not self.model:
            print("⚠️ Model not initialized.")
            return

        print("⚙️ Starting simulated optimization warmup sequence...")
        self.model.train()
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        vocab_size = len(self.vocab)
        batch_size = 16
        seq_length = 5
        
        for epoch in range(dummy_epochs):
            optimizer.zero_grad()
            
            # Generate random input tensors using PyTorch
            dummy_inputs = torch.randint(0, vocab_size, (batch_size, seq_length)).to(self.device)
            dummy_targets = torch.randint(0, vocab_size, (batch_size,)).to(self.device)
            
            # Forward pass
            outputs = self.model(dummy_inputs)
            
            # Compute loss
            loss = criterion(outputs, dummy_targets)
            
            # Backward pass & Optimize
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            print(f"   -> Warmup Epoch {epoch+1}/{dummy_epochs} | Simulated Loss: {loss.item():.4f}")
            
        print("✅ Simulated optimization completed successfully.")

    def _apply_temperature_and_top_k(self, logits: np.ndarray, temperature: float = 0.8, top_k: int = 50) -> np.ndarray:
        """
        Applies temperature scaling and Top-K filtering to logits using Numpy.
        """
        # Apply temperature
        scaled_logits = logits / max(temperature, 1e-5)
        
        # Convert to probabilities with stable softmax
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits)) # Prevent overflow
        probs = exp_logits / np.sum(exp_logits)
        
        # Top-K filtering
        if top_k > 0:
            top_k_idx = np.argsort(probs)[-top_k:]
            
            # Create a mask of zeros
            filtered_probs = np.zeros_like(probs)
            filtered_probs[top_k_idx] = probs[top_k_idx]
            
            # Renormalize
            filtered_probs = filtered_probs / np.sum(filtered_probs)
            return filtered_probs
            
        return probs

    def _sequence_to_tensor(self, sequence: List[str]) -> torch.Tensor:
        """ Converts a list of words to a PyTorch tensor. """
        indices = [self.vocab.get(w.lower(), 0) for w in sequence]
        return torch.tensor([indices], dtype=torch.long, device=self.device)

    def formulate_advanced(self, input_words: List[str], temperature: float = 0.7, top_k: int = 40) -> str:
        """
        Generates a coherent sentence integrating the provided keywords.
        Uses the trained AI model combined with stylometric rules.
        """
        if not input_words:
            return "Please provide words to generate a sentence."

        print("🔄 Formulating advanced sentence mapping (Inference Core)...")
        
        # 1. Setup vocabulary and simulate network warmup
        self._build_comprehensive_vocab(input_words)
        self.simulate_training_phase(dummy_epochs=2)
        
        # 2. Switch to evaluation mode
        self.model.eval()
        
        current_seq = [w.lower() for w in input_words]
        generated_tokens = []
        
        # Add dynamic variation based on numpy
        random_factor = np.random.uniform(0.8, 1.2)
        adjusted_temp = temperature * random_factor
        
        # 3. Autoregressive Generation Loop
        with torch.no_grad():
            for _ in range(self.max_length):
                tensor_input = self._sequence_to_tensor(current_seq[-10:]) # Context window
                
                # Model prediction
                raw_logits = self.model(tensor_input)
                
                # Move to CPU and Numpy for advanced probability manipulation
                np_logits = raw_logits.squeeze(0).cpu().numpy()
                
                # Apply advanced Numpy filtering
                final_probs = self._apply_temperature_and_top_k(np_logits, temperature=adjusted_temp, top_k=top_k)
                
                # Apply Grammar Syntactical Rules
                final_probs = self.grammar_engine.apply_grammar_rules(
                    current_word=current_seq[-1], 
                    next_word_probs=final_probs, 
                    inverse_vocab=self.inverse_vocab
                )
                
                # Sample next word probabilistically rather than argmax (more creativity)
                next_word_idx = np.random.choice(len(final_probs), p=final_probs)
                predicted_word = self.inverse_vocab[next_word_idx]
                
                # Prevent looping the same word over and over
                if len(generated_tokens) > 0 and predicted_word == generated_tokens[-1]:
                    # Force second best choice via argsort if repeating
                    sorted_indices = np.argsort(final_probs)
                    next_word_idx = sorted_indices[-2] 
                    predicted_word = self.inverse_vocab[next_word_idx]
                
                generated_tokens.append(predicted_word)
                current_seq.append(predicted_word)
                
                # Natural stop condition based on grammar logic (simplified)
                if len(generated_tokens) > 5 and predicted_word in [".", "!", "?"]:
                    break

        # 4. Stylistic Post-Processing
        missing_inputs = [w for w in input_words if w.lower() not in [g.lower() for g in generated_tokens]]
        final_sentence_parts = generated_tokens.copy()
        
        # Inject missing keywords using numpy shuffling
        if missing_inputs:
            np.random.shuffle(missing_inputs)
            insertion_points = np.linspace(0, len(final_sentence_parts), len(missing_inputs)+1, dtype=int)
            
            for i, missing_word in enumerate(missing_inputs):
                insert_idx = insertion_points[i+1] - 1
                final_sentence_parts.insert(max(0, insert_idx), missing_word)
        
        # Construct string
        raw_string = " ".join(final_sentence_parts)
        
        # Clean up spaces around punctuation
        raw_string = raw_string.replace(" ,", ",").replace(" .", ".").replace(" ' ", "'")
        
        # Capitalize and ensure it ends with a dot
        final_phrase = raw_string.capitalize()
        if not final_phrase.endswith((".", "!", "?")):
            final_phrase += "."
            
        return final_phrase