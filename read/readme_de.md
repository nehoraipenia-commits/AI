# AI - Neural Generative Core & Semantic Framework

Ein fortschrittliches, leichtgewichtiges und hybrides System für generative Intelligenz, das nativ mit PyTorch entwickelt wurde. Das Framework kombiniert eine benutzerdefinierte kausale Transformer-Architektur mit Byte-Pair Encoding (BPE), semantischer Vektorsuche (RAG), französischer Satzrekonstruktion (FSR), dynamischer Benutzerprofilverwaltung und kooperativem Offline-Fine-Tuning lokaler LLMs über Ollama-Instanzen.

### Hauptfunktionen

* Benutzerdefinierte kausale Transformer-Engine: Vollständig mit PyTorch-Primitiven entwickelt und mit sinusförmigen Positionskodierungen, kausaler Multi-Head-Self-Attention und Layer Normalization ausgestattet.

* Byte-Pair-Encoding-(BPE)-Tokenisierung: Benutzerdefinierte Subword-Tokenisierungspipeline mit dynamischer Skalierung des Vokabulars sowie vollständiger Unicode- und Akzentnormalisierung.

* Semantischer Multi-Datei-Speicher (RAG): Vektorspeicher auf Basis von TF-IDF und Kosinus-Ähnlichkeit, der Python-Codebasen (.py) und Dokumentationen (.txt) indizieren kann, um eine kontextbezogene Generierung zu ermöglichen.

* Ollama-Integration & dynamisches Fine-Tuning: Kooperatives Multi-Modell-Framework, das Wissenserweiterungen von lokalen LLMs (z. B. Gemma, Llama) abruft, um die benutzerdefinierten PyTorch-Gewichte dynamisch und offline zu trainieren.

* Französische Satzrekonstruktion (FSR): Dediziertes neuronales Seq2Seq-Modul zur Korrektur französischer Kontraktionen, Syntaxfehler und Tippfehler.

* Dynamischer Benutzerspeicher & Profil-Engine: Echtzeit-Extraktion von Benutzerinformationen (Name, Präferenzen, Kontext), die über fortlaufende Chat-Sitzungen hinweg verwaltet werden.

* Interaktive CLI-Oberfläche: Umfangreiche Terminal-Oberfläche mit Live-Ausführungsanalysen und Verwaltung des Modellstatus.

* Systemarchitektur

```text id="f3r6p7"
                                  +---------------------------+
                                  |   Benutzereingabe /       |
                                  |         Prompt            |
                                  +-------------+-------------+
                                                |
                                                v
   +----------------------------+   +-----------+-----------+   +----------------------------+
   | Dynamischer                |-->| Semantischer Speicher |<--| Indizierte Dateien        |
   | Benutzerspeicher           |   | (RAG)                 |   | des Verzeichnisses         |
   | (Fakten & Verlauf)         |   | (TF-IDF-Vektorspeicher)|  | (.py & .txt)              |
   +----------------------------+   +-----------+-----------+   +----------------------------+
                                                |
                                                v
                                 +--------------+--------------+
                                 | Byte-Pair-Encoding-(BPE)-  |
                                 | Tokenizer-Pipeline         |
                                 +--------------+--------------+
                                                |
                                                v
                                 +--------------+--------------+
                                 | Benutzerdefinierter        |
                                 | kausaler Transformer        |
                                 | (Multi-Head-Self-Attention) |
                                 +--------------+--------------+
                                                |
                                                v
  +----------------------------+    +-----------+-------------+   +----------------------------+
  | Französisches             |<--| Intelligente            |-->| Lokales Ollama-LLM       |
  | Rekonstruktionsmodul      |   | Sampling-Engine          |   | (Kooperative             |
  | (FSR-Grammatikkorrektur)  |   | (Top-K/Nucleus Sampling) |   | Warmup-Schleifen)        |
  +----------------------------+    +-----------+-------------+   +----------------------------+
                                                |
                                                v
                                  +-------------+-------------+
                                  |   Generierte KI-Antwort   |
                                  +---------------------------+
```

## Voraussetzungen

* Stellen Sie vor der Installation sicher, dass die folgende Software auf Ihrem Rechner installiert ist:

* Python: Version 3.8 oder höher

* PyTorch: Version 1.12.0 oder höher (CUDA/MPS-Unterstützung optional, aber empfohlen)

* Ollama (Optional): Nur erforderlich, wenn Sie das kooperative LLM-Fine-Tuning lokal ausführen möchten.

* Installation & Einrichtung

### Repository klonen

```bash id="k6p0j4"
git clone https://github.com/nehoraipenia-commits/AI.git
cd AI
```

### Virtuelle Umgebung erstellen und aktivieren (Empfohlen)

## Unter macOS/Linux

```bash id="0m4x3t"
python3 -m venv venv
source venv/bin/activate
```

## Unter Windows

```bash id="txk8qz"
python -m venv venv
venv\Scripts\activate
```

### Erforderliche Abhängigkeiten installieren

```bash id="cvqf5n"
pip install torch numpy scikit-learn requests
```

### Die Assistant-CLI starten

```bash id="w4t5f1"
python3 ai_pytorch.py
```

## Interaktive CLI-Befehle

Sobald die CLI gestartet wurde (`AI-CLI >>>`), können Sie normale Texteingaben verwenden oder Systemsteuerungsbefehle ausführen:

Befehl

Beschreibung

```text id="q3m1c9"
/status
```

Zeigt Hardware-Spezifikationen, Gerätezuweisung, die Größe des aktiven BPE-Vokabulars und den Status der Speicherbasisdatei an.

```text id="z6f0n2"
/train <epochs>
```

Optimiert die Transformer-Gewichte direkt anhand Ihrer lokalen Codebasis und Dokumentation neu.

```text id="x1k5p8"
/model <name>
```

Ändert die Kennung des aktiven lokalen Ollama-Modells (z. B. `/model gemma2:2b` oder `/model llama3`).

```text id="b7q2v4"
/gemma <count>
```

Führt kooperative Generierungsschleifen mit Ollama aus, um die lokale Wissensbasis zu erweitern und das Modell neu zu trainieren.

```text id="n8r3s6"
/train_fsr <ep>
```

Trainiert das Transformer-Modul zur französischen Satzrekonstruktion anhand lokaler Texte.

```text id="j2w9m5"
/reconstruct <text>
```

Korrigiert fehlerhafte französische Formulierungen, Tippfehler oder Kontraktionen mithilfe des FSR-Modells.

```text id="p4c7y1"
/save
```

Exportiert die aktuellen Checkpoints und Gewichte des Systems manuell auf die Festplatte (`checkpoints/`).

```text id="r5h8k3"
/load
```

Lädt gespeicherte Systemzustände und Tokenizer-Vokabulare aus den Checkpoint-Dateien erneut.

```text id="t9v2d6"
/clear
```

Löscht den Bildschirm der Terminal-Oberfläche.

```text id="m3x7q0"
/exit
```

Beendet die Prozesse sicher und speichert die aktuellen Gewichte.

## Repository-Struktur

```text id="s8n4k2"
AI/
├── AI.py                            # Hauptcontroller der System-Pipeline & interaktive CLI
├── AdvancedPhraseReformulator.py    # POS-basiertes grammatikalisches Reformulierungsmodul
├── ai_generation_helper.py          # Intelligentes Logit-Sampling, Markov-Engine & Textoptimierung
├── FrenchContractionHandler.py      # Französische Kontraktionsregeln & Seq2Seq-Rekonstruktion
├── dynamic_user_memory.py           # Dynamischer Benutzerfakten-Extraktor & Verlaufsverwaltung
├── nlp_tokenization_suite.py        # Byte-Pair-Encoding-(BPE)-Tokenizer-Suite
├── big_book.txt                     # Primäres Systemkorpus & Wissensspeicher
├── ollama_responses.txt             # Protokoll der kooperativen LLM-Wissenserweiterungen
└── checkpoints/                     # Verzeichnis zur Serialisierung der Modellgewichte
```

Weitere Informationen: https://nexeo-ai.netlify.app/

