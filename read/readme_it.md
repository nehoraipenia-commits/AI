# AI - Neural Generative Core & Semantic Framework

Un sistema avanzato, leggero e ibrido di intelligenza generativa, sviluppato nativamente con PyTorch. Il framework combina un'architettura Transformer causale personalizzata con Byte-Pair Encoding (BPE), ricerca vettoriale semantica (RAG), ricostruzione delle frasi francesi (FSR), gestione dinamica del profilo utente e fine-tuning cooperativo offline di LLM locali tramite istanze Ollama.

### Caratteristiche principali

* Motore Transformer causale personalizzato: Sviluppato interamente utilizzando le primitive PyTorch, con codifiche posizionali sinusoidali, attenzione causale multi-head e normalizzazione dei livelli.

* Tokenizzazione Byte-Pair Encoding (BPE): Pipeline personalizzata di tokenizzazione per sottoparole con ridimensionamento dinamico del vocabolario e normalizzazione completa di Unicode e degli accenti.

* Memoria semantica multi-file (RAG): Archivio vettoriale basato su TF-IDF e similarità coseno, in grado di indicizzare basi di codice Python (.py) e documentazione (.txt) per una generazione consapevole del contesto.

* Integrazione Ollama & fine-tuning dinamico: Framework cooperativo multi-modello che recupera espansioni della conoscenza da LLM locali (ad esempio Gemma, Llama) per addestrare dinamicamente i pesi PyTorch personalizzati offline.

* Ricostruzione delle frasi francesi (FSR): Modulo neurale Seq2Seq dedicato alla correzione delle contrazioni francesi, degli errori sintattici e degli errori di battitura.

* Memoria dinamica dell'utente & motore del profilo: Estrazione in tempo reale delle informazioni dell'utente (nome, preferenze, contesto), mantenute attraverso sessioni di chat successive.

* Interfaccia CLI interattiva: Interfaccia terminale completa con analisi dell'esecuzione in tempo reale e gestione dello stato del modello.

* Architettura del sistema

```text id="f7q2m1"
                                  +---------------------------+
                                  |    Input utente / Prompt  |
                                  +-------------+-------------+
                                                |
                                                v
   +----------------------------+   +-----------+-----------+   +----------------------------+
   | Memoria dinamica           |-->| Memoria semantica      |<--| File della directory      |
   | dell'utente               |   | (RAG)                  |   | indicizzati (.py & .txt)  |
   | (fatti & cronologia)      |   | (Archivio TF-IDF)      |   |                            |
   +----------------------------+   +-----------+-----------+   +----------------------------+
                                                |
                                                v
                                 +--------------+--------------+
                                 | Pipeline di tokenizzazione |
                                 | Byte-Pair Encoding (BPE)   |
                                 +--------------+--------------+
                                                |
                                                v
                                 +--------------+--------------+
                                 | Transformer causale        |
                                 | personalizzato             |
                                 | (Attenzione multi-head)     |
                                 +--------------+--------------+
                                                |
                                                v
  +----------------------------+    +-----------+-------------+   +----------------------------+
  | Modulo di ricostruzione    |<--| Motore di sampling       |-->| LLM Ollama locale        |
  | francese                   |   | intelligente              |   | (Cicli di warmup         |
  | (Correzione grammaticale)  |   | (Top-K/Nucleus Sampling) |   | cooperativi)             |
  +----------------------------+    +-----------+-------------+   +----------------------------+
                                                |
                                                v
                                  +-------------+-------------+
                                  |    Risposta generata      |
                                  |          dall'IA           |
                                  +---------------------------+
```

## Prerequisiti

* Prima dell'installazione, assicurati che i seguenti software siano installati sulla tua macchina:

* Python: Versione 3.1 o superiore

* PyTorch: Versione 1.12.0 o superiore (supporto CUDA/MPS opzionale ma consigliato)

* Ollama (Opzionale): Necessario solo se intendi utilizzare il fine-tuning cooperativo di LLM locali.

* Installazione e configurazione

### Clonare il repository

```bash id="v6k1p8"
git clone https://github.com/nehoraipenia-commits/AI.git
cd AI
```

### Creare e attivare un ambiente virtuale (Consigliato)

## Su macOS/Linux

```bash id="x2m7q4"
python3 -m venv venv
source venv/bin/activate
```

## Su Windows

```bash id="q9w3n5"
python -m venv venv
venv\Scripts\activate
```

### Installare le dipendenze richieste

```bash id="k5r8t2"
pip install torch numpy scikit-learn requests
```

### Avviare la CLI dell'assistente

```bash id="p4y7m1"
python3 ai_pytorch.py
```

## Comandi della CLI interattiva

Una volta avviata la CLI (`AI-CLI >>>`), puoi inserire normali prompt di testo oppure utilizzare i comandi di controllo del sistema:

Comando

Descrizione

```text id="n8c3v6"
/status
```

Visualizza le specifiche hardware, l'allocazione del dispositivo, la dimensione del vocabolario BPE attivo e lo stato del file di base della memoria.

```text id="m2q7x4"
/train <epochs>
```

Ottimizza nuovamente i pesi del Transformer direttamente sulla base del codice e della documentazione locali.

```text id="r6t1p9"
/model <name>
```

Modifica l'identificatore del modello Ollama locale attivo (ad esempio, `/model gemma2:2b` o `/model llama3`).

```text id="y4k8s2"
/gemma <count>
```

Esegue cicli di generazione cooperativa con Ollama per ampliare la base di conoscenza locale e riaddestrare il modello.

```text id="c7m3v5"
/train_fsr <ep>
```

Addestra il modulo Transformer per la ricostruzione delle frasi francesi utilizzando testi locali.

```text id="b1x9q6"
/reconstruct <text>
```

Corregge formulazioni francesi corrotte, errori di battitura o contrazioni utilizzando il modello FSR.

```text id="h5r2n8"
/save
```

Esporta manualmente i checkpoint e i pesi correnti del sistema sul disco (`checkpoints/`).

```text id="s3k7m4"
/load
```

Ricarica gli stati salvati del sistema e i vocabolari del tokenizer dai file di checkpoint.

```text id="w8p1c5"
/clear
```

Cancella lo schermo dell'interfaccia del terminale.

```text id="d4n6q2"
/exit
```

Termina in modo sicuro i processi e salva i pesi correnti.

## Struttura del repository

```text id="z7m2k9"
AI/
├── AI.py                            # Controller principale della pipeline del sistema & CLI interattiva
├── AdvancedPhraseReformulator.py    # Motore di riformulazione grammaticale basato sui POS
├── ai_generation_helper.py          # Sampling intelligente dei logits, motore Markov & ottimizzazione del testo
├── FrenchContractionHandler.py      # Regole delle contrazioni francesi & ricostruzione Seq2Seq
├── dynamic_user_memory.py           # Estrattore dinamico dei fatti dell'utente & gestione della cronologia
├── nlp_tokenization_suite.py        # Suite di tokenizzazione Byte-Pair Encoding (BPE)
├── big_book.txt                     # Corpus principale del sistema & archivio della conoscenza
├── ollama_responses.txt             # Registro delle espansioni generate dagli LLM cooperativi
└── checkpoints/                     # Directory per la serializzazione dei pesi del modello
```

Per maggiori informazioni: https://nexeo-ai.netlify.app/
