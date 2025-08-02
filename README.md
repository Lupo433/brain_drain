# 🌍 GoWhere - Trova il tuo Paese Ideale

GoWhere è una data app interattiva costruita con **Streamlit** che aiuta l'utente a identificare i paesi migliori dove trasferirsi in base a:
- Miglioramento di condizioni personali (es. lavoro, reddito, sicurezza)
- Interessi personali (es. ambiente, soddisfazione di vita)
- Origine e genere

Include anche un **diagramma di Hasse** per visualizzare relazioni di dominanza tra paesi secondo indici selezionati.

---

## 🚀 Come usare l'app

1. **Clona la repository o scarica i file.**
2. Assicurati di avere Python 3.9+ installato.
3. Installa i pacchetti richiesti:
```bash
pip install -r requirements.txt
```
> ⚠️ Nota: `pygraphviz` richiede Graphviz installato sul sistema. Su Ubuntu/Debian:
```bash
sudo apt-get install graphviz graphviz-dev
```

4. Avvia l'app Streamlit:
```bash
streamlit run app.py
```

---

## 📁 File inclusi

- `app.py` → codice principale dell’app Streamlit
- `dataset_final.csv` → dataset dei flussi migratori e indicatori OCSE
- `requirements.txt` → pacchetti Python necessari

---

## 📊 Diagramma di Hasse

Il diagramma permette di esplorare visivamente le relazioni di dominanza tra paesi, secondo tre indicatori a scelta e colorati rispetto a un quarto indicatore.

---

## 🧠 Credits

Autore: [@Lupo433](https://github.com/Lupo433)  
Dati OCSE pre-elaborati per analisi.

---

## ✅ Esempio Screenshot

![screenshot](preview.png)