# AML Challenge 2026/27: versione 2, cosa cambia e perché

Questo documento continua `scelte_progettuali.md`, che descrive la versione 1. Il notebook della v2 è `aml_semcorr_v2.ipynb`. La v1 resta invariata come riferimento: CV onesta **0.5435**, con DINOv2-L, 644 px e fine-tuning degli ultimi 2 blocchi per 5 epoche.

Ogni modifica parte da un'osservazione sui risultati della v1. Dove una scelta non è scontata, la decide la validazione e non l'intuizione.

---

## 1. Risoluzione più alta: 840 px invece di 644

**Osservazione (v1).** Con una tolleranza doppia (PCK@0.20) il risultato è 0.758, contro 0.544 della metrica ufficiale (PCK@0.10). Una parte grande degli errori cade quindi *vicino* al punto giusto. Il modello riconosce la parte corretta, ma la localizza con poca precisione.

**Scelta.** Portiamo il lato lungo a 840 px: per DINOv2 sono 60 patch invece di 46, cioè una griglia più fine e una localizzazione più precisa.

**Come la verifichiamo.** La prima cella esegue lo **zero-shot a 644 e a 840 px** sullo stesso backbone congelato. Non c'è nessun training, quindi il confronto è onesto. Se 840 non migliora, basta rimettere `CFG.long_side=644`.

**Costo.** I token in cache crescono di circa 1.7 volte. Per contenere la memoria, in modalità fine-tuning teniamo in RAM **una sola cache per backbone**, quella dei token in ingresso agli ultimi blocchi. Le feature "congelate" vengono ricalcolate al volo da quei token, perché costa solo 2 blocchi. In v1 invece c'erano due cache.

---

## 2. Ensemble DINOv2 + DINOv3

**Motivo.** DINOv3 è più recente e ha feature dense migliori secondo i suoi autori. Due backbone addestrati in modo diverso sbagliano su casi diversi, e la somma pesata delle mappe di similarità tende a correggere gli errori di ciascuno.

**Scelta.**

- Si attiva automaticamente se esiste il secret Kaggle `HF_TOKEN`: i pesi di DINOv3 richiedono di accettare la licenza su Hugging Face. Senza token il notebook usa solo DINOv2.
- **Ogni backbone ha il suo fine-tuning:** gli ultimi 2 blocchi e una testa separati. Le mappe di similarità finali vengono poi mediate con pesi uguali.
- DINOv3 usa embedding di posizione rotazionali (RoPE). La cache dei token salva anche questi, così gli ultimi blocchi ricevono esattamente gli stessi input del modello originale. Lo abbiamo verificato: prima del training, la copia dei blocchi riproduce le feature originali a meno di 1e-4, per entrambi i modelli.
- La valutazione zero-shot riporta anche ogni backbone da solo, così si vede il contributo di ciascuno.

---

## 3. Meno epoche: 3 invece di 5

**Osservazione (v1).** La curva media di validazione del fine-tuning vale 0.526 → **0.548** → 0.546 → 0.546 → 0.543. Il massimo è all'epoca 2–3, poi compare un leggero overfitting.

**Scelta.** 3 epoche. Lo schedule a coseno si adatta al numero totale di passi, quindi con 3 epoche anche il learning rate scende prima.

---

## 4. Augmentation con scambio sorgente/target

**Motivo.** I dati di training sono pochi. Una corrispondenza A→B vale anche al contrario, B→A, quindi possiamo quasi raddoppiare gli esempi senza dati nuovi.

**Scelta.** Usiamo solo le query con **un unico** target accettato. Con più target, l'inverso non è definito: da quale delle due ruote si partirebbe?

- La query inversa parte dal punto target e ha come risposta il punto sorgente.
- La metrica e la loss richiedono la dimensione dell'oggetto nell'immagine che diventa target, e quel valore non è fornito. Lo stimiamo come `scale × (estensione dei keypoint sorgente / estensione dei keypoint target)`, limitato a [0.25, 4]. Se c'è un solo punto, usiamo il rapporto tra i lati lunghi delle immagini. È una stima approssimativa, ma serve solo a fissare la larghezza della gaussiana nella loss.
- L'augmentation si applica **solo ai fold di training**. La validazione resta sui dati originali, quindi il punteggio CV è confrontabile con la v1.

---

## 5. Coerenza geometrica (prior affine con RANSAC)

**Problema.** Molti errori finiscono su una parte *simile ma sbagliata*: l'altra zampa, l'altra ruota, una zona dello sfondo. Guardando una query alla volta, il modello non può accorgersene.

**Idea.** Le altre parti dello stesso oggetto danno un contesto. Per ogni coppia:

1. troviamo il miglior match nel target per **tutti** i punti sorgente della coppia, anche quelli senza query, già nell'orientamento scelto dal flip;
2. stimiamo con **RANSAC** una trasformazione affine robusta sorgente → target, con almeno 4 inlier;
3. per ogni query, l'affine predice una posizione approssimativa. Alla mappa di similarità sottraiamo una penalità `λ · distanza²`, normalizzata sul lato lungo dell'immagine. I picchi lontani dalla posizione attesa vengono così sfavoriti.

**Perché "morbida".** Oggetti di istanze diverse non sono legati da un'affine esatta, per via di pose e articolazioni diverse. Per questo non sostituiamo la previsione con quella dell'affine: aggiungiamo solo un prior regolato da λ. Se la stima non è affidabile, cioè con troppo pochi inlier, il prior non viene applicato.

**Come scegliamo λ.** Lo decide la validazione (§6). Se λ = 0 risulta migliore, il prior resta spento: è una scelta guidata dai dati, non un'aggiunta obbligatoria.

Non usiamo alcuna etichetta del test: il prior si basa solo sui punti sorgente, che fanno parte dell'input, e sulle previsioni del modello.

---

## 6. Parametri di inferenza scelti in cross-validation

**Problema.** Raggio e temperatura del soft-argmax (6 e 0.02) erano stati fissati a mano. La forza del prior geometrico (λ) è nuova e va scelta.

**Scelta.** Durante la CV salviamo le previsioni *out-of-fold*, cioè ogni coppia predetta da un modello che non l'ha mai vista. Su queste proviamo una griglia di **45 combinazioni**:

- raggio ∈ {4, 6, 9}
- temperatura ∈ {0.01, 0.02, 0.04}
- λ ∈ {0, 0.5, 1, 2, 4}

Le mappe di similarità si calcolano una volta sola, quindi la ricerca costa pochi secondi. La combinazione migliore viene poi usata per il test.

**Onestà della stima.** Scegliere 3 iperparametri su circa 2.500 previsioni out-of-fold introduce un ottimismo minimo, molto minore di quello della scelta dell'epoca per fold che avevamo eliminato in v1. A parità di punteggio vince la configurazione più semplice: λ = 0 e valori predefiniti.

---

## 7. Cosa non cambia

- Stessa validazione: 5 fold divisi per componente connessa, quindi nessuna immagine condivisa tra i fold.
- Stessa loss: gaussiana attorno al target accettato più vicino, flip di entrambe le immagini come augmentation.
- Stesso allineamento del flip sorgente e stesso soft-argmax.
- Stesse regole: solo dati della challenge, modelli pre-addestrati pubblici, nessuna informazione dal test.

---

## 8. Come leggere i risultati della v2

| Cella del notebook | Cosa guardare |
|---|---|
| 10. Controllo della risoluzione | Lo zero-shot a 840 è migliore che a 644? Se no, torna a 644. |
| 11. Zero-shot completo | Il contributo di ciascun backbone e dell'ensemble. |
| 12. Cross-validation | La PCK CV da confrontare con **0.5435** della v1, la curva per epoca e la miglior combinazione raggio/temperatura/λ. |
| 13. Submission | Viene generata con le impostazioni scelte in CV. |

Se un componente non aiuta, si spegne dalla cella `CFG`, e ogni scelta resta così giustificata da un numero:

- prior geometrico: `geo_lambda=0`, `sweep=False`;
- scambio sorgente/target: `swap_aug=False`;
- risoluzione: `long_side=644`.

---

## 9. Risultati del primo run della v2 e correzione (v2.1)

**Cosa ha funzionato.**

- **Risoluzione:** DINOv2 senza training passa da **0.447** a 644 px a **0.506** a 840 px. Il guadagno più forte è sugli oggetti piccoli (sotto 100 px: da 0.461 a 0.578). La scelta dei 840 px è quindi confermata.
- **Fine-tuning di DINOv2 a 840 px**, con 3 epoche e scambio sorgente/target: la media dei fold di validazione è circa **0.576**, contro 0.5435 della v1. Questo numero viene dai log per epoca del solo DINOv2, perché il punteggio finale della CV era falsato dal bug descritto sotto.

**Cosa non ha funzionato: DINOv3 produceva NaN.** Il modello girava in mezza precisione (fp16) per risparmiare memoria. DINOv3 però ha attivazioni interne molto grandi, che superano il massimo rappresentabile in fp16 (circa 65.000), e producono valori infiniti e quindi NaN. Di conseguenza:

- le sue feature erano NaN anche senza training;
- la loss del suo fine-tuning era NaN;
- nell'ensemble i NaN si propagavano a tutta la mappa di similarità: CV a 0 e submission con coordinate NaN.

**Correzione (v2.1).**

1. DINOv3 gira in **fp32**, mentre DINOv2 resta in fp16, che per lui è sicuro e più veloce.
2. I token in cache vengono salvati in **bf16** quando superano l'intervallo dell'fp16: stessa memoria, intervallo molto più ampio.
3. Il training a precisione mista viene **disattivato per DINOv3**.
4. Controlli che fermano subito il run, invece di lasciarlo girare per ore su dati sbagliati: feature non finite all'avvio, loss NaN durante il training, previsioni non finite prima di scrivere la submission.
5. Il confronto tra risoluzioni è disattivato di default, perché è già stato misurato.
