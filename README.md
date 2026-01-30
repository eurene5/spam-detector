# spam-detector

**Institut**: ISPM – Institut Supérieur Polytechnique de Madagascar

**Site web**: https://www.ispm-edu.com

**Équipe**:

- LOCK Lorick Dilan - Dev
- RABE ANDRIANASOLO Johaninho Nirinjaka - Dev

**Stack Technologique**:

- Langage: Python 3.11+
- API: FastAPI + Uvicorn
- Machine Learning: scikit-learn (RandomForestClassifier, MultinomialNB)
- Prétraitement: pandas, NLTK
- Vectorisation: `sklearn.feature_extraction.text.CountVectorizer` (unigram + bigram)
- Visualisation / Analyse: seaborn, matplotlib
- Serialisation du modèle: `pickle`
- Frontend: HTML/CSS/Vanilla JavaScript (client léger)

**Processus et Modèle**:

- Chargement des données: `data-augmented.csv` (multilingue) + `data_mg.csv` (Malagasy)
- Prétraitement: nettoyage basique (suppression ponctuation, tokenisation simple). Le pipeline évite la suppression de stopwords spécifique pour supporter le Malagasy et d'autres langues.
- Vectorisation: `CountVectorizer` avec `ngram_range=(1,2)` et un `tokenizer` personnalisé.
- Modèle principal: `RandomForestClassifier` (n_estimators=100, class_weight='balanced') entraîné sur les features bag-of-ngrams.
- Évaluation: séparation entraînement/test (80/20), métriques calculées: accuracy, precision, recall, F1-score, matrice de confusion et rapport de classification.
- Production: modèle et vectorizer sérialisés (`spam_model.pkl`, `vectorizer.pkl`) pour l'inférence via l'API.

**Méthodes de Machine Learning**:

- Random Forest (classifieur principal, robuste aux caractéristiques bruitées et classes déséquilibrées)
- (Utilisé ailleurs dans le dépôt) Multinomial Naive Bayes — utile comme baseline pour texte
- Techniques: bag-of-words / n-grammes (unigrammes + bigrammes), tokenization simple, rééchantillonnage implicite via `class_weight='balanced'` pour gérer le déséquilibre

**Datasets utilisés**:

- `data-augmented.csv` — version augmentée du corpus SMS Spam Collection, colonnes multilingues (`text`, `text_fr`, etc.)
- `data_mg.csv` — dataset d'exemples en Malagasy (ajouté localement) contenant des exemples `spam` et `ham` (format: `labels,text_mg`)
- SMS Spam Collection (corpus d'origine) — utilisé pour enrichir et valider les échantillons

**Application Web**:

- API locale (développement) : `http://localhost:8000` (endpoints: `/health`, `/predict`, `/batch`, `/stats`)
- Frontend local : ouvrez `src/client/index.html` dans votre navigateur
- Lien de l'application web hébergée: `https://spam-detector-ouo8.onrender.com/`

---
