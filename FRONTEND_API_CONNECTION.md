# 🚀 Guide Complet: Connexion Front-End et API

## ✅ Prérequis

1. **Modèle entraîné**

   ```bash
   python src/main_rf.py
   ```

   Crée les fichiers:
   - `spam_model.pkl`
   - `vectorizer.pkl`

2. **Dépendances installées**
   ```bash
   pip install fastapi uvicorn pydantic scikit-learn pandas nltk
   ```

## 🔧 Configuration

### 1. Démarrer l'API

#### Avec le script fourni (Windows)

```bash
start_api.bat
```

#### Avec le script fourni (macOS/Linux)

```bash
bash start_api.sh
```

#### Manuellement

```bash
python -m uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

**Vous verrez:**

```
INFO:     Uvicorn running on http://0.0.0.0:8000
✓ Modèle chargé avec succès
```

### 2. Ouvrir le Client Web

Ouvrez le fichier dans votre navigateur:

```
public/index.html
```

Ou allez directement à:

```
http://localhost:8000/docs
```

## 🔍 Vérifier la Connexion

### Test 1: Vérifier l'API

```bash
curl -X GET "http://localhost:8000/health" \
  -H "Content-Type: application/json"
```

**Réponse attendue:**

```json
{
  "status": "en fonctionnement",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### Test 2: Tester une Prédiction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"message":"Cliquez ici pour réclamer votre prix!"}'
```

**Réponse attendue:**

```json
{
  "message": "Cliquez ici pour réclamer votre prix!",
  "prediction": "spam",
  "confidence": "95.30%",
  "is_spam": true
}
```

### Test 3: Vérifier le Client Web

1. Ouvrez `src/client/index.html` dans le navigateur
2. Vous devriez voir: "✓ L'API est prête. Vous pouvez commencer à analyser!"
3. Entrez un message et cliquez sur "Analyser le message"

## 🐛 Dépannage

### ❌ Erreur: "Impossible de se connecter à l'API"

**Cause**: L'API n'est pas en cours d'exécution

**Solution**:

```bash
# Terminal 1: Démarrer l'API
python -m uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Ouvrir le navigateur
# Allez à: http://localhost:8000/docs
```

### ❌ Erreur: "Modèle non chargé"

**Cause**: Les fichiers `spam_model.pkl` ou `vectorizer.pkl` n'existent pas

**Solution**:

```bash
python src/main_rf.py
```

Cela va:

1. Charger le dataset (5,576 messages)
2. Entraîner le modèle
3. Sauvegarder les fichiers

### ❌ Erreur CORS

**Symptôme**: Erreur dans la console du navigateur:

```
Access to XMLHttpRequest blocked by CORS policy
```

**Solution**: L'API est déjà configurée avec CORS activé.
Si le problème persiste, vérifiez que `src/api.py` contient:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### ❌ Port 8000 déjà utilisé

**Solution**: Utilisez un port différent:

```bash
python -m uvicorn src.api:app --reload --host 0.0.0.0 --port 5000
```

Puis mettez à jour `src/client/index.html`:

```javascript
const API_URL = 'http://localhost:5000'; // Changé de 8000 à 5000
```

## 📡 Architecture

```
┌─────────────────────────────────────────────────┐
│         Client Web (Frontend)                    │
│  - index.html: Interface utilisateur             │
│  - JavaScript: Communication avec l'API          │
│  - Port: Aucun (fichier statique)               │
└────────────┬────────────────────────────────────┘
             │
             │ HTTP/CORS
             │
┌────────────▼────────────────────────────────────┐
│         API FastAPI (Backend)                    │
│  - src/api.py: Endpoints REST                   │
│  - CORS activé: ✅                              │
│  - Port: 8000                                    │
│                                                  │
│  POST /predict                                   │
│  GET /health                                     │
│  GET /stats                                      │
└────────────┬────────────────────────────────────┘
             │
             │ Chargement en mémoire
             │
┌────────────▼────────────────────────────────────┐
│         Modèle Machine Learning                  │
│  - spam_model.pkl: Modèle Random Forest         │
│  - vectorizer.pkl: CountVectorizer              │
└──────────────────────────────────────────────────┘
```

## 🎯 Flux de Communication

1. **L'utilisateur entre un message** → Client HTML
2. **JavaScript envoie une requête POST** → API (`/predict`)
3. **L'API nettoie le texte** → Tokenization + stopwords français
4. **Vectorisation du message** → Features bigrammes
5. **Prédiction du modèle** → Probabilité spam/ham
6. **L'API retourne le résultat** → JSON
7. **Le client affiche le résultat** → Interface utilisateur

## ✨ Fonctionnalités de la Connexion

### Vérification Automatique

- Le client vérifie la connexion API au chargement
- Vérifie toutes les 30 secondes
- Affiche le statut en temps réel

### Gestion des Erreurs

- Messages d'erreur clairs en français
- Désactivation du bouton si l'API n'est pas disponible
- Affichage des détails d'erreur

### Performance

- Requête rapide (< 1 seconde)
- Affichage d'un spinner pendant le traitement
- Réponse JSON structurée

## 📚 Documentation API

Ouvrez dans le navigateur:

```
http://localhost:8000/docs
```

Vous verrez:

- ✓ Tous les endpoints disponibles
- ✓ Schémas de requête/réponse
- ✓ Exemples d'utilisation
- ✓ Tests interactifs (Try it out)

## 🔐 Sécurité en Production

Avant de déployer:

1. **Désactiver le mode debug**

   ```bash
   python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
   ```

2. **Restreindre CORS**

   ```python
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["https://votredomaine.com"],
       allow_credentials=True,
       allow_methods=["POST", "GET"],
       allow_headers=["Content-Type"],
   )
   ```

3. **Utiliser Gunicorn**

   ```bash
   pip install gunicorn
   gunicorn src.api:app -w 4 -b 0.0.0.0:8000
   ```

4. **Déployer avec Docker**
   ```dockerfile
   FROM python:3.11
   WORKDIR /app
   COPY . .
   RUN pip install -r requirements.txt
   CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

## 🎓 Exemple Complet

### Démarrer

```bash
# Terminal 1: Entraîner le modèle (première fois)
python src/main_rf.py

# Terminal 2: Démarrer l'API
python -m uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

# Terminal 3: Ouvrir le client
# Ouvrez dans le navigateur: src/client/index.html
```

### Tester

```bash
# Entrez: "Vous avez gagné 1000€ ! Cliquez ici maintenant!"
# Résultat: "⚠️ SPAM DÉTECTÉ - 95.30%"

# Entrez: "Bonjour, comment allez-vous?"
# Résultat: "✅ MESSAGE LÉGITIME - 98.50%"
```

## ❓ FAQ

**Q: Puis-je accéder à l'API depuis une autre machine?**
A: Oui! L'API écoute sur `0.0.0.0:8000`. Remplacez `localhost` par l'adresse IP du serveur.

**Q: Puis-je accéder au client web depuis Internet?**
A: Oui, mais configurez CORS et utilisez HTTPS en production.

**Q: Puis-je ajouter l'authentification?**
A: Oui, utilisez FastAPI Security avec JWT tokens.

**Q: Puis-je augmenter la taille maximale des requêtes?**
A: Oui, passez `max_size` à CORSMiddleware.

---

**✅ Prêt à commencer? Lancez `start_api.bat` (ou `.sh`) maintenant!**
