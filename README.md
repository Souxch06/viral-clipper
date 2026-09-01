## Installer l'application PWA sur Android (Guide rapide)

Pour pouvoir installer et tester l'application sur un appareil Android (Chrome), suis ces étapes.

Pré-requis
- Le backend & frontend doivent être accessibles via HTTPS (Chrome exige HTTPS pour l'installation PWA et le Web Share Target).
- Pour un test local rapide, tu peux utiliser ngrok pour exposer ton localhost en HTTPS.

1) Lancer les services en local
- Depuis la racine du repo (où se trouve docker-compose.yml) :

  mkdir -p storage
  docker-compose up --build

- Le frontend de demo est basé sur Next.js (dossier `frontend`). Pour développement local hors Docker tu peux lancer :

  cd frontend
  npm install
  npm run dev

2) Exposer l'app en HTTPS (option A : ngrok)
- Installer ngrok (https://ngrok.com/) et run :

  ngrok http 3000

- Copie l'URL HTTPS fournie par ngrok (ex: https://abcd1234.ngrok.io)

3) Tester l'installation PWA sur Android
- Ouvre Chrome sur ton appareil Android et va à l'URL HTTPS fournie.
- Chrome détectera la PWA si le manifeste et le service worker sont présents. Appuie sur le menu (⋮) → "Ajouter à l'écran d'accueil".
- L'application sera installée et disponible depuis l'écran d'accueil.

4) Tester le partage depuis YouTube (Web Share Target)
- Sur Android, ouvre l'app YouTube, choisis une vidéo, appuie sur "Partager" → sélectionne le navigateur (Chrome) → dans la liste des cibles tu devrais voir ton PWA (si installée) comme destination nommée "ViralClipper" ; en la sélectionnant, Chrome ouvrira la route /api/share-target et la PWA recevra l'URL (la page d'accueil se remplira automatiquement via ?shared=...)

Si le Web Share Target ne s'affiche pas :
- Vérifie que la PWA est installée (pas seulement ouverte dans l'onglet)
- Vérifie que ton manifest.json inclut "share_target" (il est déjà présent dans le projet)

5) Déploiement en production (option B : Caddy / VPS)
- Pour un déploiement réel, héberge le frontend et backend sur un VPS avec un nom de domaine et un reverse-proxy HTTPS (Caddy, Nginx + Certbot). Caddy est recommandé pour la simplicité (SSL automatique).
- Exemple de config Caddy (sommaire) :

  your-domain.tld {
    reverse_proxy /api* localhost:8000
    reverse_proxy localhost:3000
  }

6) Notes & limitations
- Les icônes SVG incluses sont des placeholders. Pour un rendu optimal sur l'écran d'accueil Android, ajoute des PNG 192x192 et 512x512 dans `frontend/public/icons/` et met à jour `manifest.json` si nécessaire.
- Le partage depuis YouTube via la PWA nécessite que la PWA soit bien installée (Add to Home). Le share_target fonctionne uniquement si l'app est installée et si l'origine (domaine) est HTTPS.
- Le backend FastAPI (docker) par défaut écoute sur le port 8000; le frontend Next.js sur 3000. Lorsque tu exposes avec ngrok, assure-toi d'exposer le port du frontend (3000) pour tester la PWA.

Besoin d'aide pour automatiser la génération des icônes ou déployer sur un VPS avec un domaine ? Dis-le moi, je peux automatiser le déploiement Docker + Caddy ou préparer des scripts Terraform/Ansible selon ton infra.
