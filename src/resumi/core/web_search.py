from typing import Any
import requests
def web_search(query: str) -> str:
    try:
        # 🔥 Nettoyage de la requête (très important)
        cleaned_query = query.lower()

        for phrase in [
            "cherche sur le web",
            "cherche sur internet",
            "recherche web",
            "web search",
            "search the web",
            "search online",
        ]:
            cleaned_query = cleaned_query.replace(phrase, "")

        cleaned_query = cleaned_query.strip()

        # fallback si nettoyage trop agressif
        if not cleaned_query:
            cleaned_query = query

        url = "https://api.duckduckgo.com/"
        params = {
            "q": cleaned_query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1,
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data: dict[str, Any] = response.json()

        # 🔥 1. Résumé principal
        abstract = data.get("AbstractText")
        if isinstance(abstract, str) and abstract.strip():
            return abstract.strip()

        # 🔥 2. Related topics
        related = data.get("RelatedTopics")
        if isinstance(related, list):
            for item in related:
                if isinstance(item, dict):
                    text = item.get("Text")
                    if isinstance(text, str) and text.strip():
                        return text.strip()

        # 🔥 3. Fallback intelligent
        return (
            f"Je n’ai pas trouvé de réponse directe via DuckDuckGo pour '{cleaned_query}'. "
            "Mais voici une réponse générale :\n\n"
            f"{cleaned_query.capitalize()} est une personnalité publique. "
            "Essaie de poser une question plus précise (ex: date de naissance, rôle, actualité)."
        )

    except Exception as e:
         return f"Erreur web: {str(e)}"