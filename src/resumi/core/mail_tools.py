"""Mail tools – classify and draft replies via LLM."""

from openai import AsyncOpenAI

from resumi.core.document_store import DocumentStore


async def classify_email(
    client: AsyncOpenAI, model: str, email_text: str
) -> dict[str, str]:
    """Classify an email into a category with priority using the LLM."""
    resp = await client.responses.create(
        model=model,
        instructions=(
            "Tu es un assistant de classification d'emails. "
            "Tu dois classer l'email dans UNE seule catégorie parmi : "
            "Personnel, Professionnel, Administratif, Académique, Commercial, Autre. "
            "Tu dois aussi proposer un niveau de priorité "
            "parmi : basse, moyenne, haute. "
            "Réponds uniquement en JSON valide avec les clés : "
            "category, priority, reason."
        ),
        input=f"Email à analyser :\n\n{email_text}",
    )
    return {"raw_result": resp.output_text.strip()}


async def classify_and_store(
    client: AsyncOpenAI,
    model: str,
    store: DocumentStore,
    doc_id: int,
    title: str,
) -> str:
    """Classify a mail by title and persist the category. Returns the category."""
    resp = await client.responses.create(
        model=model,
        instructions=(
            "Tu es un assistant de classification d'emails. "
            "Classe cet email dans UNE seule catégorie parmi : "
            "Personnel, Professionnel, Administratif, Académique, Commercial, Autre. "
            "Réponds avec UN seul mot : la catégorie."
        ),
        input=f"Sujet de l'email : {title}",
    )
    category = resp.output_text.strip().split("\n")[0].strip('" .')
    store.update_category(doc_id, category)
    return category


async def draft_email_reply(client: AsyncOpenAI, model: str, email_text: str) -> str:
    """Generate a polite, professional reply draft for the given email."""
    resp = await client.responses.create(
        model=model,
        instructions=(
            "Tu es un assistant de rédaction d'emails. "
            "Rédige un brouillon de réponse en français, "
            "clair, poli, concis et naturel. "
            "N'invente pas d'informations précises qui ne figurent pas dans l'email. "
            "Le ton doit être professionnel mais humain. "
            "Retourne uniquement le texte du brouillon."
        ),
        input=(f"Voici l'email reçu :\n\n{email_text}\n\nRédige une réponse adaptée."),
    )
    text = resp.output_text.strip()
    return text or "Impossible de générer un brouillon pour le moment."
