def classify_email(email_text: str) -> str:
    text = email_text.lower()

    if any(word in text for word in [
        "facture", "paiement", "contrat", "caf", "impôts", "sécurité sociale",
        "banque", "attestation", "document administratif", "administration"
    ]):
        return "Administratif"

    if any(word in text for word in [
        "travail", "réunion", "meeting", "client", "projet", "collaboration",
        "entreprise", "stage", "emploi", "internship", "professional"
    ]):
        return "Professionnel"

    if any(word in text for word in [
        "cours", "classe", "prof", "université", "sorbonne", "devoir",
        "assignment", "exam", "étude", "formation", "bootcamp"
    ]):
        return "Académique"

    if any(word in text for word in [
        "famille", "ami", "soirée", "dîner", "weekend", "vacances",
        "anniversaire", "personnel", "hello", "salut", "coucou"
    ]):
        return "Personnel"

    return "Autre"
    if any(word in text for word in [
        "famille", "ami", "soirée", "weekend", "vacances",
        "anniversaire", "personnel", "salut", "coucou"
    ]):
        return "Personnel"

    return "Autre"