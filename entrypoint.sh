#!/bin/bash
set -e

# If OPENAI_API_KEY is missing or still the placeholder, prompt for it.
if [[ -z "$OPENAI_API_KEY" || "$OPENAI_API_KEY" == "sk-your-key-here" ]]; then
    echo ""
    echo -e "\033[38;5;135m ____                            _ \033[0m"
    echo -e "\033[38;5;135m|  _ \\ ___  ___ _   _ _ __ ___ (_)\033[0m"
    echo -e "\033[38;5;177m| |_) / _ \\/ __| | | | '_ \` _ \\| |\033[0m"
    echo -e "\033[38;5;177m|  _ <  __/\\__ \\ |_| | | | | | | |\033[0m"
    echo -e "\033[1;97m|_| \\_\\___||___/\\__,_|_| |_| |_|_|\033[0m"
    echo ""
    echo -e "\033[2m  Personal assistant · RAG · Gmail · Calendar\033[0m"
    echo ""

    # Interactive mode: prompt for the key
    if [[ -t 0 ]]; then
        echo -e "\033[38;5;177m⚙  Aucune clé OpenAI détectée.\033[0m"
        read -rp "   Entre ta clé OpenAI (sk-...): " OPENAI_API_KEY
        export OPENAI_API_KEY
        echo ""

        if [[ -z "$OPENAI_API_KEY" ]]; then
            echo -e "\033[31m✗ Clé vide — l'app risque de ne pas fonctionner correctement.\033[0m"
        else
            echo -e "\033[32m✓ Clé enregistrée pour cette session.\033[0m"
        fi
        echo ""
    else
        echo -e "\033[33m⚠  OPENAI_API_KEY non définie. Lance le conteneur avec :\033[0m"
        echo ""
        echo "   docker run -it --env-file .env andrealoy/resumi:latest"
        echo "   # ou"
        echo "   docker run -it -e OPENAI_API_KEY=sk-... andrealoy/resumi:latest"
        echo ""
    fi
fi

exec "$@"
