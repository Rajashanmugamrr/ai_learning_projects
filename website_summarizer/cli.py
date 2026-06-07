import json

from summarizer import summarize_website


DEFAULT_MODEL = "llama3.2:3b"


def main() -> None:
    url = input("Enter the website URL to summarize: ").strip() or "https://edwarddonner.com/"
    model = input(f"Enter the Ollama model to use (default: {DEFAULT_MODEL}): ").strip() or DEFAULT_MODEL

    result = summarize_website(url, model=model)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
