#!/usr/bin/env python3
"""
Generate semantic search embeddings for all Jekyll articles.

Usage:
    python generate_embeddings.py

Output:
    assets/search-index.json  — array of { url, title, description, embedding }

The model (all-MiniLM-L6-v2, ~90 MB) is downloaded once and cached by
sentence-transformers. Embeddings are L2-normalised so dot product equals
cosine similarity — no division needed in the browser.
"""

import os
import re
import json
import glob

import frontmatter
from sentence_transformers import SentenceTransformer


# ── Text helpers ────────────────────────────────────────────────────────────

def strip_markdown(text: str) -> str:
    """Remove common markdown syntax for cleaner embedding input."""
    text = re.sub(r'\$\$.*?\$\$', ' ', text, flags=re.DOTALL)   # display math
    text = re.sub(r'\$[^\$\n]+\$', ' ', text)                    # inline math
    text = re.sub(r'```.*?```', ' ', text, flags=re.DOTALL)      # fenced code
    text = re.sub(r'`[^`]+`', ' ', text)                         # inline code
    text = re.sub(r'#{1,6}\s+', '', text)                        # headings
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)                 # bold
    text = re.sub(r'\*(.+?)\*', r'\1', text)                     # italic
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)        # links
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)                  # images
    text = re.sub(r'^\s*\|.*\|\s*$', '', text, flags=re.MULTILINE)  # tables
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE) # lists
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def build_embed_text(title: str, description: str, body: str) -> str:
    """
    Weight title and description more heavily by repeating them.
    Truncate body to keep embedding time reasonable.
    """
    clean_body = strip_markdown(body)[:1500]
    return f"{title}\n{title}\n{description}\n{description}\n{clean_body}"


# ── Main ────────────────────────────────────────────────────────────────────

SKIP_FILES = {'index.md', 'README.md', 'readme.md'}

def main():
    md_files = [
        f for f in sorted(glob.glob('*.md'))
        if f not in SKIP_FILES
    ]

    if not md_files:
        print("No article .md files found in the current directory.")
        return

    articles = []
    for filepath in md_files:
        with open(filepath, encoding='utf-8') as fh:
            post = frontmatter.load(fh)

        title       = str(post.get('title', ''))
        description = str(post.get('description', ''))
        permalink   = str(post.get('permalink', '/' + filepath.replace('.md', '/').lstrip('/')))

        articles.append({
            'url':         permalink,
            'title':       title,
            'description': description,
            '_embed':      build_embed_text(title, description, post.content),
        })

    print(f"Found {len(articles)} article(s).")
    print("Loading model sentence-transformers/all-MiniLM-L6-v2 …")

    model = SentenceTransformer('all-MiniLM-L6-v2')

    texts = [a.pop('_embed') for a in articles]

    print("Generating embeddings …")
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,   # L2-normalise → dot product = cosine sim
        batch_size=32,
        show_progress_bar=True,
    )

    output = [
        {**article, 'embedding': emb.tolist()}
        for article, emb in zip(articles, embeddings)
    ]

    os.makedirs('assets', exist_ok=True)
    out_path = os.path.join('assets', 'search-index.json')
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(output, fh, separators=(',', ':'), ensure_ascii=False)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"Saved {out_path} ({size_kb:.1f} KB) — {len(output)} article(s).")


if __name__ == '__main__':
    main()
