.PHONY: install serve build clean open start embeddings pip-install

install:
	bundle install

serve:
	bundle exec jekyll serve --livereload --baseurl ""

build:
	bundle exec jekyll build

clean:
	bundle exec jekyll clean

open:
	start http://localhost:4000

start: serve open

pip-install:
	uv sync

embeddings:
	uv run generate_embeddings.py