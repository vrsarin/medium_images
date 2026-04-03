.PHONY: install serve build clean open start

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