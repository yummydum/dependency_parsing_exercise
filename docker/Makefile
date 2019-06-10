IMG             := pytorch
TAG             := $(shell git log -1 --pretty=%H)
REMOTE_REPO     := atsushisumita0421
CONTAINER       := ${IMG}_container
PROJECT_ROOT    := $(shell cd ../ && pwd)

.PHONY: build run run_persist run_notebook commit push

build:
	@docker build -t ${IMG}:${TAG} .
	@docker tag ${IMG}:${TAG} ${IMG}:latest

run:
	@docker run -it -v ${PROJECT_ROOT}:/app --rm --name=${CONTAINER} ${IMG}:latest /bin/bash

run_persist:
	@docker run -it -v $(PROJECT_ROOT):/app --name=${CONTAINER} ${IMG}:latest /bin/bash

run_notebook:
	@docker run -it -v $(PROJECT_ROOT)\:/app -p 8888\:8888 --name=${CONTAINER} ${IMG}\:latest /bin/bash

commit:
	@docker commit $(CONTAINER) $(IMG):${TAG}
	@docker tag ${IMG}:${TAG} ${IMG}:latest

push:
	@docker push ${IMG}:${TAG}
	@docker push ${IMG}:latest
