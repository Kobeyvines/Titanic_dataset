# Makefile for Titanic Survival Project
# -------------------------------------------------------------------------

# Python Interpreter (Uses venv if it exists, otherwise system python)
PYTHON := python3
PIP := pip3

# Docker Image Name
IMAGE_NAME := titanic-app
CONTAINER_NAME := titanic-container

# =========================================================================
# 🛠️ SETUP & INSTALLATION
# =========================================================================

.PHONY: help install clean

help:		## Show this help message
	@echo "Titanic Project Makefile"
	@echo "-----------------------"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:	## Install dependencies
	$(PIP) install -r requirements.txt

clean:		## Remove cache files and build artifacts
	rm -rf __pycache__ .pytest_cache
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	rm -rf .coverage htmlcov

# =========================================================================
# 🚀 LOCAL DEVELOPMENT (No Docker)
# =========================================================================

.PHONY: run train predict format lint

run:		## Run the API locally (Reload mode)
	uvicorn api.index:app --reload --host 0.0.0.0 --port 8000

train:		## Train the model locally
	$(PYTHON) -m src.pipeline.train_pipeline

predict:	## Run a sample prediction (Test script)
	$(PYTHON) -m src.pipeline.predict_pipeline

submit:		## Generate Kaggle submission file
	$(PYTHON) -m src.pipeline.generate_submission

format:		## Format code using Black
	black .

lint:		## Check code style using Flake8
	flake8 src api

# =========================================================================
# 🐳 DOCKER COMMANDS
# =========================================================================

.PHONY: build up down shell

build:		## Build the Docker image
	docker build -t $(IMAGE_NAME) .

up:		## Run the container on port 8000
	docker run -p 8000:8000 --name $(CONTAINER_NAME) --rm $(IMAGE_NAME)

down:		## Stop the running container
	docker stop $(CONTAINER_NAME)

shell:		## Enter the container's shell (for debugging)
	docker run -it --rm $(IMAGE_NAME) /bin/bash

test-prod:	## Run the training pipeline INSIDE Docker
	docker run --rm -v $(PWD)/models:/app/models $(IMAGE_NAME) python -m src.pipeline.train_pipeline
