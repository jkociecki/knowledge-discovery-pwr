IMAGE_NAME = pdiow-lab1

build:
	docker build -t $(IMAGE_NAME) .

run_main_1:
	docker run --rm $(IMAGE_NAME) python scripts/main_1.py

run_main_2:
	docker run --rm $(IMAGE_NAME) python scripts/main_2.py

format:
	docker run --rm -v $(PWD):/app $(IMAGE_NAME) ruff format .

run_jupyter:
	docker run --rm -p 8888:8888 $(IMAGE_NAME) jupyter lab --ip=0.0.0.0 --allow-root --no-browser