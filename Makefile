ENV_NAME = plora

run:
	conda run -n $(ENV_NAME) python3 -m experiments.run_probe

clean:
	rm -rf results/logs/*
	rm -rf results/figures/*

install:
	conda run -n $(ENV_NAME) pip install -r requirements.txt
