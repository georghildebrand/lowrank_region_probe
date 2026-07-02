ENV_NAME = plora

run:
	conda run -n $(ENV_NAME) python3 -m experiments.run_probe

structural-cells: run

sweep:
	conda run -n $(ENV_NAME) python3 -m experiments.run_sweep

label-shuffle:
	conda run -n $(ENV_NAME) python3 -m experiments.run_label_shuffle

gmm:
	conda run -n $(ENV_NAME) python3 -m experiments.run_gmm

capacity-ratio:
	conda run -n $(ENV_NAME) python3 -m experiments.run_capacity_ratio

clean:
	rm -rf results/logs/*
	rm -rf results/figures/*

install:
	conda run -n $(ENV_NAME) pip install -r requirements.txt
