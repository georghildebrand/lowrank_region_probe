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

region-identity:
	conda run -n $(ENV_NAME) python3 -m experiments.run_region_identity

real-data:
	conda run -n $(ENV_NAME) python3 -m experiments.run_real_data

lora-prediction:
	conda run -n $(ENV_NAME) python3 -m experiments.run_lora_prediction

directional-probe:
	conda run -n $(ENV_NAME) python3 -m experiments.run_directional_probe

lora-sweep:
	conda run -n $(ENV_NAME) python3 -m experiments.run_lora_sweep

multilayer-lora:
	conda run -n $(ENV_NAME) python3 -m experiments.run_multilayer_lora

polytopeness:
	conda run -n $(ENV_NAME) python3 -m experiments.run_polytopeness

functional-flips:
	conda run -n $(ENV_NAME) python3 -m experiments.run_functional_flips

clean:
	rm -rf results/logs/*
	rm -rf results/figures/*

install:
	conda run -n $(ENV_NAME) pip install -r requirements.txt
