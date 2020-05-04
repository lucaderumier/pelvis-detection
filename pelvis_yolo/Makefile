PYTHON=python3

DATA_PATH=pelvis_scan/data/
SET=train/
WEIGHTS=''
RESULTS_DIR=results/
LOGS=training.log
ANNOT_FILE=annotations_train.p
FILE=pelvis_data_train.npz
STAGE_DIR=stage_X/
SAVE_DIR=


.DEFAULT: help
help:
	@echo "make dataset"
	@echo "       launches the dataset transformation phase."
	@echo "make train"
	@echo "       launches the training phase."
	@echo "make pred"
	@echo "       launches the prediction phase."
	@echo "make eval"
	@echo "       launches the evaluation phase."
	@echo "make draw"
	@echo "       launches the visualization phase."
	@echo "make hist"
	@echo "       launches the history plotting phase."
	@echo "make analysis"
	@echo "       launches the prediction, evaluation and prediction phases and saves the results in specified directory."


dataset:
	$(PYTHON) dataset_utils.py --data_path $(DATA_PATH)

train:
	$(PYTHON) training.py --data_path $(DATA_PATH) --weights $(WEIGHTS) | tee $(RESULTS_DIR)$(LOGS)

pred:
	$(PYTHON) prediction.py --weights $(WEIGHTS) --data_path $(DATA_PATH)$(SET) --file $(FILE) --training_path $(DATA_PATH)train --results_dir $(RESULTS_DIR)

eval:
	$(PYTHON) evaluation.py --data_path $(DATA_PATH)$(SET) --annot_file=$(ANNOT_FILE) --results_dir $(RESULTS_DIR)

draw:
	$(PYTHON) visualize.py --data_path $(DATA_PATH)$(SET) --annot_file=$(ANNOT_FILE) --results_dir $(RESULTS_DIR) -V

hist:
	$(PYTHON) visualize.py --data_path $(DATA_PATH)$(SET) --annot_file=$(ANNOT_FILE) --results_dir $(RESULTS_DIR) -H

analysis:
	make pred
	make eval
	make draw
	mkdir -p stages/$(SET)$(STAGE_DIR)
	mv $(RESULTS_DIR)* stages/$(SET)$(STAGE_DIR)
