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
#STEPS = 0 1 2 3 4 5 6 7 8 9 10 11 12 13  14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35
STEPS = 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33


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
	$(PYTHON) training.py --data_path $(DATA_PATH) --weights $(WEIGHTS) | tee $(LOGS)

pred:
	$(PYTHON) prediction.py --weights $(WEIGHTS) --data_path $(DATA_PATH)$(SET) --file $(FILE) --training_path $(DATA_PATH)train --results_dir $(RESULTS_DIR)

eval:
	$(PYTHON) evaluation.py --data_path $(DATA_PATH)$(SET) --annot_file=$(ANNOT_FILE) --results_dir $(RESULTS_DIR)

draw:
	$(PYTHON) visualize.py --data_path $(DATA_PATH)$(SET) --annot_file=$(ANNOT_FILE) --results_dir $(RESULTS_DIR) -V

hist:
	$(PYTHON) visualize.py --results_dir $(RESULTS_DIR) -H

graphs:
	$(PYTHON) graphs.py

analysis:
	make pred
	make eval
	make draw
	mkdir -p stages/$(SET)$(STAGE_DIR)
	mv $(RESULTS_DIR)* stages/$(SET)$(STAGE_DIR)

allval:
	 $(foreach var,$(STEPS),make analysis DATA_PATH=pelvis_scan/data/ WEIGHTS=trained_stage_$(var).h5 STAGE_DIR=stage$(var)/ SET=val/ FILE=pelvis_data_val.npz ANNOT_FILE=annotations_val.p ;)

alltrain:
	$(foreach var,$(STEPS),make analysis DATA_PATH=pelvis_scan/data/ WEIGHTS=trained_stage_$(var).h5 STAGE_DIR=stage$(var)/ SET=train/ FILE=pelvis_data_train.npz ANNOT_FILE=annotations_train.p ;)
