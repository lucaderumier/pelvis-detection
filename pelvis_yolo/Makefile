PYTHON=python3

DATA_PATH=pelvis_scan/data/
EXTRACT_PATH=pelvis_scan/FULL_IMAGES_CT/
3D_IMAGE_PATH=charleroi_1/
SET=train/
WEIGHTS=''
RESULTS_DIR=results/
LOGS=training.log
ANNOT_FILE=annotations_train.p
FILE=pelvis_data_train.npz
STAGE_DIR=stage_X/
SAVE_DIR=
#STEPS = 0 1 2 3 4 5 6 7 8 9 10 11 12 13  14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39
STEPS = 1 3 4 5 6 7 8 9 15 16 17 18 19 20 21 23 24 25 26 27 30 31 32 34 35 36 37 38 39 40 41 42 43 45 46 47 48 49 51 52 53 54 55 56 58 59 60 61 62 64 65 66 67 70 71 72 74 75 77 79 80 81 82 83 84 87 88 89 90 91 92 93 97 98 99 100


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

extract:
	$(PYTHON) extract.py 

analysis:
	make pred
	make eval
	make draw
	mkdir -p stages/$(SET)$(STAGE_DIR)
	mv $(RESULTS_DIR)* stages/$(SET)$(STAGE_DIR)

allpredict:
	$(foreach var,$(STEPS),$(PYTHON) prediction.py --weights $(WEIGHTS) --data_path $(EXTRACT_PATH)charleroi_$(var)/ --file pelvis_data.npz --training_path $(DATA_PATH)train --results_dir $(EXTRACT_PATH)charleroi_$(var)/ ;)

allval:
	 $(foreach var,$(STEPS),make analysis DATA_PATH=pelvis_scan/data/ WEIGHTS=trained_stage_$(var).h5 STAGE_DIR=stage$(var)/ SET=val/ FILE=pelvis_data_val.npz ANNOT_FILE=annotations_val.p ;)

alltrain:
	$(foreach var,$(STEPS),make analysis DATA_PATH=pelvis_scan/data/ WEIGHTS=trained_stage_$(var).h5 STAGE_DIR=stage$(var)/ SET=train/ FILE=pelvis_data_train.npz ANNOT_FILE=annotations_train.p ;)
