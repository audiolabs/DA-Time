## RESULT & OUTPUT

This folder stores the results and output of different model with their experiments.

Folder structures are as follow:

Folder name | Experiment | Model | Data | Method | Embedding |
------------|------------|-------|------|--------|-----------|
FT2_Simplfied_Snips_Chain | Fine-tune | Simplified | Snips | Chain | English
FT3_Simplified_Pate_Full | Fine-tune | Simplified | PATE | Full | English
FT3_Original_Pate_Chain | Fine-tune | Original | PATE | Chain| English
FT3_Snips+PATE | Fine-tune | Simplified | Snips + PATE | Chain | English
FT3_Simplified_Pate_Chain | Fine-tune | Simplified | PATE | Chain | English
Tempeval3_BS32_ML50 | Train | Original | TE-3/PATE | - | English
TempEval3Simplified_ML_30 | Train | Simplified | TE-3/PATE | - | English 
FT_German_Original_Chain_de | Fine-tune | Original | German PATE | Chain | German
FT_German_Original_Chain_mul | Fine-tune | Original | German PATE | Chain | Multilingual

Additionally, 

* norma_output_en.txt - contains normalization output for one model
* results_en.csv - contains different experimental results for EN model
* results_de.csv - contains different experimental results for DE model
* timex_output.json - contains prediction output in a json format