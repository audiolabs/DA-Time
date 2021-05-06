# DA-Time Temporal Tagger

a domain-adapted temporal expression recognizer for the English Voice Assistant domain

### Citing DA-Time

To refer to DA-Time in any publication, please cite the following paper: 

Alam, T., Zarcone, A., & Padó, S. (2021). New Domain, Major Effort? How Much Data is Necessary to Adapt a Temporal Tagger To the Voice Assistant Domain. In Proceedings of the 14th International Conference on Computational Semantics (IWCS 2021).

    @inproceedings{alam2021:new,
        author       = {Touhidul Alam and Alessandra Zarcone and Sebastian Pad\'{o}},
        title        = {{New Domain, Major Effort? How Much Data is Necessary to Adapt a Temporal Tagger To the Voice Assistant Domain}},
        year         = 2021,
        booktitle    = {{Proceedings of the 14th International Conference on Computational Semantics (IWCS 2021)}}
        }

### Getting Started

#### Installation Guide

* Clone the repository
    ```
    git clone https://github.com/audiolabs/DA-Time.git
    ```
* Install Python >= 3.7.3
* Make sure to have virtualenv installed by executing
    ```
    python -m pip install --user virtualenv
    ```
* Create the virtual environment
    ```
    python -m venv <path-to-env>
    ```
* Source the environment
    ```
    source <path-to-env>/bin/activate
    ```
* Install the following packages
    ```
    pip install -r requirements.txt
    ```
* Download Spacy English model
    ```
    python -m spacy download en_core_web_sm
    ```
* Download Stanford CoreNLP 4.0.0 (English model)
    ```
    https://stanfordnlp.github.io/CoreNLP/download.html
    ```

#### Running Guide

* Make sure to check the necessary experimental setup from the config.yaml file.
* To run the model with different mode (Training/testing/fine-tuning) run the following command
    ```
    python main.py exec --mode=<train/test/ft>
    ```
* [Optional] Parameter for multiple run (these will overwrite configs value)
    ```
    python main.py exec --mode=<train/test/ft> [--run=<run>] [--amount=<amount>] [--emb=<emb>]
    ```
    Here, run indicates #of run, amount is to define fine-tune data amount%, emb is the model embedding. To run multiple times, edit and use the script:
    ```
    ./RUN.sh
    ```
* To plot different data statistics run the following command
    ```
    python main.py plot
    ```
* To predict on a trained model, run the following command with n-number of sentences as argument
    ```
    python main.py predict "when is our tomorrow's meeting?" "Make a schedule from Friday to Monday"
    ```

* To simplify text with the CoreNLP run the following command in terminal 
    ```
    java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000000
    ```
* In parallel, run the simplifier script with this commands:
    ```
    cd recognition
    python simplifier.py
    ```

