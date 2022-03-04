# nlp-pico-snorkel
2/4 Updates: Got snorkel and BERT running on server, spent a lot of time on this. Snorkel is currently running, should have results by morning. I will need more time to understand BERT to do the error analysis.
2/25 Got BERT working!!!! WooHOOOO!
2/18 Data is consistent now.
Still working... just finished
2/14 Progress:
- got code running on GPU
- need to debug some of the metric calculations, but model is being trained

2/11 Action Items:
- Logging into the GPU. (done)
- Running the supervised model with PICO on the GPU. (not done)
    - Currently calling IT support (resolved)
    - Currently installing some things like jupyter in my directory (is this ok?)
    - Creating a virtual env currently
        - export PATH=/share/apps/anaconda3/2021.05/bin:$PATH
    - asking IT for help with GPU access still
    - Currently trying to run jupyter file
    - Currently stuck with an error while running .ipynb on GPU
    - Gn
 

Current progress:
- (done) reading Bojian's paper. 
- (done) looking at his code.
- (done) looking for a BERT tutorial
- (done) finished BERT implementation



**new_snorkel_pico.ipynb** is applying Snorkel to the EBM-NLP PICO dataset.

**supervised_learning.ipynb** is the baseline BERT supervised learning model on the same intervention labeling task with the same dataset.
