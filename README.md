The enviroment is based on Cross Image Attention

Useage:
conda env create -f environment/environment.yaml
conda activate M3S
pip install -r requirements.txt

Change the pretrained stable diffusion model path if you need:
cd SDv1.5 (or SDXL)
vim utils/model_utils.py

Then, please run in python:
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

To generate sketches:
cd SDv1.5 (or SDXL)
bash test.sh
You can change the parameters in test.sh for different styles, contents, and style tendencies.

To reproduce the qualitative results:
cd SDv1.5 (or SDXL)
python -u Generation_demo.py

To evaluate the results:
cd SDv1.5 (or SDXL)
cd notebooks
run the jupyter notebook metrics_test.ipynb# M3S
# M3S
