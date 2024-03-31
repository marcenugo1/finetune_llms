

# Exploring Large Language Models (LLMs) for Text Generation with PyTorch and Hugging Face

Implementation of Large Language Models (LLMs) for code generative purposes using PyTorch and Hugging Face's Transformers library. 
Fine tune models: 
- LaMa-2(7B)
- Phi-2 (2.7B)
- Mistral (7B).

#### Requirements

Instructions to install libraries using *requirements.txt* file.

```shell
cd FineTune 
pip install -r requirements.txt
```


### Training and Evaluation

For training, please refer to the notebooks with the "fine_tune" prefix in their file name. For Evaluation, refer to the notebooks with "eval" prefix in their file name. 

Llama 2 
Eval: llama2_eval.ipynb
Finetune:llama2_fine_tune.ipynb

Mistral 
Eval: mistral_eval.ipynb
Finetune:mistral_fine_tune.ipynb

Phi 2
Eval: phi_eval2.ipynb
Finetune:phi_2_finetune.ipynb

For Metric Evaluation documentantion please refer to file: 

### Datasets

- Dataset used for fine tuning with python code: [Dataset](https://huggingface.co/datasets/flytech/python-codes-25k)

## License
As a free open-source implementation, our repository is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. All other warranties including, but not limited to, merchantability and fitness for purpose, whether express, implied, or arising by operation of law, course of dealing, or trade usage are hereby disclaimed. I believe that the programs compute what I claim they compute, but I do not guarantee this. The programs may be poorly and inconsistently documented and may contain undocumented components, features or modifications. I make no guarantee that these programs will be suitable for any application.
