# Data annotation competition, Naver Boostcamp AI Tech 2κΈ°
## Competition Abstract
π€ Relation Extractions taskμ μ¬μ©ν λ°μ΄ν°λ₯Ό μ§μ  μ μνλ task.
π€ Relation set μ μ, κ°μ΄λλΌμΈ μμ±, Pilot/Main annotation, Model Fine-tuning μ§ν

## [Team Portfolio]()
## [Competition Report(PDF)]()
## Reulst
- [Relation map](https://docs.google.com/spreadsheets/d/1rXz57Yxs80HhgqP2W4A016liz0dfz2a4_sZE96uokGM/edit)
- [Guideline](https://docs.google.com/document/d/1qDx4riQMJLYZf97j8BPY0FGs31iv_klXg2YztoJAD_g/edit)
## Quickstart
### Installation
```
pip install -r requirements.txt
```
### Train model
```python
# default wandb setting in train.py
run = wandb.init(project= 'dataset', entity= 'quarter100', name= f'KFOLD_{fold}_{args.wandb_path}')
```

```
python train.py
```
### Inference
```
python inference_fold.py
```
Prediction csv files are saved in "./prediction".
