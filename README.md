# Classify Name

### Training
You can retrain the model using

```bash
cd classify/function/core/
python train.py
cd -
```


### Build

```bash
faas-cli build classify --tag sha
```

### Deploy

```bash
conda env create -f devenvironment.yml
faas-cli deploy -f ./stack.yml --gateway=<GATEWAY> --tag sha
```