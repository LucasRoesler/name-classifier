# Classify Name

### Training
After adding more datat to `/classify/function/data/names`, you can retrain the model using

```bash
cd classify/
python train.py
cd -
```


### Build

```bash
faas-cli template pull https://github.com/LucasRoesler/pydatascience-template.git
faas-cli build classify --tag sha
```

### Quick test

```bash
docker run -it -p 8080:8080 theaxer/classify:latest
curl -X POST http://localhost:8080 -d "Bob"
```

### Deploy

```bash
conda env create -f devenvironment.yml
faas-cli deploy -f ./stack.yml --gateway=<GATEWAY> --tag sha
```
