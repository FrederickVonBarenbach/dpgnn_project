# DPGNN Project

To set up the project execute the following code

```
git clone https://FrederickVonBarenbach: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@github.com/FrederickVonBarenbach/dpgnn_project.git
cd dpgnn_project
python -m venv .venv && source .venv/bin/activate
python setup.py
```

In the ```config.py``` file in the ```configs``` folder, you can change the base hyperparameters. Then, you can define your experiments by identifying the new hyperparameters as a list.

Now you can run your experiments with

```
python main.py
```
