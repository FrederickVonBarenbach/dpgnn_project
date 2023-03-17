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
