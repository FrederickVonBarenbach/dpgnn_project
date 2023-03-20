# DPGNN Project

To set up the project execute the following code

```
git clone https://FrederickVonBarenbach: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@github.com/FrederickVonBarenbach/dpgnn_project.git
cd dpgnn_project
python -m venv .venv && source .venv/bin/activate
python setup.py
```

If you perfer to use ```cpu```, you can use

```
python setup.py --device cpu
```

In the ```configs``` folder, you can change the base hyperparameters of your experiments by creating a JSON file and executing ```make_runner.py```. Refer to the ```README.md``` there for more information.

Now you can run your experiments with

```
python main.py
```
