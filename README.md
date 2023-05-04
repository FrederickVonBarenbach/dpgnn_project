# DPGNN Project

To set up the project execute the following code

```
git clone https://github.com/FrederickVonBarenbach/dpgnn_project.git
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

## Compute Canada

<br />

To run a batch on Compute Canada, run ```cc_executor.sh <start> <end> <command_file>``` where ```start``` and ```end``` are the lines of the experiments you want to run from ```command_file``` (which will be generated automatically when you run ```make_runner.py``` with the ```compute_canada``` flag).

To set up the virtual environment run the following commands in the login node. For all packages that are not included in Compute Canada's wheels, run

```
module load python/3.9
pip download <package>
pip install <path-to-downloaded-package>
```

Then, set up your requirements by installing any packages included in [Compute Canada's wheels](https://docs.alliancecan.ca/wiki/Available_Python_wheels) (replace the ```<>``` part),

```
module load python/3.9
ENVDIR=/tmp/$RANDOM
virtualenv --no-download $ENVDIR
source $ENVDIR/bin/activate
pip install --no-index --upgrade pip
pip install --no-index <packages-included-in-compute-canada>
pip freeze --local > requirements.txt
deactivate
rm -rf $ENVDIR
```

In a compute node, you can then run the commands

```
module load python/3.9
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt
```

to quickly get your virtual environment set up from ```requirements.txt``` without any downloads.
