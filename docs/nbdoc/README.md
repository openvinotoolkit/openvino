# Jupyter notebooks autodoc

Auto fetching documentations designed for openvino notebooks tutorials.
This module is responsible for fetching artifacts, in this particular example jupyter tutorial notebooks and converting them to notebook documentation.

## Step 0. Prepare venv

To start new venv based on system interpreter

``` bash
python -m venv ./venv
```

Linux:

``` bash
source ./venv/bin/activate
```

Windows (cmd):

``` bash
venv/Scripts/activate
```

## Step 1. Download requirements

``` bash
python -m pip install -r requirements.txt
```

## Step 2. Configure consts to meet project directions

[Consts file](consts.py) contains multiple variables that might differ for different environments.

## Step 3. Add classes with methods to makefile or other executed file

[Main file](main.py) contains example usecases of auto generator for notebooks. Informations placed in [main](main.py) should be further used to add it to makefile and possibly fully automate notebook documentation process.

## Step 4. Run python file (optional)

If step 4 was skipped use command

``` bash
python main.py
```

or use any other command that is responsible for generating documentation
