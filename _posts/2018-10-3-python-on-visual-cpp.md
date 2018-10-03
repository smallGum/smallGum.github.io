---
layout: post
title: Run Python Model on Visual Studio (C#)
description: "A tutorial of using Python on Visual Studio"
modified: 2018-10-03
tags: [Programming Tricks]
image:
  feature: abstract-10.jpg
---

## Introduction

Recently, I'm working on a C# project from Visual Studio which requires using `statsmodels.tsa.regime_switching.markov_autoregression.MarkovAutoregression` model of Python to do data analysis. One method is to package the model into *Dynamic Link Library* file (i.e. the `.dll` file on Windows) so that our C# programs can call the functions through importing the DLL file. There are three steps:

1. Call the Python scripts on C++ from Visual Studio
2. Package `.cpp` files of step 1 into DLL file
3. Import the DLL file of step 2 and use the model on C# programs

This blog will show the entire process of building our project.

## Environment

+ OS: Windows10 64-bit x64
+ IDE: Visual Studio Ultimate 2012 in Chinese Version
+ Python: Python 2.7.15 Anaconda 4.5.4 32-bit

## Experiment

### Step 1: Call the Python scripts on C++ from Visual Studio

#### Step 1.1: Write the Python scripts

We first define following functions in Python:

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression

def getParameters(param_file_path):
    """
    read parameters from file and change into dict
    :param param_file_path [string] the path of the parameter file
    :return [dict]:{ param_name:param_value } 
    """

def msvar(param_file_path):
    """
    Read data from source data file and save result of msvar to result file
    :param param_file_path [string] the path of the parameter file
    Note that source data file must have column names so that pd.read_csv can read it correctly
    """

    # 1. get the parameters from param_file_path
    # 2. read source data
    # 3. start MarkovAutoregression process
    # 4. calculate the result
    # 5. save the result to result file
```

Note that in order to avoid type errors while passing parameters from C# to C++ and Python, we store all parameters to files and read them when necessary.

(Unfinished)