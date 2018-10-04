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

#### Step 1.2: Create msvar C++ project

Open Visual Studio Ultimate 2012 and create a new Visual C++ Empty Project:

<figure>
	<img src="/images/python_on_visual_cpp/new_C++_project.png" alt="msvar console application">
</figure>

Enter Python installation directory and copy **include** and **libs** to our msvar project's location:

<figure>
	<img src="https://smallGum.github.io/images/python_on_visual_cpp/copy_files.png" alt="copy python files">
</figure>

Since the **Debug** mode on Visual Studio requires `python27_d.lib` file rather than `python27.lib`, we must change this file's name in **libs** directory:

<figure>
	<img src="https://smallGum.github.io/images/python_on_visual_cpp/change_lib_name.png" alt="change library name">
</figure>

Open the attribute of our msvar project and set additional Python library:

<figure>
	<img src="https://smallGum.github.io/images/python_on_visual_cpp/set_include.png" alt="set include">
</figure>

<figure>
	<img src="https://smallGum.github.io/images/python_on_visual_cpp/set_lib.png" alt="set library">
</figure>

<figure>
	<img src="https://smallGum.github.io/images/python_on_visual_cpp/set_python_lib.png" alt="set python27_d">
</figure>

#### Step 1.3: Write C++ code and test

Create a new `.cpp` file and add following function:

```c++
#include <python.h>
#include <stdlib.h>
#include <iostream>
#include <string>

using namespace std;

// call Python's msvar model
// param: pFile the directory of your Python scripts
// param: cFile the path of your config file
void msvar(char* pFile, char* cFile) {
	string pythonFilePath = string(pFile);
	string configFilePath = string(cFile);

	// Initialize Python and load models
	Py_Initialize();

	// check if the initialization success
	if (!Py_IsInitialized()) {
		cout << "Initialized failed !" << endl;
		return ;
	}

	// add path of Python script to system path
	// PyRun_SimpleString can execute Python script directly
	PyRun_SimpleString("import sys");
	PyRun_SimpleString("print ('---import sys---')");
	string wholePath = "sys.path.append(\'" + pythonFilePath + "\')";
	PyRun_SimpleString(wholePath.c_str());
	PyRun_SimpleString("print(sys.path)");

	PyObject *pName = NULL, *pModule = NULL, *pDict = NULL, *pFunc = NULL, *pArgs = NULL;
	
	// load python script
	cout << "Finding python file msvar......" << endl;
	pName = PyUnicode_FromString("msvar");
	pModule = PyImport_Import(pName);
	if (!pModule) {
		cout << "can't find python file." << endl;
		return ;
	}

	// get the functions
	pDict = PyModule_GetDict(pModule);
	if (!pDict) {
		cout << "Get functions failed !" << endl;
	}

	// find the msvar function  
	cout << "Finding msvar function......" << endl;
	pFunc = PyDict_GetItemString(pDict, "msvar");
	if (!pFunc || !PyCallable_Check(pFunc)) {
		cout << "can't find function msvar." << endl;
		return ;
	}

	// add parameters
	pArgs = PyTuple_New(1);

	// PyObject* Py_BuildValue(char *format, ...) 
	// translate variables in C++ to Python objects  
	// common format:  
	// s string
	// i integer
	// f float
	// O a Python object
	PyTuple_SetItem(pArgs, 0, Py_BuildValue("s", configFilePath.c_str()));

	// call the Python function  
	PyObject_CallObject(pFunc, pArgs);

	// finalize Python
	Py_Finalize();
}

int main() {
	// enter your corresponding file path
	msvar("G:\\MSVAR\\Python_scripts", "G:\\MSVAR\\ControlParam\\controlParam.csv");
	system("pause");

	return 0;
}
```

Run the code above under **Debug** mode. If there is no error, we can package it into DLL file.

### Step 2: Package `.cpp` files of step 1 into DLL file

Create a new Win32 Console Application called **msvarDLL** under the same solution:

<figure>
	<img src="https://smallGum.github.io/images/python_on_visual_cpp/new_win32_project.png" alt="new Win32 project">
</figure>

Enter the guide, click **next step**, choose **DLL** type and finish creation:

<figure>
	<img src="https://smallGum.github.io/images/python_on_visual_cpp/create_DLL.png" alt="create DLL">
</figure>

Then open **msvarDLL** attribute and set Python **include** and **libs** path like step 1.2. Next, we create **msvar.h** file and write following configuration:

```c++
#include <string>

using namespace std;

#ifndef MSVARDLL_H_
#define MSVARDLL_H_
#ifdef MYLIBDLL
#define MYLIBDLL extern "C" _declspec(dllimport) 
#else
#define MYLIBDLL extern "C" _declspec(dllexport) 
#endif
MYLIBDLL void msvar(char* pythonFilePath, char* configFilePath);
#endif
```

Finally, we move our msvar function in step 1.3 into **msvarDLL.cpp** file:

```c++
#include "stdafx.h"
#include "msvar.h"
#include <python.h>
#include <stdlib.h>
#include <iostream>

// call Python's msvar model
// param: pFile the directory of your Python scripts
// param: cFile the path of your config file
void msvar(char* pFile, char* cFile) {...}
```

Set our **msvarDLL** project as the startup project and recreate the solution, we eventually get the `.dll` file of our msvar model:

<figure>
	<img src="https://smallGum.github.io/images/python_on_visual_cpp/get_DLL_file.png" alt="get DLL file">
</figure>

### Step 3: Import the DLL file of step 2 and use the model on C# programs

Create a new C# Console Application called **msvarDLLTest** under the same solution:

<figure>
	<img src="https://smallGum.github.io/images/python_on_visual_cpp/create_CSharp_project.png" alt="test DLL">
</figure>

Add following C# program to run the msvar model of Python:

```c#
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace msvarDLLTest
{
    class Program
    {
        // import your dll file
        [DllImport(@"G:\MSVAR\msvar\Debug\msvarDLL.dll", EntryPoint = "msvar", CallingConvention=CallingConvention.Cdecl)]
        extern static void msvar(byte[] pythonFilePath, byte[] configFilePath);

        static void Main(string[] args)
        {
            byte[] pFile = System.Text.Encoding.Default.GetBytes("G:\\MSVAR\\Python_scripts");
            byte[] cFile = System.Text.Encoding.Default.GetBytes("G:\\MSVAR\\ControlParam\\controlParam.csv");
            msvar(pFile, cFile);
            Console.ReadLine();
        }
    }
}
```

## Result

Finally, we run the C# msvar model using close price data of IF with 2 regime, and got the correct result:

<figure>
	<img src="https://smallGum.github.io/images/python_on_visual_cpp/get_result.png" alt="result">
</figure>