# Fleet-To-Lab

## Assets

- The assets for SpaceRoboticsBench can be found in the zip folder here: https://drive.google.com/drive/folders/1rrHhKuD8BpEVye5IY_SY9nEWqi4pD-u_?usp=sharing

## Capturing the Data

- Use the human_interaction_transfer_learning_collecting_data.py file to collect the data.
- Change the wheel radius according to which rover model you are using

## Changing Networks

- you can use a 3-hidden-layer neural network or a 1-hidden-layer neural network
- When changing the network comment and decomment in networks.py
- Change the for cycle of git rebasin to for(1,2) or for(1,4) according to your network
- Comment in the aco file of the otimizers the 543 part or 133351 part accordingly

# AcoMerge

Aco Merge has been written in C code, to use it in python must be compiled and wrapped in Cython

# Build .whl package On Linux:
```
pip install -r requirements.txt
sh generate_wheel_unix.sh
```

- On Linux specifically you have to fix the wheel package, use the container with the repair tool:

```
sudo docker run -i -t -v `pwd`:/io quay.io/pypa/manylinux1_x86_64 /bin/bash
```

when you are in run these lines:

```
cd io
sh repair_wheel_linux.sh
```

in the wheelhouse directory you have the fixed wheel package

# Build .whl package on MacOS
```
pip install -r requirements.txt
sh generate_wheel_unix.sh
sh repair_wheel_macos.sh
```

# Build .whl package on Windows

- First of all you need to comment this line in __init__.pyx:

```
ctypedef stdint.uint64_t uint64_t
```

- Install Mingw with MYSYS2: https://www.msys2.org/ follow the steps and also the passages to install mingw-w64 

-  now navigate to Pyezspark with MYSYS2 and create the .lib library:

```
sh create_library.sh
```

- Go to your Python Folder (We assume as example we are using python 3.7):

- You can find your python folder with:

```
import sys

locate_python = sys.exec_prefix
print(locate_python)
```

- Create the file Python37\Lib\distutils\distutils.cfg that should look like this:

```
[build]
compiler=mingw32
 
[build_ext]
compiler=mingw32
```

- Download this .dll file https://it.dllfile.net/microsoft/vcruntime140-dll:

- Move it to Python37\libs and to Python37\DLLs

- Move libllab.lib previously built to Python37\libs

- Now, Move to Python37\Lib\distutils\cygwinccompiler.py and modify the function get_msvcr() as follows: https://bugs.python.org/file40608/patch.diff

- The last part

```
elif msc_ver == '1900':
```

change it to 

```
elif msc_ver == '1916':
```

- Now other bug fix: go to Python37\include\pyconfig.h and add these lines:

 ```
 /* Compiler specific defines */
 

#ifdef __MINGW32__
#ifdef _WIN64
#define MS_WIN64
#endif
#endif
```

- Now you can run from MYSYS2:

```
sh build_python_library.sh
```

- Links:
 - https://www.msys2.org/ (To get MYSYS2 with mingw-w64)
 - https://wiki.python.org/moin/WindowsCompilers (distutils.cfg)
 - https://datatofish.com/locate-python-windows/ (Python folder search script)
 - https://stackoverflow.com/questions/34135280/valueerror-unknown-ms-compiler-version-1900 (vcruntime140-dll lacking dll)
 - https://bugs.python.org/file40608/patch.diff (function fix, changing from 1900 to 1916 was my guess and it worked, it enables the use of the .dll previously installed)
 - https://github.com/cython/cython/issues/3405 (For the ifdef stuff, cython bug)

# Install .whl files

Once you have created the .whl file, you can install it locally using pip:

```
pip install package.whl
```
