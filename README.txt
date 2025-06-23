README - Cythonizing VNTree V2
=============================

This guide explains how to build and integrate the Cythonized version of `VNTree V2` (`binomial.pyx`) into the `mixvol` package.

Prerequisites
-------------
- Python 3.7 or later
- A C++ compiler (e.g., `g++` on Linux/macOS or MSVC on Windows)
- `pip` for installing Python packages

Required Python Packages
------------------------
```bash
pip install cython numpy scipy
```

Project Layout
--------------
```text
/mixvol/
  ├── bsm_model.py
  ├── binomial_tree.py      # Original V1
  ├── binomial.pyx          # Cythonized VNTree V2
  ├── forward_tools.py
  ├── mixture_model.py
  └── __init__.py
setup.py
README.txt
```

Key Steps
---------
1. **Ensure `binomial.pyx` is in `mixvol/`**  
   - This file contains the Cython code with inlined forward logic and full Greeks.

2. **Prepare `setup.py`**  
   - At project root (`/mnt/data`), create a `setup.py` with:
     ```python
     from setuptools import setup, Extension
     from Cython.Build import cythonize
     import numpy as np

     extensions = [
         Extension(
             name="mixvol.binomial",
             sources=["mixvol/binomial.pyx"],
             include_dirs=[np.get_include()],
             language="c++",
         ),
     ]

     setup(
         name="mixvol",
         version="0.1.0",
         packages=["mixvol"],
         ext_modules=cythonize(
             extensions,
             compiler_directives={
                 'boundscheck': False,
                 'wraparound': False,
                 'cdivision': True,
             }
         ),
         install_requires=["numpy", "scipy", "Cython"],
         zip_safe=False,
     )
     ```

3. **Clean previous builds**
   ```bash
   rm -rf build/ mixvol/binomial*.c mixvol/binomial*.so
   ```

4. **Build the extension**
   ```bash
   python setup.py build_ext --inplace
   ```
   - This generates `mixvol/binomial.so` (or `.pyd` on Windows).

5. **Verify installation**
   ```python
   >>> import mixvol.binomial as bt2
   >>> tree = bt2.VNTree(100, 100, 1.0, lambda t:0.05, lambda t:0.0, [(0.5,1.0)], 0.2)
   >>> print(tree.price())
   ```

6. **Run benchmarks or tests**
   - Use your existing test scripts to compare performance between:  
     - `BSMModel`  
     - `binomial_tree.VNTree` (V1)  
     - `mixvol.binomial.VNTree` (Cython V2)

Troubleshooting
---------------
- **Compiler errors**: Ensure your C++ compiler is installed and in `PATH`.  
- **Import errors**: Verify `mixvol/binomial.so` is next to `binomial.pyx` and your `PYTHONPATH` includes the project root.  
- **Stale artifacts**: Remove old `.c`, `.so`, and `build/` before rebuilding.

That’s it! You now have a fast, Cythonized VNTree V2 integrated into your Python project.
