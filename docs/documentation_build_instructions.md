# Building documentation with CMake

**NOTE**: Instructions were tested inside container based on Ubuntu 22.04 docker image. 

1. Clone the OpenVINO repository and setup its submodules
```
$ git clone <openvino_repository_url> <repository_path>
$ cd <repository_path>
$ git submodule update --init --recursive
```
2. Install build dependencies using the `install_build_dependencies.sh` script located in OpenVINO root directory
```
$ chmod +x install_build_dependencies.sh
$ ./install_build_dependencies.sh
```
3. Install additional packages needed to build documentation
```
$ apt install -y doxygen graphviz texlive
```
4. Create python virtualenv and install needed libraries
```
$ python3 -m venv env
$ source env/bin/activate
(env) $ pip install --upgrade setuptools pip
(env) $ pip install -r docs/requirements.txt
```
5. Install the sphinx theme
```
(env) $ python -m pip install docs/openvino_sphinx_theme
``````
6. Install the custom sphinx sitemap
```
(env) $ pip install docs/openvino_custom_sphinx_sitemap
``````
7. Create build folder:
```
(env) $ mkdir build && cd build
```
8. Build documentation using these commands:
```
(env) $ cmake .. -DENABLE_DOCS=ON
(env) $ cmake --build . --target sphinx_docs
```
Depending on the needs, following variables can be added to first cmake call:
- building C/C++ API:  `-DENABLE_CPP_API=ON`
- building Python API: `-DENABLE_PYTHON_API=ON`
- building Notebooks:  `-DENABLE_NOTEBOOKS=ON`
- building OVMS:       `-DENABLE_OVMS=ON -DOVMS_DOCS_DIR=<path_to_OVMS_repo>`
