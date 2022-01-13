# Build Documentation Using CMake

1. Clone submodules:
```
cd openvino
git submodule update --init --recursive
```
2. Install build dependencies using the `install_build_dependencies.sh` script in the project root folder.
```
chmod +x install_build_dependencies.sh
./install_build_dependencies.sh
```

3. Install [doxyrest](https://github.com/vovkos/doxyrest/releases/tag/doxyrest-2.1.2) and put the `bin` folder in your path

4. Install python dependencies:

```
python -m pip install -r docs/requirements.txt
```

5. Install the sphinx theme

6. Create a build folder:

```
mkdir build && cd build
```

7. Build documentation using these commands:
```
cmake .. -DENABLE_DOCS=ON -DENABLE_DOCKER=ON
```

```
cmake --build . --target sphinx_docs
```
