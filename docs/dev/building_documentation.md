# How to build documentation

The following procedure was validated on Windows and Ubuntu operation systems.

## Table of content:

* [Installing dependencies](#installing-dependencies)
* [Building documentation](#installing-dependencies)
* [Additional Resources](#additional-resources)

## Installing dependencies

The `doxygen` and `latex` must be installed in addition to usual build dependencies:

### Windows

* [miktex](https://miktex.org/)
* [doxygen](http://doxygen.nl/files/doxygen-1.8.20-setup.exe) (version >= 1.8.20)
* [Graphviz](https://graphviz.org/download/)

### Ubuntu systems

* Latex and graphviz:

```sh
apt-get install texlive-full graphviz
```

* goxygen version >= 1.8.20:

```
$ git clone https://github.com/doxygen/doxygen.git
$ cd doxygen
$ git checkout Release_1_8_20
$ mkdir build
$ cd build
$ cmake ..
$ cmake --build . -j8
$ sudo make install
```

## Building documentation

You should run `cmake` with `-DENABLE_DOCS=ON` and it will find all dependencies automatically on Ubuntu systems, while on Windows we still need to specify paths to the installed dependencies:

```sh
cmake -DLATEX_COMPILER="C:/Program Files/MiKTeX/miktex/bin/x64/latex.exe" \
      -DDOXYGEN_DOT_EXECUTABLE="C:/Program Files (x86)/Graphviz2.38/bin/dot.exe" \
      -DDOXYGEN_EXECUTABLE="C:/Program Files/doxygen/bin/doxygen.exe" \
      -DENABLE_DOCS=ON \
```

Once the dependencies are found, the project must generated using CMake. The target `openvino_docs` must be built to generate doxygen documentation, the generated files can be found at `<binary dir>/docs/html/index.html`

## See also

 * [OpenVINO README](../../README.md)
 * [OpenVINO Developer Documentation](index.md)
 * [How to Build OpenVINO](build.md)
