# Python requirements and version constraints management

OpenVINO uses a pip built-in feature called "constraints" in order to reduce the complexity of requirements. 

## What are constraints files?

Official constraints [documentation](https://pip.pypa.io/en/stable/user_guide/#constraints-files) describes them as:
> Constraints files are requirements files that only control which version of a requirement is installed, not whether it is installed or not. Their syntax and contents is a subset of Requirements Files, with several kinds of syntax not allowed: constraints must have a name, they cannot be editable, and they cannot specify extras. In terms of semantics, there is one key difference: Including a package in a constraints file does not trigger installation of the package.

Constraints files have the `.txt` extension and are addons to regular `requirements.txt` files. They are useful when the project has multiple components, each with its own dependencies.

This means we can specify only package names for different components and keep their versions in a centralized constraints file, which is significantly easier to maintain. 


### Example
Two requirements files are linked to a centralized constraints file using the `-c` flag.

```text
# main/constraints.txt

coverage>=4.4.2
astroid>=2.9.0
pylint>=2.7.0
pyenchant>=3.0.0
test-generator==0.1.1
```

```text
# requirements_pytorch.txt

-c main/constraints.txt  # versions are already defined here!
coverage
astroid
pylint
pyenchant
```

```text
# requirements_tensorflow.txt

-c main/constraints.txt  # versions are already defined here!
coverage
pylint
pyenchant
test-generator
```

Note that `astroid` is not required by `requirements_tensorflow.txt`, so it's not going to be installed even though its version has been specified in `main/constraints.txt`.

## Benefits of this approach

Using `constraints.txt` files brings many benefits, some of which are:
- Lower project complexity
    - Significantly fewer requirements files to maintain
    - Centralization of project requirements
- Easier pip conflict management
    - Upgrading package versions is now easier
    - Package versions are soft-forced to be aligned across the project
- Groundwork for future package managers and utilities

## Known limitations and best practices

There are several known limitations to the `constraints.txt` approach. The most notable are:

#### 1. Environment markers
Placing [environment markers](https://peps.python.org/pep-0508/) in `constraints.txt` files can be prone to bugs due to how pip resolver interprets them. Markers should be placed only in `requirements.txt` files unless it is certain they will work as intended.

#### 2. Inconsistent version requirements
If a package version differs between `requirements.txt` files, it can't be unified in the common `constraints.txt` file. The workarounds, in order of preference, are:
- Align the package versions between `requirements.txt` files
- Exclude this package from `constraints.txt` and keep its version in `requirements.txt`
- Exclude this `requirements.txt` file from the constraints system

## Implementation in OpenVINO
The implementation in OpenVINO is a subject to change. At the time of writing, there are three `constraints.txt` files with the following requirement coverage:

|Constraints file                               |Coverage                   |
|-----------------------------------------------|---------------------------|
|`openvino/src/bindings/python/constraints.txt` |Python bindings, frontends |
|`openvino/tests/constraints.txt`               |tests                      |
|`openvino/tools/constraints.txt`               |tools, openvino_dev        |

## See also

 * [OpenVINO™ README](../../../../README.md)

 * [OpenVINO™ bindings README](../../README.md)

 * [Developer documentation](../../../../docs/dev/index.md)
