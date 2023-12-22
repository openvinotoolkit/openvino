# CPU plugin functional tests

## Structure of the platform specific tests

Taken CPU plugin specific single layer tests as an example:

``` shell
single_layer_tests/
├── classes # test classes
│   ├── activation.cpp # test class with common parameters
│   ├── activation.h
│   ├── convolution.cpp
│   └── convolution.h
└── instances # test instances
    ├── arm # arch specific instances if any
    │   ├── acl # backend specific instances if any
    │   │   └── actication.cpp 
    │   └── onednn
    │       └── convolution.cpp
    ├── common # common instances across all the architecture
    │   ├── activation.cpp
    │   └── onednn
    │       └── convolution.cpp
    ├── _some_new_arch
    │   ├── activation.cpp
    │   └── convolution.cpp
    └── x64 # arch specific instances if any
        ├── activation.cpp # native instances for the arch (no backend involved)
        ├── onednn # backend specific instances if any
        │   └── convolution.cpp
        └── _some_new_backend
            └── convolution.cpp
```
