# Examples of OpenVINO™ Python API code

### Building and environment
Instructions can be found in ["Building the OpenVINO™ Python API"](./build.md).

## Different ways of extending OpenVINO™ Python API

### Before start: Project's naming conventions
General guide:
* Snake case (also known as the *lower_case_with_underscores* style) is used across the codebase. That includes modules (`runtime`, `offline_transformations`), function names, and arguments/variables (`async_infer`, `wait`, `path_to_xml`).
* Naming of classes is an exception to the above rule. The *CamelCase* style is used in this case, for example: `Core`, `InferRequest` or `AsyncInferQueue`.
* If bindings (explained later in the [Pure pybind11 solution](#pure-pybind11-solution) section) are created to expose existing C++ code, make them similar to their C++ counterparts, regarding both names and placement, for example, C++'s `ov::InferRequest` and Python's `openvino.runtime.InferRequest`. If alignment is not possible, try to describe your class/function/module as well as possible, such as the pair of `openvino.runtime.ConstOutput/openvino.runtime.Output` which relates to `ov::Output<const ov::Node>/ov::Output<ov::Node>`. This naming points out the functional difference between both classes - one is an immutable and the other a mutable version.

<!-- Pure Python solution describes Python based approach -->
## Pure Python solution
One of the simplest ways to extend the existing codebase is by writing it in pure Python.

### Layout of the project
How does OpenVINO™ packaging work? It is strictly connected to the layout of the Python API itself and reused in different supporting packages like tools and extensions. The main namespace of `openvino` provides a unified place that connects all packages together during import, **which is the required part**. However, it is up to the developer how to organize the rest of the package. There are also other common namespaces which follow the same rules:
* `openvino.tools`
* ...

For further reading, please refer to: https://packaging.python.org/en/latest/guides/packaging-namespace-packages/

### Creating new package that extends OpenVINO™ project namespace
Let's go over the example available in [examples folder](./examples/openvino):

```
openvino/               <-- Main package/namespace
├── __init__.py         <-- Unified file between all packages
└── mymodule/           <-- This is your new module and its contents
    ├── __init__.py
    ├── ...
    └── myclass.py
```

Now let's add it to your exisiting `PYTHONPATH` (replace `[your_path]` with correct path to the OpenVINO™ project):

    export PYTHONPATH=$PYTHONPATH:[your_path]/openvino/src/bindings/python/docs/examples/

Test how it works:

```python
import openvino.mymodule as ov_module

obj = ov_module.MyClass()
obj.say_hello()
>>> "Hello! Let's work on OV together!"
```

### Extending of existing API (sub)modules
But how to extend existing API? Let's navigate to [main bindings folder](./../src/openvino/) and add something to project helpers. Create new directory and fill it's contents:

```
openvino/
├── frontend/
├── helpers/                        <-- Working directory
│   ├── __init__.py
│   ├── custom_module/              <-- New directory
│   │   ├── __init__.py             <-- New file
│   │   ├── custom_helpers.py       <-- New file
│   │   └── packing.py
│   ├── ...
│   ├── runtime/
│   ├── test_utils/
│   └── __init__.py
└── utils.py
```

Let's add in [`custom_helpers.py`](./examples/custom_module/custom_helpers.py):
<!-- TODO: Link with code -->
```python
def top1_index(results: list) -> int:
    return results.index(max(results))
```

Import it to a new module in [`custom_module/__init__.py`](./examples/custom_module/__init__.py):
<!-- TODO: Link with code -->
```python
from openvino.helpers.custom_module.custom_helpers import top1_index
```

Follow it with correct import in [`helpers/__init__.py`](./../src/openvino/helpers/__init__.py):
```python
from openvino.helpers.custom_module import top1_index
```
**Do not forget to include a license on the top of each file!** For demonstration purposes, it has been skipped in the snippets above.

To see the changes take effect, [rebuild the project](../../../../docs/dev/build.md) and run your solution:
```python
import openvino.helpers as ov_helpers

ov_helpers.top1_index([0.7, 2.99, 3.0, -1.0])
>>> 2
```

Following this method, developers can add new modules and adjust existing ones, including structure, naming, and imported functions.
<!-- TODO: Mention about adding it to setup.py as well -->

<!-- Pure pybind11 solution describes C++ based approach -->

## Pure pybind11 solution
The second approach to extend OpenVINO™ codebase is utilizing the *pybind11* library. It allows to write C++ based code, thus creating so-called Python bindings.

**The example in this section covers the scenario of adding new features to a newly created submodule. Extending existing codebase can be done in a similar fashion by working on already implemented classes and modules.**

### Before start: What is pybind11?
It is a thridparty project that allows to expose C++ code as a Python library.

Link to offical documentation: https://pybind11.readthedocs.io/en/stable/
Link to project repository: https://github.com/pybind/pybind11

### Adding new (sub)module
Adding a new module could be done only by using *pybind11* built-in capabilities.

Navigate to the main project file responsible for creation of the whole package [`pyopenvino.cpp`](./../src/pyopenvino/pyopenvino.cpp), let's call it "registering-point".

Add a new submodule by writing:
```cpp
py::module mymodule = m.def_submodule("mymodule", "My first feature - openvino.runtime.mymodule");
```
This is a shorthand way of adding new submodules which can later be used to extend the package. The mysterious `m` is actaully the main OpenVINO™ module called `pyopenvino` -- it is registered with `PYBIND11_MODULE(pyopenvino, m)` at the top of the "registering-point" file. Later imports from it are done by calling upon the `openvino._pyopenvino` package.

Keep in mind that in most real-life scenarios, modules and classes are registered in different files. The general idea is to create a helper function that will hold all of the registered modules, classes, and functions. This function needs to be exposed within a separate header file and included in "registering-point". The project's common guideline suggests to use names in the following convention: `regmodule_[domain]_[name_of_the_module]` or `regclass_[domain]_[name_of_the_class]`. Where optional `[domain]` generally points to parts of the API such as graph or frontend, or stay empty in the case of core runtime. Examples can be found in the "registering-point" file [`pyopenvino.cpp`](../src/pyopenvino/pyopenvino.cpp).

*Note: Submodules can be "chained" as well. Refer to the official documentation for more details: https://pybind11.readthedocs.io/en/stable/reference.html#_CPPv4N7module_13def_submoduleEPKcPKc*

### Binding of classes and functions
When the module is created, classes can be added to it. Let's assume a class called `MyTensor` needs to be added in a new module. Here is a list of required features:

* Handle construction from the `ov::Tensor` class and 1-D initialization from Python built-in list.
* If there is a call upon `get_size`, return correct number of underlaying `Tensor` elements.
* If there is a call upon `say_hello`, the printed massage *"Hello there!"* and data held by `Tensor` should appear on the screen.

Here is a simple blueprint for the `MyTensor` class. Notice how **not all** requirements are met:
```cpp
class MyTensor {
public:
    // Constructing MyTensor is done with already created Tensor
    MyTensor(ov::Tensor& t) : _tensor(t) {};

    ~MyTensor() = default;

    // Gets size from Tensor and return it
    size_t get_size() const {
        return _tensor.get_size();
    }

    // Public member that allows all operations on Tensor
    ov::Tensor _tensor;
};
```

Let's start binding code. Align a class name with its C++ counterpart and add a documentation string: 
```cpp
// Add class to the module
py::class_<MyTensor, std::shared_ptr<MyTensor>> cls(mymodule, "MyTensor");

cls.doc() = "These are my first class bindings!"
```

Create `__init__` and functions for the class. `py::arg` stand for actual arguments of the given function. Remember, these should match their C++ equivalents both in order and number. However, argument names are not required to be exact copies of the C++ ones, as different restricted keywords or shortnames could appear. Pick them tastefully :) A very unusual construct of `R"(...)"` adds a documentation string to the function. The OpenVINO™ project follows the *reST (reStructuredText)* format of docstrings.
```cpp
// This initialize use already implemented C++ constructor 
cls.def(py::init<ov::Tensor&>(),
        py::arg("tensor"),
        R"(
            MyTensor's constructor.

            :param tensor: `Tensor` to create new `MyTensor` from.
            :type tensor: openvino.runtime.Tensor
        )");

// This initialize use custom constructor, implemented via lambda 
cls.def(py::init([](std::vector<float>& list) {
                auto tensor = ov::Tensor(ov::element::f32, ov::Shape({list.size()}));
                std::memcpy(tensor.data(), &list[0], list.size() * sizeof(float));
                return MyTensor(tensor);
            }),
        py::arg("list"),
        R"(
            MyTensor's constructor.

            :param list: List to create new `MyTensor` from.
            :type list: list
        )");

// Binds class function directly
cls.def("get_size", &MyTensor::get_size,
        R"(
            Get MyTensor's size.
        )");

// Adds function only on pybind's layer -- this function will become exclusive to Python API
cls.def("say_hello", [](const MyTensor& self) {
            py::print("Hello there!");
            for (size_t i = 0; i < self.get_size(); i++) {
                py::print(self._tensor.data<float>() + i);
            }
        },
        R"(
            Say hello and print contents of Tensor.
        )");
```

*Tip: To add a function on the module's level as a "free function", simply define it on the module's object:*
```cpp
mymodule.def("get_smile", []() {
    py::print(":)");
});
```

**Don't forget to rebuild the project** and test it out:
```python
import openvino._pyopenvino.mymodule as mymodule
from openvino.runtime import Tensor, Type

a = mymodule.MyTensor([1,2,3])
a.get_size()
>>> 3
a.say_hello()
>>> Hello there!
>>> 1.0
>>> 2.0
>>> 3.0

t = Tensor(Type.f32, [5])
t.data[:] = 1
b = mymodule.MyTensor(t)
b.get_size()
>>> 5
b.say_hello()
>>> Hello there!
>>> 1.0
>>> 1.0
>>> 1.0
>>> 1.0
>>> 1.0

mymodule.get_smile()
>>> :)
```

Note: **bindings** that are created for classes are sometimes called **wrappers**. It is not uncommon to see a statement like:

    MyTensor wraps (around) Tensor class.

However, in OpenVINO™ there is an unwritten distinction between "everyday" wrappers and more complex ones (with this article published... it is now a written one ;) ). An example may be found in [`core/infer_request.hpp`](./../src/pyopenvino/core/infer_request.hpp), where `InferRequest` is actually wrapped inside `InferRequestWrapper`, similarly to the `Tensor` and `MyTensor` scenario. It helps to extend original object capabilities with members and functions that do not necessarily belong to the C++ API. Thus, explicitly calling something a **wrapper** in the project indicates that binding is probably inheriting or using the composition technique to include the original class, later extending it in some way.

### Overloads of functions
One of the main advantages of *pybind11* is the ability to resolve overloaded functions. Let's assume that a previously created function is extended to print any message passed by the user.

```cpp
cls.def("say_hello", [](const MyTensor& self) {
    py::print("Hello there!");
    for (size_t i = 0; i < self.get_size(); i++) {
        py::print(self._tensor.data<float>() + i);
    }
});

cls.def("say_hello", [](const MyTensor& self, std::string& message) {
    py::print(message);
    for (size_t i = 0; i < self.get_size(); i++) {
        py::print(self._tensor.data<float>() + i);
    }
});
```

**Don't forget to rebuild the project** and test it out:
```python
import openvino._pyopenvino.mymodule as mymodule
a = mymodule.MyTensor([1,2,3])
a.say_hello()
>>> Hello there!
>>> 1.0
>>> 2.0
>>> 3.0
a.say_hello("New message!")
>>> New message!
>>> 1.0
>>> 2.0
>>> 3.0
# Let's try different message
a.say_hello(777)
>>> Traceback (most recent call last):
>>>   File "<stdin>", line 1, in <module>
>>> TypeError: say_hello(): incompatible function arguments. The following argument types are supported:
>>>     1. (self: openvino._pyopenvino.mymodule.MyTensor) -> None
>>>     2. (self: openvino._pyopenvino.mymodule.MyTensor, arg0: str) -> None
>>> 
>>> Invoked with: <openvino._pyopenvino.mymodule.MyTensor object at >>> 0x7fdfef5bb4f0>, 777
```
Notice that only functions with correct arguments are **not** throwing exceptions. It is helpful to combine this method with your code when binding templated or multi-argument functions. Most of the time (but not always!), using this approach saves a lot of written code, reducing `if-else/switch-case` blocks to a minimum, thus making it cleaner and easier to understand.
<!-- Link to one of our classes? --->

*Note: Please refer to offical documentation for more on overloading:*
* https://pybind11.readthedocs.io/en/stable/classes.html#overloaded-methods
* https://pybind11.readthedocs.io/en/stable/advanced/functions.html#overload-resolution-order

<!-- Mixed solution describes both approaches combined -->

## Mix between Python and pybind11
Although *pybind11* is a powerful tool, it is sometimes required (or simply easier and more efficent) to combine both approaches and utilize both languages to achive best results.

### Making pybind11-based module/class visible in OpenVINO™ package
Let's move a new class from `openvino._pyopenvino.mymodule` to the actual package. Simply introduce a new import statement in the desired file. Let it be [`openvino/runtime/__init__.py`](./../src/openvino/runtime/__init__.py): 
```python
from openvino._pyopenvino.mymodule import MyTensor
```

Now, while importing `openvino`, a new class is accessible from the `runtime` level:
```python
import openvino.runtime as ov
ov.MyTensor
>>> <class 'openvino._pyopenvino.mymodule.MyTensor'>
```

Same rule applies to whole modules and free functions. **This is a required step when adding something to the public API**. Without exposing it, all of the work is hidden in the depths of the `pyopenvino` namespace, rendering it hard to access for the user.

### Yet another Python layer
As mentioned earlier, it may be helpful to utilize Python in-between to achieve hard C++ feats in a more efficient way. Let's extend the previously created `say_hello` function a little bit.

First, create a new file in the [runtime directory](./../src/openvino/runtime/) and call it `mymodule_ext.py`. There are no strict rules for naming, just make sure the names are in good taste. Import the class here:
```python
from openvino._pyopenvino.mymodule import MyTensor as MyTensorBase
```

Notice how an alias is created for the `MyTensor` class. Do not worry, it will make sense as we progress. Let's follow it up with a more advanced class implementation:
```python
# Inherit from pybind implementation everything is preserved
class MyTensor(MyTensorBase):
    """MyTensor created as part of tutorial, it overrides pure-pybind class."""
    # Function name must be aligned with pybind one!
    def say_hello(self, arg=None):
        """Say hello to the world!

        :param arg: Argument of the function.
        :type arg: Union[str, int], optional
        """
        # If None invoke pybind implementation
        if arg is None:
            super().say_hello()
            return None
        # If string invoke pybind implementation and return 0
        elif type(arg) is str:
            super().say_hello(arg)
            return 0
        # If int convert argument, invoke pybind implementation
        # and return string
        elif type(arg) is int:
            # Additionally if less than 3, return itself plus one
            if arg < 3:
                return arg + 1
            super().say_hello(str(arg))
            return "Converted int to string!"
        # Otherwise raise an error
        raise TypeError("Unsupported argument type!")
```

Finally, import it in the same place as in the previous section, but this time use the improved version:
```python
from openvino.runtime.mymodule_ext import MyTensor
```

**Don't forget to rebuild the project** and test it out:
```python
import openvino.runtime as ov

a = ov.MyTensor([1,2,3])             # Notice that initializers are preserved from pybind class
a.say_hello()
>>> Hello there!
>>> 1.0
>>> 2.0
>>> 3.0

a.say_hello("I know how to make new features!")
>>> I know how to make new features! # String takes effect!
>>> 1.0
>>> 2.0
>>> 3.0
>>> 0                                # Correct return!

a.say_hello(8)
>>> 8                                # String conversion takes effect!
>>> 1.0
>>> 2.0
>>> 3.0
>>> 'Converted int to string!'       # Correct return!

a.say_hello(2)
>>> 3

a.say_hello([1,2,3])
>>> Traceback (most recent call last):
>>>   File "<stdin>", line 1, in <module>
>>>   File ".../openvino/runtime/mymodule_ext.py", line 20, in say_hello
>>>     raise TypeError("Unsupported argument type!")
>>> TypeError: Unsupported argument type!
```

Great! Now the class has reached its destination, from C++, to Python, to Python once more. Such aliasing is a common technique in the project and gives a lot of power to the developer. With easy-to-understand code, the `say_hello` function is now able to dispatch arguments based on their type and apply necessary preprocessing to feed data into the function. However, you might say that this could be done with "a few more lines of C++" as well. Where is the tricky part? The answer is, the difficult feat of returning different types based on the same argument type is achieved here (look at the dispatching of integer arguments).

This concludes developer work on OpenVINO™ Python API. Don't forget to recompile your builds and have a good time while writing your code!:)

## Testing the new code

Coding is now finished. Let's move on to testing.

To learn how to test your code, refer to the guide on [how to test OpenVINO™ Python API?](./test_examples.md#Running_OpenVINO™_Python_API_tests)

## See also
 * [OpenVINO™ README](../../../../README.md)
 * [OpenVINO™ bindings README](../../README.md)
 * [torchvision to OpenVINO™ preprocessing converter README](../src/openvino/preprocess/README.md)
 * [Developer documentation](../../../../docs/dev/index.md)
