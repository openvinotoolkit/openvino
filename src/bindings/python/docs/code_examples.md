# Examples of OpenVINO:tm: Python API code
<!-- Do we keep it as is? -->

#### Prerequisites
<!-- Do we keep it here? -->
*To be added...*

##### Enviroment
<!-- Link to enviroment setup -->
*To be added...*

##### Building
<!-- Link to building instructions -->
*To be added...*

### Different ways of extending OpenVINO:tm: Python API

###### Before start: Project's naming conventions
General guide:
* Snake case is used across the code. That includes modules (`runtime`, `offline_transformations`), function names and arguments/varaibles (`async_infer`, `wait`, `path_to_xml`).
* Exception to above rule is naming of classes. For example: `Core`, `InferRequest` and `AsyncInferQueue` start each word with a capital letter.
* If bindings (term explained later in section "Pure pybind11 solution" <!-- Add link to section? https://stackoverflow.com/questions/2822089/how-to-link-to-part-of-the-same-document-in-markdown -->) are created to already existing C++ code, try to get close to their C++ counterpart with names and placement (example: C++'s `ov::InferRequest` and Python's `openvino.runtime.InferRequest`). If aligement is not possible, try to describe your class/function/module in the best way. One of the examples is pair of `openvino.runtime.ConstOutput/openvino.runtime.Output` which relates to `ov::Output<const ov::Node>/ov::Output<ov::Node>`, this naming points out to functional difference between both classes - first is immutable version and second is mutable.

<!-- Pure Python solution describes Python based approach -->
#### Pure Python solution
One of the simplest way to extend existing codebase is by writing it in pure Python.

###### Before start: Layout of the project
How is OpenVINO:tm: packaging working? It is strictly connected to the layout of the Python API itself and reused in different supporting packages like tools and extensions. Main namespace `openvino` provides unified place that connects all packages together during importing, **this is the required part**. However, it is up to the developer how to organize the rest of the package. There are also other common namespaces which follows same rules:
* `openvino.tools`
* ...

For further reading, please refer to: https://packaging.python.org/en/latest/guides/packaging-namespace-packages/

##### Creating new package that extends OpenVINO:tm: project namespace
Let's go over the example available in `openvino/src/bindings/python/docs/examples/openvino`:

```
openvino/               <-- Main package/namespace
    __init__.py         <-- Unified file between all packages
    mymodule/           <-- This is your new module and it's contents:)
        __init__.py
        ...
        myclass.py
```

Now let's add it to your exisiting `PYTHONPATH` (replace `[your_path]` with correct path to the OpenVINO:tm: project):

    export PYTHONPATH=$PYTHONPATH:[your_path]/openvino/src/bindings/python/docs/examples/

Test how it works:

```python
import openvino.mymodule as ov_module

obj = ov_module.MyClass()
obj.say_hello()
>>> "Hello! Let's work on OV together!"
```

##### Extending of existing API (sub)modules
But how to extend existing API? Let's navigate to `openvino/src/bindings/python/src/openvino` and add something to project helpers. Create new directory and fill it's contents:

```
openvino/
    frontend/
    helpers/                        <-- Working directory
        __init__.py
        custom_module/              <-- New directory
            __init__.py             <-- New file
            custom_helpers.py       <-- New file
        packing.py
    ...
    runtime/
    test_utils/
    __init__.py
    utils.py
```

Let's add in `custom_module/custom_helpers.py`:
```python
def top1_index(results: list) -> int:
    return results.index(max(results))
```

Import it to new module in `custom_module/__init__.py`:
```python
from openvino.helpers.custom_module.custom_helpers import top1_index
```

Follow it with correct import in `helpers/__init__.py`:
```python
from openvino.helpers.custom_module import top1_index
```
**Do not forget to include a license on the top of each file!** For demonstrational puroposes (and to keep this markdown file short) it is not included in above snippets.

To see changes taking effect, rebuild project (Cmake's install step should be good enough) and run your solution:
```python
import openvino.helpers as ov_helpers

ov_helpers.top1_index([0.7, 2.99, 3.0, -1.0])
>>> 2
```

Following this method developers can add new modules and adjust existing ones including structure, naming and imported functions.
<!-- !!!!!!!!Mention about adding it to setup.py as well!!!!!!!! -->

<!-- Pure pybind11 solution describes C++ based approach -->

#### Pure pybind11 solution
Second approach to extend OpenVINO:tm: codebase is utilizing *pybind11* library. It allows to write C++ based code, thus creating so-called Python bindings.

**The example in this section covers the scenario of adding new features to newly created submodule. Extending existing codebase can be done in similar fashion by working on already implemented classes and modules.**

###### Before start: What is pybind11?
<!-- Do we keep it here? -->
Link to offical documentation: https://pybind11.readthedocs.io/en/stable/
Link to project repository: https://github.com/pybind/pybind11

##### Adding new (sub)module
Adding new module could be done only by using *pybind11* built-in capabilities.

Navigate to main project file responsible for creation of whole package, let's call it "registering-point":

    openvino/src/bindings/python/src/pyopenvino/pyopenvino.cpp

Add new submodule by writing:
```cpp
py::module mymodule = m.def_submodule("mymodule", "My first feature - openvino.runtime.mymodule");
```
This is a shorthand way of adding new submodules which can later be used to extend package. The mysterious `m` is actaully the main OpenVINO:tm: module called `pyopenvino` -- it is registered with `PYBIND11_MODULE(pyopenvino, m)` at the top of the "registering-point" file. Later imports from it are done by calling upon `openvino.pyopenvino` package.

Please keep in mind, in most of real-life scenarios, modules and classes are registered in different files. General idea is to create helper function that will hold all of the registered modules, classes and functions. This function needs to be exposed within a separate header file and included in "registering-point". Project's common guideline suggest to use names in the following convention `reg[class|module]_[domain]_[name_of_the_class|module]`. Where optional `[domain]` generally points to parts of the API like graph, frontend or stay empty in case of core runtime. Examples can be found in "registering-point" file `openvino/src/bindings/python/src/pyopenvino/pyopenvino.cpp`.

*Note: Submodules can be "chained" as well, please refer to offical documentation for more: https://pybind11.readthedocs.io/en/stable/reference.html#_CPPv4N7module_13def_submoduleEPKcPKc*

##### Binding of classes and functions
When the module is created, classes can be added to it. Let's assume class called `MyTensor` needs to be added in new module. Here is a list of required features:

* Handle construction from `ov::Tensor` class and 1-D initialization from Python built-in list.
* If there is a call upon `get_size`, return correct number of underlaying `Tensor` elements.
* If there is a call upon `say_hello`, the printed massage *"Hello there!"* and data held by `Tensor` should appear on the screen.

Here is a simple blueprint for `MyTensor` class, notice how **not all** requirements are met here:
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

Let's start binding code, align class name with C++ counterpart and add documentation string: 
```cpp
// Add class to the module
py::class_<MyTensor, std::shared_ptr<MyTensor>> cls(mymodule, "MyTensor");

cls.doc() = "These are my first class bindings!"
```

Create `__init__` and functions for the class. `py::arg` stand for actual arguments of the given function, remember these should match their C++ equivalent in order and count <!-- number? -->. However argument names are not required to be an exact copy of C++ ones, as different restricted keywords or shortnames could appear, pick them with a taste. :) Very unusual `R"(...)"` construct adds documentation string to the function. OpenVINO:tm: project follows *reST (reStructuredText)* format of docstrings.
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
import openvino.pyopenvino.mymodule as mymodule
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

Note: In nomenclature **bindings** that are created for classes may be called **wrappers**. It is often seen statement that:

    MyTensor wraps (around) Tensor class.

However, in OpenVINO:tm: there is an unwritten distinction (after uploading this file... it is now written:)) between "everyday" warppers and more complex ones. Example could be found in `openvino/src/bindings/python/src/pyopenvino/core/infer_request.hpp` where `InferRequest` is actually wrapped inside `InferRequestWrapper`, similar to `Tensor` and `MyTensor` scenario. It helps to extend original object capabilities with members and functions that does not necessary belong to C++ API. Thus explicitly calling something **wrapper** in the project indicates that binding is probably inheriting or using the composition technique to include original class, later extending it in some way.

##### Overloads of functions
One of the main advantages of *pybind11* is an ability to resolve overloaded functions. Let's assume that previously created function is extended to print any message passed by the user.

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
import openvino.pyopenvino.mymodule as mymodule
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
>>>     1. (self: openvino.pyopenvino.mymodule.MyTensor) -> None
>>>     2. (self: openvino.pyopenvino.mymodule.MyTensor, arg0: str) -> None
>>> 
>>> Invoked with: <openvino.pyopenvino.mymodule.MyTensor object at >>> 0x7fdfef5bb4f0>, 777
```
Notice that only functions with correct arguments are **not** throwing exceptions. It is helpful to combine this method with your code when binding templated or multi-argument functions. Most of the time (but not always!), using this approach saves a lot of written code, reducing `if-else/switch-case` blocks to minimum and thus making code cleaner and easier to understand.
<!-- Link do one of our classes? --->

*Note: Please refer to offical documentation for more on overloading:*
* https://pybind11.readthedocs.io/en/stable/classes.html#overloaded-methods
* https://pybind11.readthedocs.io/en/stable/advanced/functions.html#overload-resolution-order

<!-- Mixed solution describes both approaches combined -->

#### Mix between Python and pybind11
Although *pybind11* is a powerful tool, it is sometimes required (or simply easier and more efficent) to combine both approaches and utilize both languages to achive best results.

##### Making pybind11-based module/class visible in OpenVINO:tm: package
Let's move new class from `openvino.pyopenvino.mymodule` to the actual pacakge. Simply introduce new import statement in the desired file, let it be `openvino/src/bindings/python/src/openvino/runtime/__init__.py`: 
```python
from openvino.pyopenvino.mymodule import MyTensor
```

Now while importing `openvino`, new class is accessible from `runtime` level:
```python
import openvino.runtime as ov
ov.MyTensor
>>> <class 'openvino.pyopenvino.mymodule.MyTensor'>
```

Same rule applies to whole modules and free functions. **This is required step when adding something to public API**. Without exposing it, all of the work is hidden in depths of  `pyopenvino` namespace, rendering it hard-to-access for the user.

##### Yet another Python layer
As mentioned eariler it may be helpful to utilize Python language in-between to achive hard C++ feats in more efficent way. Let's extend previously created `say_hello` function a little bit.

First create new file in `openvino/src/bindings/python/src/openvino/runtime` directory and call it `mymodule_ext.py`. There are no strict rules about naming, use your good taste for that. Import the class here:
```python
from openvino.pyopenvino.mymodule import MyTensor as MyTensorBase
```

Notice how the alias is created for `MyTensor` class, do not worry it will make sense in a while. Let's follow it up with more advanced class implementation:
```python
# Inherit from pybind implementation everything is preserved
class MyTensor(MyTensorBase):
    # Function name must be aligned with pybind one!
    def say_hello(self, arg=None):
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

Finally, import it in the same place as in a previous section, but this time use improved version:
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

Great! Now the class has gone a full cricle, from C++ to Python (and again to Python). Such aliasing is a common technique in the project and gives a lot of power to the developer. With easy to understand code, `say_hello` function is now able to dispatch arguments based on their type and apply necessary preprocessing to feed data into the function. However, this could be done in "few more lines of C++" as well, where is the tricky part? The difficult feat of returning different types based on same argument type is achived (look at the dispatching of integer arguments).

This concludes developer work on OpenVINO:tm: Python API. Don't forget to recompile your builds and have a good time while writing your code!:)

#### Testing out new code
All of the code is now written. Let's move on to the testing of it.

Please refer to Test Guide available here:

    openvino/src/bindings/python/docs/test_examples.md
