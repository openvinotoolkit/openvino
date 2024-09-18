String Tensors
==============


.. meta::
   :description: Learn how to pass and retrieve text to and from OpenVINO model.

OpenVINO tensors can hold not only numerical data, like floating-point or integer numbers,
but also textual information, represented as one or multiple strings.
Such a tensor is called a string tensor and can be passed as input or retrieved as output of a text-processing model, such as
`tokenizers and detokenizers <https://github.com/openvinotoolkit/openvino_tokenizers/tree/master>`__.

While this section describes basic API to handle string tensors, more practical examples that leverage both
string tensors and OpenVINO tokenizer can be found in
`GenAI Samples <https://github.com/openvinotoolkit/openvino.genai/tree/master/samples/cpp/greedy_causal_lm>`__.


Representation
##############

String tensors are supported in C++ and Python APIs, represented as instances of the ``ov::Tensor``
class with the ``element_type`` parameter equal to ``ov::element::string``. Each element of a string tensor is a string
of arbitrary length, including an empty string, and can be set independently of other elements in the same tensor.

Depending on the API used (C++ or Python), the underlying data type that represents the string when accessing the tensor elements is
different:

* in C++, std::string is used
* in Python, ``numpy.str_`` / ``numpy.bytes_`` populated Numpy arrays are used, as a read-only copy of the underlying C++ content

String tensor implementation doesn't imply any limitations on string encoding, as underlying ``std::string`` doesn't have such limitations.
It is capable of representing all valid UTF-8 characters but also any other byte sequence outside of the UTF-8 encoding standard.
Users should pay extra attention when handling arbitrary byte sequences when accessing tensor content as encoded UTF-8 symbols.

As the string representation is more sophisticated in contrast to for example ``float`` or ``int`` data type,
the underlying memory that is used for string tensor representation cannot be handled without properly constructing and destroying string objects.
Also, in contrast to numerical data, C++ and Python do not share the same memory layout, so there is no immediate
sharing of tensor content between the two APIs. Python provides only a numpy-compatible view of the data
allocated and held in C++ core as an array of the ``std::string`` objects.

A developer must consider these restrictions when writing code using string tensors and
avoid treating the content as raw bytes or as a view of data in Python.

Create a String Tensor
######################

The following is an example of how to create a small 1D tensor pre-populated with three elements:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. code-block:: py
         :force:

         import openvino as ov

         tensor = ov.Tensor(['text', 'more text', 'even more text'])

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

         #include <vector>
         #include <string>
         #include <openvino/openvino.hpp>

         std::vector<std::string> strings = {"text", "more text", "even more text"};
         ov::Tensor tensor(ov::element::string, ov::Shape{strings.size()}, &strings[0]);

The example demonstrates that similarly to tensors with numerical information,
a tensor object can be created on top of existing memory in C++ by providing a pointer to a pre-allocated array of elements.
Here, an instance of std::vector is used to hold the memory and consists of three std::string objects.
So, the ``tensor`` object in the C++ example will share the same memory as the ``strings`` vector.

Note that ``ov::Tensor``, when initialized with a pointer, requires pre-initialized memory with valid ``std::string`` objects
created by calling one of the available ``std::string`` constructors even for empty string. It is undefined behaviour if
not initialized memory is passed to this ``ov::Tensor`` constructor.

In the Python version of the example above, a regular list of strings is used as an initializer.
No memory sharing is available this time, in contrast to C++,
and the strings from the initialization list are copied to a separately allocated storage underneath the ``tensor`` object.

Besides a plain Python list of strings, an initializer can be one of the supported ``numpy`` arrays initialized
with Unicode or byte strings:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. code-block:: python
         :force:

         import numpy as np

         tensor = ov.Tensor(np.array(['text', 'more text', 'even more text']))
         tensor = ov.Tensor(np.array([b'text', b'more text', b'even more text']))

If ``ov::Tensor`` is created without providing initialization strings,
a tensor of a specified shape and empty strings as elements is created:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. code-block:: python
         :force:

         tensor = ov.Tensor(dtype=str, shape=[3])

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

         ov::Tensor tensor(ov::element::string, ov::Shape{3});

``ov::Tensor`` allocates and initializes the required number of ``std::string`` objects under the hood.


Accessing Elements
##################

The following code prints all elements in the 1D string tensor constructed above.
In C++ code the same ``.data`` template method is used for other data types,
and to access string data it should be called with the ``std::string`` type.
In Python, dedicated ``std_data`` and ``byte_data`` fields are used instead of ``data`` field for numerical data.

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. code-block:: python
         :force:

         data = tensor.str_data  # use tensor.byte_data instead to access encoded strings as `bytes`
         for i in range(tensor.get_size()):
            print(data[i])

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

         #include <iostream>

         std::string* data = tensor.data<std::string>();
         for(size_t i = 0; i < tensor.get_size(); ++i)
            std::cout << data[i] << '\n';

In the case of Python, an object retrieved with ``tensor.str_data`` (or ``tensor.bytes_data``) is a numpy array
with ``numpy.str_`` elements (or ``numpy.bytes_`` correspondingly). It is a copy of underlying data from
the ``tensor`` object and cannot be used for tensor content modification.
To set new values, the entire tensor content should be set as a list or as a ``numpy`` array, as demonstrated
below.

In contrast to Python, when using ``tensor.data<std::string>()`` in C++, a pointer to the underlying data
storage is returned and it can be used for tensor element modification:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. code-block:: python

         # Unicode strings:
         tensor.str_data = ['one', 'two', 'three']
         # Do NOT use tensor.str_data[i] to set a new value, it won't update the tensor content

         # Encoded strings:
         tensor.bytes_data = [b'one', b'two', b'three']
         # Do NOT use tensor.bytes_data[i] to set a new value, it won't update the tensor content

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

         std::string new_content[] = {"one", "two", "three"};
         std::string* data = tensor.data<std::string>();
         for(size_t i = 0; i < tensor.get_size(); ++i)
            data[i] = new_content[i];

When reading or setting string tensor elements in Python, it is recommended to use ``str`` objects (or ``numpy.str_`` if used in numpy array)
when it is known that the underlying byte sequence forms a valid UTF-8 encoded string.
Otherwise, if arbitrary byte sequences are allowed,
not necessarily within the UTF-8 standard, use ``bytes`` strings (or ``numpy.bytes_`` correspondingly) instead.

Accessing tensor content through ``str_data`` implicitly applies UTF-8 decoding.
If parts of the byte stream cannot be represented as valid Unicode symbols,
the � replacement symbol is used to signal errors in such invalid Unicode streams.

Additional Resources
####################

* Learn about the :doc:`basic steps to integrate inference in your application <integrate-openvino-with-your-application>`.

* Use `OpenVINO tokenizers <https://github.com/openvinotoolkit/openvino_contrib/tree/master/modules/custom_operations/user_ie_extensions/tokenizer/python>`__ to produce models that use string tensors to work with textual information as pre- and post-processing for the large language models.

* Check out `GenAI Samples <https://github.com/openvinotoolkit/openvino.genai/tree/master/text_generation/causal_lm/cpp>`__ to see how string tensors are used in real-life applications.
