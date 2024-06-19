.. {#openvino_docs_ops_movement_Pad_1}

Pad
===


.. meta::
  :description: Learn about Pad-1 - a data movement operation,
                which can be performed on three required and one optional input tensor.

**Versioned name**: *Pad-1*

**Category**: *Data movement*

**Short description**: *Pad* operation extends an input tensor on edges. The amount and value of padded elements are defined by inputs and attributes.

**Detailed Description**: The attributes specify a number of elements to add along each axis and a rule by which new element values are generated: for example, whether they are filled with a given constant or generated based on the input tensor content.

The following examples illustrate how output tensor is generated for the *Pad* layer for a given input tensor:

.. code-block:: cpp

    INPUT =
    [[ 1  2  3  4 ]
    [ 5  6  7  8 ]
    [ 9 10 11 12 ]]


with the following attributes:

.. code-block:: cpp

    pads_begin = [0, 1]
    pads_end = [2, 3]


depending on the *pad_mode*.

* ``pad_mode = "constant"``:

.. code-block:: cpp

    OUTPUT =
    [[ 0  1  2  3  4  0  0  0 ]
    [ 0  5  6  7  8  0  0  0 ]
    [ 0  9 10 11 12  0  0  0 ]
    [ 0  0  0  0  0  0  0  0 ]
    [ 0  0  0  0  0  0  0  0 ]]


* ``pad_mode = "edge"``:

.. code-block:: cpp

    OUTPUT =
    [[ 1  1  2  3  4  4  4  4 ]
    [ 5  5  6  7  8  8  8  8 ]
    [ 9  9 10 11 12 12 12 12 ]
    [ 9  9 10 11 12 12 12 12 ]
    [ 9  9 10 11 12 12 12 12 ]]


* ``pad_mode = "reflect"``:

.. code-block:: cpp

    OUTPUT =
    [[  2  1  2  3  4  3  2  1 ]
    [  6  5  6  7  8  7  6  5 ]
    [ 10  9 10 11 12 11 10  9 ]
    [  6  5  6  7  8  7  6  5 ]
    [  2  1  2  3  4  3  2  1 ]]

* ``pad_mode = "symmetric"``:

.. code-block:: cpp

    OUTPUT =
    [[ 1  1  2  3  4  4  3  2 ]
    [ 5  5  6  7  8  8  7  6 ]
    [ 9  9 10 11 12 12 11 10 ]
    [ 9  9 10 11 12 12 11 10 ]
    [ 5  5  6  7  8  8  7  6 ]]

**Attributes**

* *pad_mode*

  * **Description**: *pad_mode* specifies the method used to generate new element values.
  * **Range of values**: Name of the method in string format:

    * ``constant`` - padded values are equal to the value of the *pad_value* input, if input not provided zero value is padded.
    * ``edge`` - padded values are copied from the respective edge of the input ``data`` tensor.
    * ``reflect`` - padded values are a reflection of the input ``data`` tensor; values on the edges are not duplicated. ``pads_begin[D]`` and ``pads_end[D]`` must be not greater than ``data.shape[D] – 1`` for any valid ``D``.
    * ``symmetric`` - padded values are symmetrically added from the input ``data`` tensor. This method is similar to the ``reflect``, but values on edges are duplicated. Refer to the examples above for more details. ``pads_begin[D]`` and ``pads_end[D]`` must be not greater than ``data.shape[D]`` for any valid ``D``.
  * **Type**: ``string``
  * **Required**: *yes*

**Inputs**

* **1**: ``data`` tensor of arbitrary shape and type *T*. **Required.**

* **2**: ``pads_begin`` 1D tensor of type *T_INT*. Number of elements matches the number of indices in *data* attribute. Specifies the number of padding elements at the beginning of each axis. **Required.**

* **3**: ``pads_end`` 1D tensor of type *T_INT*. Number of elements matches the number of indices in *data* attribute. Specifies the number of padding elements at the ending of each axis. **Required.**

* **4**: ``pad_value`` scalar tensor of type *T*. Used with the ``pad_mode = "constant"`` only. All new elements are populated with this value or with 0 if input not provided. Shouldn't be set for other ``pad_mode`` values. **Optional.**


**Outputs**

* **1**: Output padded tensor of type *T* with dimensions ``pads_begin[D] + data.shape[D] + pads_end[D]`` for each ``D`` from ``0`` to ``len(data.shape) - 1``.

**Types**

* *T*: any numeric type.

* *T_INT*: any non-negative integer type.


**Example**: constant mode

.. code-block:: xml
   :force:

    <layer ... type="Pad" ...>
        <data pad_mode="constant"/>
        <input>
            <port id="0">
                <dim>1</dim>
                <dim>3</dim>
                <dim>32</dim>
                <dim>40</dim>
            </port>
            <port id="1">
                <dim>4</dim>     <!-- pads_begin = [0, 5, 2, 1]  -->
            </port>
            <port id="2">
                <dim>4</dim>     <!-- pads_end = [1, 0, 3, 7] -->
            </port>
            <port id="3">
                                <!-- pad_value = 15.0 -->
            </port>
        </input>
        <output>
            <port id="0">
                <dim>2</dim>     <!-- 2 = 0 + 1 + 1 = pads_begin[0] + input.shape[0] + pads_end[0] -->
                <dim>8</dim>     <!-- 8 = 5 + 3 + 0 = pads_begin[1] + input.shape[1] + pads_end[1] -->
                <dim>37</dim>    <!-- 37 = 2 + 32 + 3 = pads_begin[2] + input.shape[2] + pads_end[2] -->
                <dim>48</dim>    <!-- 48 = 1 + 40 + 7 = pads_begin[3] + input.shape[3] + pads_end[3] -->
                                <!-- all new elements are filled with 15.0 value -->
            </port>
        </output>
    </layer>


**Example**: edge mode

.. code-block:: xml
   :force:

    <layer ... type="Pad" ...>
        <data pad_mode="edge"/>
        <input>
            <port id="0">
                <dim>1</dim>
                <dim>3</dim>
                <dim>32</dim>
                <dim>40</dim>
            </port>
            <port id="1">
                <dim>4</dim>     <!-- pads_begin = [0, 5, 2, 1]  -->
            </port>
            <port id="2">
                <dim>4</dim>     <!-- pads_end = [1, 0, 3, 7] -->
            </port>
        </input>
        <output>
            <port id="0">
                <dim>2</dim>     <!-- 2 = 0 + 1 + 1 = pads_begin[0] + input.shape[0] + pads_end[0] -->
                <dim>8</dim>     <!-- 8 = 5 + 3 + 0 = pads_begin[1] + input.shape[1] + pads_end[1] -->
                <dim>37</dim>    <!-- 37 = 2 + 32 + 3 = pads_begin[2] + input.shape[2] + pads_end[2] -->
                <dim>48</dim>    <!-- 48 = 1 + 40 + 7 = pads_begin[3] + input.shape[3] + pads_end[3] -->
            </port>
        </output>
    </layer>

