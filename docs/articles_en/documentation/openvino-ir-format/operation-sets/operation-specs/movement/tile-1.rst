Tile
====


.. meta::
  :description: Learn about Tile-1 - a data movement operation, which can be
                performed on two required input tensors.

**Versioned name**: *Tile-1*

**Category**: *Data movement*

**Short description**: *Tile* operation repeats an input tensor *"data"* the number of times given by *"repeats"* input tensor along each dimension.

* If number of elements in *"repeats"* is more than shape of *"data"*, then *"data"* will be promoted to "*repeats*" by prepending new axes, e.g. let's shape of *"data"* is equal to (2, 3) and *"repeats"* is equal to [2, 2, 2], then shape of *"data"* will be promoted to (1, 2, 3) and result shape will be (2, 4, 6).
* If number of elements in *"repeats"* is less than shape of *"data"*, then *"repeats"* will be promoted to "*data*" by prepending 1's to it, e.g. let's shape of *"data"* is equal to (4, 2, 3) and *"repeats"* is equal to [2, 2], then *"repeats"* will be promoted to [1, 2, 2] and result shape will be (4, 4, 6)

**Attributes**:

No attributes available.

**Inputs**:

* **1**: "data" - an input tensor to be padded. A tensor of type *T1*. **Required.**
* **2**: "repeats" - a per-dimension replication factor. For example, *repeats* equal to 88 means that the output tensor gets 88 copies of data from the specified axis. A tensor of type *T2*. **Required.**

**Outputs**:

* **1**: The count of dimensions in result shape will be equal to the maximum from count of dimensions in "data" shape and number of elements in "repeats". A tensor with type matching 1st tensor.

**Types**

* *T1*: arbitrary supported type.
* *T2*: any integer type.

**Detailed description**:

*Tile* operation extends input tensor and filling in output tensor by the following rules:

.. math::

   out_i=input_i[inner_dim*t]

.. math::

   t \in \left ( 0, \quad tiles \right )

**Examples**

*Example 1: number elements in "repeats" is equal to shape of data*

.. code-block:: xml
   :force:

    <layer ... type="Tile">
        <input>
            <port id="0">
                <dim>2</dim>
                <dim>3</dim>
                <dim>4</dim>
            </port>
            <port id="1">
                <dim>3</dim>  <!-- [1, 2, 3] -->
            </port>
        </input>
        <output>
            <port id="2">
                <dim>2</dim>
                <dim>6</dim>
                <dim>12</dim>
            </port>
        </output>
    </layer>

*Example 2: number of elements in "repeats" is more than shape of "data"*

.. code-block:: xml
   :force:

    <layer ... type="Tile">
        <input>
            <port id="0">  <!-- will be promoted to shape (1, 2, 3, 4) -->
                <dim>2</dim>
                <dim>3</dim>
                <dim>4</dim>
            </port>
            <port id="1">
                <dim>4</dim>  <!-- [5, 1, 2, 3] -->
            </port>
        </input>
        <output>
            <port id="2">
                <dim>5/dim>
                <dim>2</dim>
                <dim>6</dim>
                <dim>12</dim>
            </port>
        </output>
    </layer>

*Example 3: number of elements in "repeats" is less than shape of "data"*

.. code-block:: xml
   :force:

    <layer ... type="Tile">
        <input>
            <port id="0">
                <dim>5</dim>
                <dim>2</dim>
                <dim>3</dim>
                <dim>4</dim>
            </port>
            <port id="1">
                <dim>3</dim>  <!-- [1, 2, 3] will be promoted to [1, 1, 2, 3] -->
            </port>
        </input>
        <output>
            <port id="2">
                <dim>5</dim>
                <dim>2</dim>
                <dim>6</dim>
                <dim>12</dim>
            </port>
        </output>
    </layer>


