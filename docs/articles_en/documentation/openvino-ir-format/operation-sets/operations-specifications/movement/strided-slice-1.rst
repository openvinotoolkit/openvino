.. {#openvino_docs_ops_movement_StridedSlice_1}

StridedSlice
============


.. meta::
  :description: Learn about StridedSlice-1 - a data movement operation,
                which can be performed on three required and one optional input tensor.

**Versioned name**: *StridedSlice-1*

**Category**: *Data movement*

**Short description**: *StridedSlice* extracts a strided slice of a tensor.

**Detailed description**: The *StridedSlice* operation extracts a slice from a given tensor based on computed indices from the inputs: begin (inclusive of the element at the given index), end (exclusive of the element at the given index), and stride, for each dimension.

The operation takes inputs with the following properties:

   * The N-dimensional input tensor to slice.
   * begin, end, and stride inputs - 1D lists of integers of the same length M. Stride input cannot contain any zeros.
   * begin_mask, end_mask, new_axis_mask, shrink_axis_mask, ellipsis_mask inputs - bitmasks, 1D lists of integers (0 or 1). Each mask can have a unique length. ellipsis_mask can have up to one occurrence of the value 1.
   * new_axis_mask, shrink_axis_mask, ellipsis_mask are used to modify the output dimensionality of the data. If they are unused, N == M. Otherwise, N does not necessarily equal M.

The basic slicing operation accumulates output elements as follows:

   * The operation iterates over the values of begin, end, and stride. At every step, the operation uses the i-th element of begin, end, and stride to perform the slicing.
   * Let slicing_index = begin[i]. This value determines the first index to start slicing. This sliced element is added to the output.
   * If begin[i] == end[i], only a single element is added to the output. The corresponding output dimension is then equal to 1 (in other words, the dimension is kept).
   * At each step, the slicing_index is incremented by the value of stride[i]. As long as the slicing_index < end[i], the element corresponding to the slicing_index is added to the output.
   * Whenever slicing_index >= end[i], the slicing stops, and the corresponding element is not added to the output.

Notice that the basic slicing operation assumes N = M (that is, i-th slicing step corresponds to i-th dimension), as no masks are used. The description above assumes that begin[i] <= end[i] and stride[i] > 0, and that begin[i] >= 0 and end[i] >= 0.

.. note:: Negative Values in Begin and End (Negative Values Adjusting)

   Negative values represent indexing from the back, i.e., the value of -1 represents the last element of the input dimension. In practice, negative values are automatically incremented by the size of the dimension. For example, if data = [0, 1, 2, 3], size(data) = 4, and begin(i) = -1 for some i, this value will be modified to be begin(i) = -1 + 4 = 3. Note that if begin(i) = -5 for some i, this value will be modified to -5 + 4 = -1, and might cause an error.

.. note:: Indexing in Reverse

   If stride[i] < 0, the indexing will happen in reverse. At each step, the value of stride[i] will be subtracted from the slicing_index. As long as the slicing_index > end[i], the corresponding element is added to the output. Whenever slicing_index <= end[i], the slicing stops. Note that when slicing in reverse, it is impossible to select the first element of the dimension, as that would require end[i] = -1, which has a different meaning.

.. note:: Value Out-of-Bounds (Silent Clamping)

   If a value in begin or end is out of bounds for the corresponding dimension, it is silently clamped. In other words:

      * If begin[i] >= dim, begin[i] = dim. If begin[i] < 0 (after Negative Values Adjusting), begin[i] = 0.
      * If end[i] >= dim, end[i] = dim. If end[i] < 0 (after Negative Values Adjusting), end[i] = 0.

   If slicing in reverse, the clamping behavior changes to the following:

      * If begin[i] >= dim, begin[i] = dim - 1. If begin[i] < 0 (after Negative Values Adjusting), begin[i] = 0.
      * If end[i] >= dim, end[i] = dim. If end[i] < 0 (after Negative Values Adjusting), end[i] = -1.

The operation accepts multiple bitmasks in the form of integer arrays to modify the above behavior. If the length of the bitmask is less than the length of the corresponding input, it is assumed that the bitmask is extended with zeros.

During the i-th slicing step:

   * If the begin_mask[i] is set to one, the value of begin[i] is set to 0 (dim - 1 if slicing in reverse).
   * If the end_mask[i] is set to one, the value of end[i] is set to dim (0 if slicing in reverse - note that this does not allow slicing inclusively with the first value).
   * If the new_axis_mask[i] is set to one, the values of begin[i], end[i], and stride[i] ARE IGNORED, and a new dimension with size 1 appears in the output. No slicing occurs at this step.
   * If the shrink_axis_mask[i] is set to one, the values of begin[i] MUST EQUAL end[i] (Note that this would normally result in a size 1 dimension), and the stride[i] value IS IGNORED. The corresponding dimension is removed, with only a single element from that dimension remaining.
   * If the ellipsis_mask[i] is set to one, the begin[i], end[i], and stride[i] ARE IGNORED, and a number of dimensions are skipped. The exact number of dimensions skipped in the original input is dim - (M - new_axes - 1). The corresponding dimension is treated as an ellipsis ('...'), or in other words, it is treated as multiple, sequential, unaffected by slicing dimensions, that match the rest of the slicing operation. This allows for a concise and flexible way to perform slicing operations, effectively condensing the slicing parameters for dimensions marked with ellipsis into a single slice notation. For example, given a 10D input, and tasked to select the first element from the 1st and last dimension, normally one would have to write [0, :, :, :, :, :, :, :, :, :, 0], but with ellipsis, it is only necessary to write [0, ..., 0].

.. note:: The i-th Slicing Step and Dimension Modification

   The i-th slicing step does not necessarily correspond to the i-th dimension modification. Let i be the index of the slicing step, and j be the corresponding processed dimension.
   For these cases:

      * Every time all of the masks are not set (set to 0), j is incremented by one.
      * Every time begin_mask[i] or end_mask[i] is set to one, j is incremented by one.
      * Every time shrink_axis_mask[i] is set to one, j is incremented by one.

   However:

      * Every time new_axis_mask[i] is set to one, j is not incremented.
      * When the value of one occurs at ellipsis_mask[i], j is incremented by size(dim) - (size(begin) - sum(new_axis_mask) - 1).

**Attributes**

* *begin_mask*

  * **Description**: *begin_mask* is a bit mask. *begin_mask[i]* equal to ``1`` means that the corresponding dimension of the ``begin`` input is ignored and the 'real' beginning of the tensor is used along corresponding dimension.
  * **Range of values**: a list of ``0`` s and ``1`` s
  * **Type**: ``int[]``
  * **Default value**: None
  * **Required**: *yes*

* *end_mask*

  * **Description**: *end_mask* is a bit mask. If *end_mask[i]* is ``1``, the corresponding dimension of the ``end`` input is ignored and the real 'end' of the tensor is used along corresponding dimension.
  * **Range of values**: a list of ``0`` s and ``1`` s
  * **Type**: ``int[]``
  * **Default value**: None
  * **Required**: *yes*

* *new_axis_mask*

  * **Description**: *new_axis_mask* is a bit mask. If *new_axis_mask[i]* is ``1``, a length 1 dimension is inserted on the ``i``-th position of input tensor.
  * **Range of values**: a list of ``0`` s and ``1`` s
  * **Type**: ``int[]``
  * **Default value**: ``[0]``
  * **Required**: *no*

* *shrink_axis_mask*

  * **Description**: *shrink_axis_mask* is a bit mask. If *shrink_axis_mask[i]* is ``1``, the dimension on the ``i``-th position is deleted.
  * **Range of values**: a list of ``0`` s and ``1`` s
  * **Type**: ``int[]``
  * **Default value**: ``[0]``
  * **Required**: *no*

* *ellipsis_mask*

  * **Description**: *ellipsis_mask* is a bit mask. It inserts missing dimensions on a position of a non-zero bit.
  * **Range of values**: a list of ``0`` s and ``1``. Only one non-zero bit is allowed.
  * **Type**: ``int[]``
  * **Default value**: ``[0]``
  * **Required**: *no*

**Inputs**:

*   **1**: ``data`` - input tensor to be sliced of type *T* and arbitrary shape. **Required.**

*   **2**: ``begin`` - 1D tensor of type *T_IND* with begin indexes for input tensor slicing. **Required.**
    Out-of-bounds values are silently clamped. If ``begin_mask[i]`` is ``1`` , the value of ``begin[i]`` is ignored and the range of the appropriate dimension starts from ``0``. Negative values mean indexing starts from the end. For example, if ``data=[1,2,3]``, ``begin[0]=-1`` means ``begin[0]=3``.

*   **3**: ``end`` - 1D tensor of type *T_IND* with end indexes for input tensor slicing. **Required.**
    Out-of-bounds values will be silently clamped. If ``end_mask[i]`` is ``1``, the value of ``end[i]`` is ignored and the full range of the appropriate dimension is used instead. Negative values mean indexing starts from the end. For example, if ``data=[1,2,3]``, ``end[0]=-1`` means ``end[0]=3``.

*   **4**: ``stride`` - 1D tensor of type *T_IND* with strides. **Optional.**

**Types**

* *T*: any supported type.
* *T_IND*: any supported integer type.

**Example**

Basic example with different strides, standard slicing and in reverse.

.. code-block:: xml
   :force:

    <layer ... type="StridedSlice" ...>
        <data/>
        <input>
            <port id="0">
                <dim>4</dim>
                <dim>4</dim>
                <dim>4</dim>
                <dim>4</dim>
                <dim>4</dim>
                <dim>4</dim>
            </port>
            <port id="1">
                <dim>6</dim> <!-- begin: [0, 1, 0, 1, 3, 3] -->
            </port>
            <port id="2">
                <dim>6</dim> <!-- end: [4, 4, 4, 4, 0, 0] -->
            </port>
            <port id="3">
                <dim>6</dim> <!-- stride: [1, 1, 2, 2, -1, -2] -->
            </port>
        </input>
        <output>
            <port id="4">
                <dim>4</dim> <!-- element ids: [0, 1, 2, 3] -->
                <dim>3</dim> <!-- element ids: [1, 2, 3] -->
                <dim>2</dim> <!-- element ids: [0, 2] -->
                <dim>2</dim> <!-- element ids: [1, 3] -->
                <dim>4</dim> <!-- element ids: [3, 2, 1, 0] -->
                <dim>2</dim> <!-- element ids: [3, 1] -->
            </port>
        </output>
    </layer>

Example of clamping in standard and reverse slicing.

.. code-block:: xml
   :force:

    <layer ... type="StridedSlice" ...>
        <data/>
        <input>
            <port id="0">
                <dim>2</dim>
                <dim>2</dim>
            </port>
            <port id="1">
                <dim>2</dim> <!-- begin: [1234, 2] -->
            </port>
            <port id="2">
                <dim>2</dim> <!-- end: [1234, 4321] -->
            </port>
            <port id="3">
                <dim>2</dim> <!-- stride: [1, -1] - note that second slicing is in reverse,, which modifies the clamping behavior -->
            </port>
        </input>
        <output>
            <port id="4">
                <dim>1</dim> <!-- begin clamped to 2, end clamped to 3, element ids: [2] -->
                <dim>1</dim> <!-- begin clamped to 2, end clamped to 1, element ids: [2] -->
            </port>
        </output>
    </layer>

Example of negative slicing.

.. code-block:: xml
   :force:

    <layer ... type="StridedSlice" ...>
        <data/>
        <input>
            <port id="0">
                <dim>2</dim>
                <dim>3</dim>
                <dim>4</dim>
            </port>
            <port id="1">
                <dim>3</dim> <!-- begin: [0, 0, 0] -->
            </port>
            <port id="2">
                <dim>3</dim> <!-- end: [2, 2, -1] - note that -1 will be replaced by 4 - 1 = 3 -->
            </port>
            <port id="3">
                <dim>3</dim> <!-- stride: [1, 1, 1] -->
            </port>
        </input>
        <output>
            <port id="4">
                <dim>2</dim> <!-- element ids: [0, 1] -->
                <dim>2</dim> <!-- element ids: [0, 1] -->
                <dim>3</dim> <!-- element ids: [0, 1, 2] -->
            </port>
        </output>
    </layer>

Example of ``begin_mask`` & ``end_mask`` usage.

.. code-block:: xml
   :force:

    <layer ... type="StridedSlice" ...>
        <data begin_mask="0,1,1" end_mask="1,1,1" new_axis_mask="0,0,0" shrink_axis_mask="0,0,0" ellipsis_mask="0,0,0" />
        <input>
            <port id="0">
                <dim>2</dim>
                <dim>3</dim>
                <dim>4</dim>
            </port>
            <port id="1">
                <dim>3</dim> <!-- begin: [1, 1, 123] - notice that since begin_mask overrides value, it can be left as any value, commonly 0 -->
            </port>
            <port id="2">
                <dim>3</dim> <!-- end: [0, 0, 2] - notice that since end_mask overrides value, it can be left as any value, commonly 0 -->
            </port>
            <port id="3">
                <dim>3</dim> <!-- stride: [1, 1, -1] - notice that last slicing happens in reverse, so masks behavior changes -->
            </port>
        </input>
        <output>
            <port id="4">
                <dim>1</dim> <!-- begin = 1, end = 2 (end_mask override), element ids: [1] -->
                <dim>3</dim> <!-- begin = 0 (begin_mask override), end = 3 (end_mask override), element ids: [0, 1, 2] -->
                <dim>3</dim> <!-- begin = 3 (begin_mask override), end = 0 (end_mask override), element ids: [3, 2, 1] -->
            </port>
        </output>
    </layer>

Example of ``new_axis_mask`` usage.

.. code-block:: xml
   :force:

    <layer ... type="StridedSlice" ...>
        <data begin_mask="0,0,0,0" end_mask="0,0,0,0" new_axis_mask="1,0,1,0" shrink_axis_mask="0,0,0,0" ellipsis_mask="0,0,0,0"/>
        <input>
            <port id="0">
                <dim>2</dim>
                <dim>4</dim>
            </port>
            <port id="1">
                <dim>4</dim> <!-- begin: [1234, 0, -1, 0] - notice that since new_axis_mask skips the corresponding value, it can be left as any value, commonly 0 -->
            </port>
            <port id="2">
                <dim>4</dim> <!-- end: [1234, 2, 9876, 4] - notice that since new_axis_mask skips the corresponding value, it can be left as any value, commonly 0 -->
            </port>
            <port id="3">
                <dim>4</dim> <!-- stride: [132, 1, 241, 1] - notice that since new_axis_mask skips the corresponding value, it can be left as any value, commonly 0 -->
            </port>
        </input>
        <output>
            <port id="4">
                <dim>1</dim> <!-- new dimension appears -->
                <dim>2</dim> <!-- second value from begin, end, stride used to slice first dimension of input -->
                <dim>1</dim> <!-- new dimension appears -->
                <dim>4</dim> <!-- fourth value from begin, end, stride used to slice second dimension of input -->
            </port>
        </output>
    </layer>

Example of ``shrink_axis_mask`` usage.

.. code-block:: xml
   :force:

    <layer ... type="StridedSlice" ...>
        <data begin_mask="0,0,0,0,0" end_mask="0,0,0,0,0" new_axis_mask="0,0,0,0,0" shrink_axis_mask="0,1,0,0,0" ellipsis_mask="0,0,0,0,0"/>
        <input>
            <port id="0">
                <dim>1</dim> <!-- first dim -->
                <dim>2</dim> <!-- second dim -->
                <dim>384</dim>
                <dim>640</dim>
                <dim>8</dim>
            </port>
            <port id="1">
                <dim>5</dim> <!-- begin: [0, 0, 0, 0, 0] -->
            </port>
            <port id="2">
                <dim>5</dim> <!-- end: [1, 1, 384, 640, 8] -->
            </port>
            <port id="3">
                <dim>5</dim> <!-- stride: [1, 1, 1, 1, 1] -->
            </port>
        </input>
        <output>
            <port id="4">
                <dim>1</dim> <!-- first dim kept, as shrink_axis_mask is 0, second dim is missing as shrink_axis_mask is 1 -->
                <dim>384</dim>
                <dim>640</dim>
                <dim>8</dim>
            </port>
        </output>
    </layer>

Example of ``ellipsis_mask`` usage.

.. code-block:: xml
   :force:

    <layer ... type="StridedSlice" ...>
        <data begin_mask="0,0,0" end_mask="0,0,0" new_axis_mask="0,0,0" shrink_axis_mask="0,0,0" ellipsis_mask="0,1,0"/>
        <input>
            <port id="0">
                <dim>10</dim> <!-- first dim -->
                <dim>10</dim> 
                <dim>10</dim>
                <dim>10</dim>
                <dim>10</dim>
                <dim>10</dim>
                <dim>10</dim>
                <dim>10</dim>
                <dim>10</dim>
                <dim>10</dim>
                <dim>10</dim>
                <dim>10</dim> <!-- last dim -->
            </port>
            <port id="1">
                <dim>3</dim> <!-- begin: [0, 0, 0] - with second dimension marked as ellipsis, this pattern aims to modify first and last dimension -->
            </port>
            <port id="2">
                <dim>3</dim> <!-- end: [4, 0, 5] -->
            </port>
            <port id="3">
                <dim>3</dim> <!-- stride: [1, -1, 1] -->
            </port>
        </input>
        <output>
            <port id="4">
                <dim>4</dim> <!-- first dim modified -->
                <dim>10</dim> 
                <dim>10</dim>
                <dim>10</dim>
                <dim>10</dim>
                <dim>10</dim>
                <dim>10</dim> <!-- ellipsis flexibly skipped over dimensions to match requested pattern -->
                <dim>10</dim>
                <dim>10</dim>
                <dim>10</dim>
                <dim>10</dim>
                <dim>5</dim> <!-- last dim modified -->
            </port>
        </output>
    </layer>
