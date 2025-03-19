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

* :math:`input` tensor to slice, with N dimensions.
* :math:`begin, end, stride` - 1D lists of integers of the same length M. **Stride input cannot contain any zeros.**
* :math:`begin\_mask, end\_mask, new\_axis\_mask, shrink\_axis\_mask, ellipsis\_mask` - bitmasks, 1D lists of integers (0 or 1). :math:`ellipsis\_mask` can have up to one occurrence of the value 1. **Each mask can have a unique length. The length of the masks can differ from the rank of the input shape.**
* :math:`new\_axis\_mask, shrink\_axis\_mask, ellipsis\_mask` modify the output dimensionality of the data. If they are unused, :math:`N == M`. Otherwise, N does not necessarily equal M.

.. note:: Negative Values in Begin and End (Negative Values Adjusting)

   Negative values present in :math:`begin` or :math:`end` represent indices starting from the back, i.e., the value of -1 represents the last element of the input dimension. In practice, negative values are automatically incremented by the size of the dimension. For example, if :math:`data = [0, 1, 2, 3]`, :math:`size(data) = 4`, :math:`begin(i) = -1` for some i, this value will be modified to be :math:`begin(i) = -1 + 4 = 3`. Note that if :math:`begin(i) = -5` for some i, this value will be adjusted as follows: :math:`begin(i) -5 + 4 = -1`, which will trigger value clamping.

The basic slicing operation accumulates output elements as follows:

* The operation iterates over the values of begin, end, and stride. At every step, the operation uses the i-th element of begin, end, and stride to perform the slicing at the corresponding dimension.
* Let :math:`slicing\_index = begin[i]`. This value determines the first index to start slicing. This sliced element is added to the output.
* If :math:`begin[i] == end[i]`, only a single element from the corresponding dimension is added to the output. The corresponding output dimension is then equal to 1 (in other words, the dimension is kept).
* At each step, the :math:`slicing\_index` is incremented by the value of :math:`stride[i]`. As long as the :math:`slicing\_index < end[i]`, the element corresponding to the :math:`slicing\_index` is added to the output.
* Whenever :math:`slicing\_index >= end[i]`, the slicing stops, and the corresponding element is not added to the output.

Notice that the basic slicing operation assumes :math:`N == M` (that is, i-th slicing step corresponds to i-th dimension), as no masks are used.

For the purposes of this specification, assume that :math:`dim` is the dimension corresponding to the i-th slicing step.

.. note:: Indexing in Reverse (Slicing in Reverse)

   If :math:`stride[i] < 0`, the indexing will happen in reverse. At each step, the value of :math:`stride[i]` will be subtracted from the :math:`slicing\_index`. As long as the :math:`slicing\_index > end[i]`, the corresponding element is added to the output. Whenever :math:`slicing\_index <= end[i]`, the slicing stops.

.. note:: Value Out-of-Bounds (Silent Clamping)

   If a value in begin or end is out of bounds for the corresponding dimension, it is silently clamped. In other words:

   * If :math:`begin[i] >= size(dim)`, then :math:`begin[i] = size(dim)`. If :math:`begin[i] < 0` (after Negative Values Adjusting), then :math:`begin[i] = 0`.
   * If :math:`end[i] >= size(dim)`, then :math:`end[i] = size(dim)`. If :math:`end[i] < 0` (after Negative Values Adjusting), then :math:`end[i] = 0`.

   If slicing in reverse, the clamping behavior changes to the following:

   * If :math:`begin[i] >= size(dim)`, then :math:`begin[i] = size(dim) - 1`. If :math:`begin[i] < 0` (after Negative Values Adjusting), then :math:`begin[i] = 0`.
   * If :math:`end[i] >= size(dim)`, then :math:`end[i] = size(dim)`. If :math:`end[i] < 0` (after Negative Values Adjusting), then :math:`end[i] = -1`.

The operation accepts multiple bitmasks in the form of integer arrays to modify the above behavior. **If the length of the bitmask is less than the length of the corresponding input, it is assumed that the bitmask is extended (padded at the end) with zeros. If the length of the bitmask is greater than necessary, the remaining values are ignored.**

For examples of usage of each mask, please refer to the examples provided at the end of the document.

During the i-th slicing step:

* If the :math:`begin\_mask[i]` is set to one, the value of :math:`begin[i]` is set to :math:`0`` (:math:`size(dim) - 1` if slicing in reverse). Equivalent of swapping left handside of Python slicing operation :math:`array[0:10]` with :math:`array[:10]` (slice from the start).
* If the :math:`end\_mask[i]` is set to one, the value of :math:`end[i]` is set to :math:`size(dim)` (:math:`0` if slicing in reverse - note that this does not allow slicing inclusively with the first value). Equivalent of swapping right handside of Python slicing operation :math:`array[0:10]` (assume :math:`len(array) = 10`) with :math:`array[0:]` (slice till the end, inclusive).
* If the :math:`new\_axis\_mask[i]` is set to one, the values of :math:`begin[i]`, :math:`end[i]`, and :math:`stride[i]` **ARE IGNORED**, and a new dimension with size 1 appears in the output. No slicing occurs at this step. Equivalent of inserting a new dimension into a matrix using numpy :math:`array[..., np.newaxis, ...]`: :math:`shape(array) = [..., 1, ...]`.
* If the :math:`shrink\_axis\_mask[i]` is set to one, the value of  :math:`begin[i]` **MUST EQUAL** :math:`end[i]` (Note that this would normally result in a size 1 dimension), and the :math:`stride[i]` value **IS IGNORED**. The corresponding dimension is removed, with only a single element from that dimension remaining. Equivalent of selecting only a given element without preserving dimension (numpy equivalent of keepdims=False) :math:`array[..., 0, ...] -> array[..., ...]` (one less dimension).
* If the :math:`ellipsis\_mask[i]` is set to one, the values of :math:`begin[i], end[i],` and :math:`stride[i]` **ARE IGNORED**, and a number of dimensions is skipped. The exact number of dimensions skipped in the original input is :math:`rank(input) - (M - sum(new\_axis\_mask) - 1)`. The corresponding dimension is treated as an ellipsis ('...'), or in other words, it is treated as multiple, sequential, and unaffected by slicing dimensions, that match the rest of the slicing operation. This allows for a concise and flexible way to perform slicing operations, effectively condensing the slicing parameters for dimensions marked with ellipsis into a single slice notation. For example, given a 10D input, and tasked to select the first element from the 1st and last dimension, normally one would have to write :math:`[0, :, :, :, :, :, :, :, :, :, 0]`, but with ellipsis, it is only necessary to write :math:`[0, ..., 0]`. Equivalent of Equivalent of using the '...' (ellipsis) opeartion in numpy :math:`array[0, ..., 0] (rank(array) = 10)` is the same as writing :math:`array[0, :, :, :, :, :, :, :, :, 0]`.

.. note:: The i-th Slicing Step and Dimension Modification

   The i-th slicing step does not necessarily correspond to the i-th dimension modification. Let i be the index of the slicing step and j be index of the corresponding processed dimension.
   For trivial cases:

   * Every time all of the masks are not set (set to 0), j is incremented by one.
   * Every time :math:`begin\_mask[i]` or :math:`end\_mask[i]` is set to one, j is incremented by one.
   * Every time :math:`shrink\_axis\_mask[i]` is set to one, j is incremented by one.

   However:

   * Every time :math:`new\_axis\_mask[i]` is set to one, j is not incremented.
   * When the value of one occurs at :math:`ellipsis\_mask[i]`, j is incremented by :math:`rank(input) - (M - sum(new\_axis\_mask) - 1)`.

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

* **1**: ``data`` - input tensor to be sliced of type *T* and arbitrary shape. **Required.**
* **2**: ``begin`` - 1D tensor of type *T_IND* with begin indexes for input tensor slicing.  Out-of-bounds values are silently clamped. If ``begin_mask[i]`` is ``1`` , the value of ``begin[i]`` is ignored and the range of the appropriate dimension starts from ``0``. Negative values mean indexing starts from the end. **Required.**
* **3**: ``end`` - 1D tensor of type *T_IND* with end indexes for input tensor slicing. Out-of-bounds values will be silently clamped. If ``end_mask[i]`` is ``1``, the value of ``end[i]`` is ignored and the full range of the appropriate dimension is used instead. Negative values mean indexing starts from the end. **Required.**
* **4**: ``stride`` - 1D tensor of type *T_IND* with strides. If not provided, stride is assumed to be equal to 1. **Optional.**

**Outputs**:

* **1**: A tensor of type *T* with values selected by the slicing operation according to the rules specified above.

**Types**

* *T*: any supported type.
* *T_IND*: any supported integer type.

**Example**

Basic example with different strides, standard slicing and in reverse. Equivalent of performing :math:`array[0:4, 1:4, 0:4:2, 1:4:2, 3:0:-1, 3:0:-2]` on a 6D array.

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

Example of clamping in standard and reverse slicing. Equivalent of performing :math:`array[2:3, 2:1:-1]` on a 2D array.

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
                <dim>2</dim> <!-- stride: [1, -1] - second slicing is in reverse-->
            </port>
        </input>
        <output>
            <port id="4">
                <dim>1</dim> <!-- begin clamped to 2, end clamped to 3, element ids: [2] -->
                <dim>1</dim> <!-- begin clamped to 2, end clamped to 1, element ids: [2] -->
            </port>
        </output>
    </layer>

Example of negative slicing. Equivalent of performing array[0:2, 0:2, 0:-1] on a 3D array.

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
                <dim>3</dim> <!-- end: [2, 2, -1] - -1 will be replaced by 4 - 1 = 3 -->
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

Example of ``begin_mask`` & ``end_mask`` usage. Equivalent of performing :math:`array[1:, :, ::-1]` on a 3D array.

.. code-block:: xml
   :force:

    <layer ... type="StridedSlice" ...>
        <data begin_mask="0,1,1" end_mask="1,1,1" new_axis_mask="0,0,0,0,0" shrink_axis_mask="0,0" ellipsis_mask="0" />
        <input>
            <port id="0">
                <dim>2</dim>
                <dim>3</dim>
                <dim>4</dim>
            </port>
            <port id="1">
                <dim>3</dim> <!-- begin: [1, 1, 123] begin_mask ignores provided values -->
            </port>
            <port id="2">
                <dim>3</dim> <!-- end: [0, 0, 2] end_mask ignores provided values -->
            </port>
            <port id="3">
                <dim>3</dim> <!-- stride: [1, 1, -1] - last slicing is in reverse, masks' behavior changes -->
            </port>
        </input>
        <output>
            <port id="4">
                <dim>1</dim> <!-- begin = 1, end = 2 (end_mask), element ids: [1] -->
                <dim>3</dim> <!-- begin = 0 (begin_mask), end = 3 (end_mask), element ids: [0, 1, 2] -->
                <dim>3</dim> <!-- begin = 3 (begin_mask), end = 0 (end_mask), element ids: [3, 2, 1] -->
            </port>
        </output>
    </layer>

Example of ``new_axis_mask`` usage. Equivalent of performing :math:`array[np.newaxis, 0:2, np.newaxis, 0:4]` on a 2D array.

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
                <dim>4</dim> <!-- begin: [1234, 0, -1, 0] - new_axis_mask skips the value -->
            </port>
            <port id="2">
                <dim>4</dim> <!-- end: [1234, 2, 9876, 4] - new_axis_mask skips the value -->
            </port>
            <port id="3">
                <dim>4</dim> <!-- stride: [132, 1, 241, 1] - new_axis_mask skips the value -->
            </port>
        </input>
        <output>
            <port id="4">
                <dim>1</dim> <!-- new dimension appears -->
                <dim>2</dim> <!-- second dimension created from first dimension of the input -->
                <dim>1</dim> <!-- new dimension appears -->
                <dim>4</dim> <!-- fourth dimension created from second dimension of the input -->
            </port>
        </output>
    </layer>

Example of ``shrink_axis_mask`` usage. Equivalent of performing :math:`array[0:1, 0, 0:384, 0:640, 0:8]` on a 5D array.

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
                <dim>5</dim> <!-- end: [1, 0, 384, 640, 8] -->
            </port>
            <port id="3">
                <dim>5</dim> <!-- stride: [1, 1, 1, 1, 1] -->
            </port>
        </input>
        <output>
            <port id="4">
                <dim>1</dim> <!-- first dim kept, as shrink_axis_mask is 0 -->
                <dim>384</dim> <!-- second dim is missing as shrink_axis_mask is 1 -->
                <dim>640</dim>
                <dim>8</dim>
            </port>
        </output>
    </layer>

Example of ``ellipsis_mask`` usage. Equivalent of performing :math:`array[0:4, ..., 0:5]` on a 10D array.

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
                <dim>3</dim> <!-- begin: [0, 0, 0] - with second dimension marked as ellipsis -->
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
                <dim>10</dim> <!-- ellipsis skipped over 8 dimensions to match pattern -->
                <dim>10</dim>
                <dim>10</dim>
                <dim>10</dim>
                <dim>10</dim>
                <dim>5</dim> <!-- last dim modified -->
            </port>
        </output>
    </layer>

Example of ``ellipsis_mask`` usage with other masks of unequal length. Equivalent of performing :math:`array[2:, ..., np.newaxis, :10]` on a 10D array.

.. code-block:: xml
   :force:

    <layer ... type="StridedSlice" ...>
        <data begin_mask="0,0,1,1" end_mask="1,1,0,0" new_axis_mask="0,0,1" shrink_axis_mask="0" ellipsis_mask="0,1"/>
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
                <dim>3</dim> <!-- begin: [2, 1, 10, 10] - second dimension marked as ellipsis. third dimension marked as a new axis -->
            </port>
            <port id="2">
                <dim>3</dim> <!-- end: [123, 1, 10, 5] -->
            </port>
            <port id="3">
                <dim>3</dim> <!-- stride: [1, -1, 1, 1] -->
            </port>
        </input>
        <output>
            <port id="4">
                <dim>8</dim> <!-- first dim modified, begin = 2, end = 10 -->
                <dim>10</dim>
                <dim>10</dim>
                <dim>10</dim>
                <dim>10</dim> <!-- ellipsis skipped over 8 dimensions -->
                <dim>10</dim> <!-- 8 = 10 - (4 - 1 - 1) -->
                <dim>10</dim> <!-- 10 - rank(input), 4 - rank(begin), 1 - new_axis_mask -->
                <dim>10</dim>
                <dim>10</dim>
                <dim>1</dim> <!-- new dimension from new_axis_mask, 'consumes' the penultimate slicing arguments -->
                <dim>5</dim> <!-- last dim modified, begin = 0, end = 5 -->
            </port>
        </output>
    </layer>
