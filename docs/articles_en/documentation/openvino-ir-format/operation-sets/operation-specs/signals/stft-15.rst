.. {#openvino_docs_ops_signals_STFT_15}

Short Time Fourier Transformation for real-valued input (STFT)
============================================================


.. meta::
  :description: Learn about STFT-15 - a signal processing operation

**Versioned name**: *STFT-15*

**Category**: *Signal processing*

**Short description**: *STFT* operation performs Short-Time Fourier Transform (real-to-complex).


**Detailed description**: *STFT* performs Short-Time Fourier Transform of real-valued batched input tensor of shape ``[batch, signal_size]``, and produce complex result represented by separate values for real and imaginary part.


**Attributes**:

* *transform_frames*

  * **Description**: Flag to set output shape layout. If true the ``frames`` dimension is at out_shape[2], otherwise it is at out_shape[1].
  * **Range of values**:

    * ``false`` - do not transpose output shape
    * ``true`` - transpose output shape
  * **Type**: ``boolean``
  * **Required**: *yes*

**Inputs**

*   **1**: ``signal`` - Tensor of type *T* and 2D shape [batch, signal_size] with signal data for the STFT. **Required.**
*   **2**: ``window`` - Tensor of type *T* and 1D shape [window_length], specifying the window values for the signal slice multiplication. **Required.**
*   **3**: ``frame_size`` - Scalar tensor of type *T_INT* describing the size of a single frame of the signal to be provided as input to FFT. **Required.**
*   **4**: ``frame_step`` - Scalar tensor of type *T_INT* describing The distance (number of samples) between successive frames. **Required.**


**Outputs**

*   **1**: The result of STFT operation, tensor of the same type as input ``signal`` tensor and shape:

    + When ``transform_frames == false`` the output shape is ``[batch, frames, fft_results, 2]``
    + When ``transform_frames == true`` the output shape is ``[batch, fft_results, frames, 2]``

    where:

    + ``batch`` is a batch size dimension
    + ``frames`` is a number calculated as ``(signal_shape[1] - frame_size) / frame_step) + 1``
    + ``fft_results`` is a number calculated as ``(frame_size / 2) + 1``
    + ``2`` is the last dimension is for complex value real and imaginary part


**Types**

* *T*: any supported floating-point type.

* *T_INT*: ``int64`` or ``int32``.


**Example**:

There is no ``signal_size`` input (3D input tensor):

.. code-block:: xml
   :force:

    <layer ... type="STFT" ... >
        <input>
            <port id="0">
                <dim>2</dim>
                <dim>48</dim>
            </port>
            <port id="1">
                <dim>8</dim>
            </port>
            <port id="2"></port>
            <port id="3"></port>
        <output>
            <port id="2">
                <dim>2</dim>
                <dim>9</dim>
                <dim>9</dim>
                <dim>2</dim>
            </port>
        </output>
    </layer>
