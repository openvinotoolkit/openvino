.. {#openvino_docs_ops_signals_STFT_15}

Short Time Fourier Transformation for real-valued input (STFT)
==============================================================


.. meta::
  :description: Learn about STFT-15 - a signal processing operation

**Versioned name**: *STFT-15*

**Category**: *Signal processing*

**Short description**: *STFT* operation performs Short-Time Fourier Transform (real-to-complex).


**Detailed description**: *STFT* performs Short-Time Fourier Transform of real-valued input tensor of shape ``[signal_size]`` or ``[batch, signal_size]``, and produces complex result represented by separate values for real and imaginary part.


**Attributes**:

* *transpose_frames*

  * **Description**: Flag to set output shape layout. If true the ``frames`` dimension is at out_shape[-2], otherwise it is at out_shape[-3].
  * **Range of values**:

    * ``false`` - do not transpose output shape
    * ``true`` - transpose output shape
  * **Type**: ``boolean``
  * **Required**: *yes*

**Inputs**

* **1**: ``signal`` - Tensor of type *T* and 1D shape [signal_size] or 2D shape [batch, signal_size] with signal data for the STFT. **Required.**
* **2**: ``window`` - Tensor of type *T* and 1D shape [window_length], specifying the window values for the signal slice multiplication. **Required.**
* **3**: ``frame_size`` - Scalar tensor of type *T_INT* describing the size of a single frame of the signal to be provided as input to FFT. **Required.**
* **4**: ``frame_step`` - Scalar tensor of type *T_INT* describing The distance (number of samples) between successive frames. **Required.**


**Outputs**

* **1**: The result of STFT operation, tensor of the same type as input ``signal`` tensor and shape:

  * When ``transpose_frames == false`` the output shape is ``[frames, fft_results, 2]`` for 1D signal input or ``[batch, frames, fft_results, 2]`` for 2D signal input.
  * When ``transpose_frames == true`` the output shape is ``[fft_results, frames, 2]`` for 1D signal input or ``[batch, fft_results, frames, 2]`` for 2D signal input.

  where:

  * ``batch`` is a batch size dimension
  * ``frames`` is a number calculated as ``(signal_shape[-1] - frame_size) / frame_step) + 1``
  * ``fft_results`` is a number calculated as ``(frame_size / 2) + 1``
  * ``2`` is the last dimension is for complex value real and imaginary part


**Types**

* *T*: any supported floating-point type.

* *T_INT*: ``int64`` or ``int32``.


**Examples**:

*Example 1D signal, transpose_frames=false:*

.. code-block:: xml
   :force:

    <layer ... type="STFT" ... >
        <data transpose_frames="false"/>
        <input>
            <port id="0">
                <dim>56</dim>
            </port>
            <port id="1">
                <dim>7</dim>
            </port>
            <port id="2"></port> <!-- value: 11 -->
            <port id="3"></port> <!-- value: 3 -->
        <output>
            <port id="4">
                <dim>16</dim>
                <dim>6</dim>
                <dim>2</dim>
            </port>
        </output>
    </layer>


*Example 1D signal, transpose_frames=true:*

.. code-block:: xml
   :force:

    <layer ... type="STFT" ... >
        <data transpose_frames="true"/>
        <input>
            <port id="0">
                <dim>56</dim>
            </port>
            <port id="1">
                <dim>7</dim>
            </port>
            <port id="2"></port> <!-- value: 11 -->
            <port id="3"></port> <!-- value: 3 -->
        <output>
            <port id="4">
                <dim>6</dim>
                <dim>16</dim>
                <dim>2</dim>
            </port>
        </output>
    </layer>

*Example 2D signal, transpose_frames=false:*

.. code-block:: xml
   :force:

    <layer ... type="STFT" ... >
        <data transpose_frames="false"/>
        <input>
            <port id="0">
                <dim>3</dim>
                <dim>56</dim>
            </port>
            <port id="1">
                <dim>7</dim>
            </port>
            <port id="2"></port> <!-- value: 11 -->
            <port id="3"></port> <!-- value: 3 -->
        <output>
            <port id="4">
                <dim>3</dim>
                <dim>16</dim>
                <dim>6</dim>
                <dim>2</dim>
            </port>
        </output>
    </layer>


*Example 2D signal, transpose_frames=true:*

.. code-block:: xml
   :force:

    <layer ... type="STFT" ... >
        <data transpose_frames="true"/>
        <input>
            <port id="0">
                <dim>3</dim>
                <dim>56</dim>
            </port>
            <port id="1">
                <dim>7</dim>
            </port>
            <port id="2"></port> <!-- value: 11 -->
            <port id="3"></port> <!-- value: 3 -->
        <output>
            <port id="4">
                <dim>3</dim>
                <dim>6</dim>
                <dim>16</dim>
                <dim>2</dim>
            </port>
        </output>
    </layer>
