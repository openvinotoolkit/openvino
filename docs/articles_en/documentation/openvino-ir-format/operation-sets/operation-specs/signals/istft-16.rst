.. {#openvino_docs_ops_signals_ISTFT_16}

Inverse Short Time Fourier Transformation (ISTFT)
=================================================

.. meta::
  :description: Learn about ISTFT-16 - a signal processing operation

**Versioned name**: *ISTFT-16*

**Category**: *Signal processing*

**Short description**: *ISTFT* operation performs Inverse Short-Time Fourier Transform (complex-to-real).

**Detailed description**: *ISTFT* performs Inverse Short-Time Fourier Transform of complex-valued input tensor 
of shape ``[fft_results, frames, 2]`` or ``[batch, fft_results, frames, 2]``, where:

  * ``batch`` is a batch size dimension
  * ``frames`` is a number of frames calculated as ``((signal_length - frame_size) / frame_step) + 1`` of the original signal if not centered, or ``(signal_length / frame_step) + 1`` otherwise.
  * ``fft_results`` is a number calculated as ``(frame_size / 2) + 1`` of the original signal
  * ``2`` is the last dimension for complex value represented by floating-point values pair (real and imaginary part accordingly)

The output is a restored real-valued signal in a discrete time domain. The shape of the output is 1D ``[signal_length]`` or 2D ``[batch, signal_length]``.
If the ``signal_length`` is not provided as an input value, it is calculated according to the following rules:

  * ``default_signal_length = (frames - 1) * frame_step`` for ``center == true`` 
  * ``default_signal_length = (frames - 1) * frame_step + frame_size`` for ``center == false`` 

If the ``signal_length`` input is provided, the number of output values will be adjusted accordingly. 
  * If ``signal_length > default_signal_length`` the output is padded with zeros at the end.
  * If ``signal_length < default_signal_length`` any additional generated samples are cut to the ``signal_length`` size.

The ``window_length`` can not be larger than ``frame_size``, but if smaller the window values will be padded with zeros on the left and right side. The size of the left padding is calculated as ``(frame_size - window_length) // 2``, then right padding size is filled to match the ``frame_size``.  

**Attributes**:

* *center*

  * **Description**: Flag that indicates whether padding has been applied to the original signal. It affects output shape, if the ``signal_length`` input is not provided.
  * **Range of values**:

    * ``false`` - padding has not been applied, default signal length is calculated as ``(frames - 1) * frame_step + frame_size``
    * ``true`` - padding has been applied, default signal length is calculated as ``(frames - 1) * frame_step``
  * **Type**: ``boolean``
  * **Required**: *yes*

* *normalized*

  * **Description**: Flag that indicates whether the input has been normalized. It is needed to correctly restore the signal and denormalize the output. Output of the STFT is divided by ``sqrt(frame_size)``, when normalized.
  * **Range of values**:

    * ``false`` - input has not been normalized
    * ``true`` - input has been normalized
  * **Type**: ``boolean``
  * **Required**: *yes*


**Inputs**

* **1**: ``data`` - Tensor of type *T*, the ISTFT data input (compatible with a result of STFT operation). **Required.**

  * The data input shape can be 3D ``[fft_results, frames, 2]`` or 4D ``[batch, fft_results, frames, 2]``.
* **2**: ``window`` - Tensor of type *T* and 1D shape ``[window_length]``, specifying the window values applied to restore the signal. The ``window_length`` is required to be equal or smaller than ``frame_size``, if smaller the window will be padded with zeros on the left and right sides. **Required.**
* **3**: ``frame_size`` - Scalar tensor of type *T_INT* describing the size of a single frame of the signal to be provided as input to FFT. **Required.**
* **4**: ``frame_step`` - Scalar tensor of type *T_INT* describing the distance (number of samples) between successive frames. **Required.**
* **5**: ``signal_length`` - Scalar or single element 1D tensor of type *T_INT* describing the desired length of the output signal, if not provided it's calculated accordingly to the rules presented in the detailed description above. **Optional.**


**Outputs**

* **1**: ``signal`` - Tensor of type *T* and 1D shape ``[signal_length]`` or 2D shape ``[batch, signal_length]`` with a real valued signal data. **Required.**

**Types**

* *T*: any supported floating-point type.

* *T_INT*: ``int64`` or ``int32``.


**Examples**:

*Example 3D input, 1D output signal, center=false, default signal_length:*

.. code-block:: xml
   :force:

    <layer ... type="ISTFT" ... >
        <data center="false" ... />
        <input>
            <port id="0">
                <dim>6</dim>
                <dim>16</dim>
                <dim>2</dim>
            </port>
            <port id="1">
                <dim>7</dim>
            </port>
            <port id="2"></port> <!-- frame_size value: 11 -->
            <port id="3"></port> <!-- frame_step value: 3 -->
        </input>
        <output>
            <port id="4">
                <dim>56</dim>
            </port>
        </output>
    </layer>

*Example 4D input, 2D output signal, center=false, default signal_length:*

.. code-block:: xml
   :force:

    <layer ... type="ISTFT" ... >
        <data center="false" ... />
        <input>
            <port id="0">
                <dim>4</dim>
                <dim>6</dim>
                <dim>16</dim>
                <dim>2</dim>
            </port>
            <port id="1">
                <dim>7</dim>
            </port>
            <port id="2"></port> <!-- frame_size value: 11 -->
            <port id="3"></port> <!-- frame_step value: 3 -->
        </input>
        <output>
            <port id="4">
                <dim>4</dim>
                <dim>56</dim>
            </port>
        </output>
    </layer>


*Example 3D input, 1D output signal, center=true, default signal_length:*

.. code-block:: xml
   :force:

    <layer ... type="ISTFT" ... >
        <data center="true" ... />
        <input>
            <port id="0">
                <dim>6</dim>
                <dim>16</dim>
                <dim>2</dim>
            </port>
            <port id="1">
                <dim>7</dim>
            </port>
            <port id="2"></port> <!-- frame_size value: 11 -->
            <port id="3"></port> <!-- frame_step value: 3 -->
        </input>
        <output>
            <port id="4">
                <dim>45</dim>
            </port>
        </output>
    </layer>

*Example 4D input, 2D output signal, center=true, default signal_length:*

.. code-block:: xml
   :force:

    <layer ... type="ISTFT" ... >
        <data center="true" ... />
        <input>
            <port id="0">
                <dim>4</dim>
                <dim>6</dim>
                <dim>16</dim>
                <dim>2</dim>
            </port>
            <port id="1">
                <dim>7</dim>
            </port>
            <port id="2"></port> <!-- frame_size value: 11 -->
            <port id="3"></port> <!-- frame_step value: 3 -->
        </input>
        <output>
            <port id="4">
                <dim>4</dim>
                <dim>45</dim>
            </port>
        </output>
    </layer>


*Example 3D input, 1D output signal, center=false, signal_length input provided:*

.. code-block:: xml
   :force:

    <layer ... type="ISTFT" ... >
        <data center="false" ... />
        <input>
            <port id="0">
                <dim>6</dim>
                <dim>16</dim>
                <dim>2</dim>
            </port>
            <port id="1">
                <dim>7</dim>
            </port>
            <port id="2"></port> <!-- frame_size value: 11 -->
            <port id="3"></port> <!-- frame_step value: 3 -->
            <port id="4"></port> <!-- signal_length value: 64 -->
        </input>
        <output>
            <port id="5">
                <dim>64</dim>
            </port>
        </output>
    </layer>
