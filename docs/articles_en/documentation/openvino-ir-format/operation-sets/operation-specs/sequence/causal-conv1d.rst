CausalConv1D
============


.. meta::
  :description: Learn about CausalConv1D - a stateful causal 1D convolution operation used in sequence models such as Mamba.

**Versioned name**: *CausalConv1D*

**Category**: *Sequence processing*

**Short description**: *CausalConv1D* performs a causal grouped 1D convolution over an input sequence while maintaining and updating a convolution state buffer. It is primarily used in state-space models such as Mamba.

**Detailed description**:

*CausalConv1D* prepends the convolution state (a cache of past input values) to the current input, applies a grouped 1D convolution with no padding, and returns both the convolution output and the updated state. Only the last ``seq_len`` output time steps are returned, ensuring causality. The state is updated by retaining the last ``kernel_size`` time steps from the concatenated input.

The operation provides functionality according to the following pseudo-code:

.. code-block:: py
   :force:

   def CausalConv1D(input_embeds, conv_state, conv_weight, conv_bias=None):
       # input_embeds:  [batch_size, seq_len, hidden_size]
       # conv_state:    [batch_size, hidden_size, kernel_size]  (kernel_size == conv_kernel_size - 1)
       # conv_weight:   [out_channels, hidden_size / group_size, conv_kernel_size]
       # conv_bias:     [out_channels]  (optional)

       batch_size, seq_len, hidden_size = input_embeds.shape
       out_channels, w_in_channels, conv_kernel_size = conv_weight.shape
       state_len = conv_state.shape[-1]  # == conv_kernel_size - 1
       groups = hidden_size // w_in_channels

       # Transpose input to channels-first layout required by conv1d:
       # [batch_size, seq_len, hidden_size] -> [batch_size, hidden_size, seq_len]
       input_t = Transpose(input_embeds, axes=[0, 2, 1])

       # Prepend conv state and cast to weight dtype:
       # [batch_size, hidden_size, state_len + seq_len]
       input_new = ConvertLike(Concat([conv_state, input_t], axis=-1), conv_weight)

       # Grouped 1D convolution (no padding):
       # output shape: [batch_size, out_channels, state_len + seq_len - conv_kernel_size + 1]
       conv_out = grouped_conv1d(input_new, conv_weight, conv_bias, padding=0, groups=groups)

       # Keep only the last seq_len time steps:
       # [batch_size, out_channels, seq_len]
       conv_out = conv_out[:, :, -seq_len:]

       # Update state: last kernel_size values of the extended input:
       # [batch_size, hidden_size, kernel_size]
       new_conv_state = input_new[:, :, -state_len:]

       # Transpose output back to sequence-last layout:
       # [batch_size, seq_len, out_channels]  (out_channels == hidden_size)
       output_embeds = Transpose(conv_out, axes=[0, 2, 1])

       return output_embeds, new_conv_state


**Inputs**

* **1**: ``input_embeds`` - 3D tensor of type *T* and shape ``[batch_size, seq_len, hidden_size]``. The input sequence embeddings. **Required.**

* **2**: ``conv_state`` - 3D tensor of type *T* and shape ``[batch_size, hidden_size, kernel_size]``. The cached past input values used to maintain causality across inference steps. **Required.**

* **3**: ``conv_weight`` - 3D tensor of type *T* and shape ``[out_channels, hidden_size / group_size, conv_kernel_size]``. The convolution filter weights. **Required.**

* **4**: ``conv_bias`` - 1D tensor of type *T* and shape ``[out_channels]``. The convolution bias. **Optional.**


**Outputs**

* **1**: ``output_embeds`` - 3D tensor of type *T* and shape ``[batch_size, seq_len, hidden_size]``. The output sequence embeddings after applying the causal grouped convolution.

* **2**: ``output_conv_state`` - 3D tensor of type *T* and shape ``[batch_size, hidden_size, kernel_size]``. The updated convolution state containing the last ``kernel_size`` time steps of the extended input.


**Types**

* *T*: any supported floating-point type.


**Dimension symbols**

* ``batch_size`` - number of sequences in the batch.
* ``seq_len`` - length of the input sequence (number of time steps).
* ``hidden_size`` - number of channels in the input embeddings; must equal ``out_channels``.
* ``out_channels`` - number of output channels produced by the convolution; must equal ``hidden_size``.
* ``kernel_size`` - size of the convolution state buffer; always equals ``conv_kernel_size - 1``, since the state stores exactly the number of past time steps needed to apply the filter causally.
* ``conv_kernel_size`` - spatial size of the convolution kernel.
* ``group_size`` - number of input channels per convolution group; ``groups = hidden_size / group_size``.


**Example**

*Example 1: Without bias, batch_size=1, seq_len=1 (single-token decode step)*

.. code-block:: xml
   :force:

   <layer id="0" name="causal_conv1d_0" type="CausalConv1D">
       <input>
           <port id="0" precision="FP32"> <!-- input_embeds -->
               <dim>1</dim>   <!-- batch_size -->
               <dim>1</dim>   <!-- seq_len -->
               <dim>4096</dim> <!-- hidden_size -->
           </port>
           <port id="1" precision="FP32"> <!-- conv_state -->
               <dim>1</dim>   <!-- batch_size -->
               <dim>4096</dim> <!-- hidden_size -->
               <dim>3</dim>   <!-- kernel_size -->
           </port>
           <port id="2" precision="FP32"> <!-- conv_weight -->
               <dim>4096</dim> <!-- out_channels -->
               <dim>1</dim>   <!-- hidden_size / group_size (depthwise: group_size = hidden_size) -->
               <dim>4</dim>   <!-- conv_kernel_size -->
           </port>
       </input>
       <output>
           <port id="3" precision="FP32"> <!-- output_embeds -->
               <dim>1</dim>   <!-- batch_size -->
               <dim>1</dim>   <!-- seq_len -->
               <dim>4096</dim> <!-- hidden_size -->
           </port>
           <port id="4" precision="FP32"> <!-- output_conv_state -->
               <dim>1</dim>   <!-- batch_size -->
               <dim>4096</dim> <!-- hidden_size -->
               <dim>3</dim>   <!-- kernel_size -->
           </port>
       </output>
   </layer>

*Example 2: With bias, batch_size=2, seq_len=8 (prefill step)*

.. code-block:: xml
   :force:

   <layer id="1" name="causal_conv1d_1" type="CausalConv1D">
       <input>
           <port id="0" precision="FP16"> <!-- input_embeds -->
               <dim>2</dim>   <!-- batch_size -->
               <dim>8</dim>   <!-- seq_len -->
               <dim>2048</dim> <!-- hidden_size -->
           </port>
           <port id="1" precision="FP16"> <!-- conv_state -->
               <dim>2</dim>   <!-- batch_size -->
               <dim>2048</dim> <!-- hidden_size -->
               <dim>3</dim>   <!-- kernel_size -->
           </port>
           <port id="2" precision="FP16"> <!-- conv_weight -->
               <dim>2048</dim> <!-- out_channels -->
               <dim>1</dim>   <!-- hidden_size / group_size (depthwise) -->
               <dim>4</dim>   <!-- conv_kernel_size -->
           </port>
           <port id="3" precision="FP16"> <!-- conv_bias -->
               <dim>2048</dim> <!-- out_channels -->
           </port>
       </input>
       <output>
           <port id="4" precision="FP16"> <!-- output_embeds -->
               <dim>2</dim>   <!-- batch_size -->
               <dim>8</dim>   <!-- seq_len -->
               <dim>2048</dim> <!-- hidden_size -->
           </port>
           <port id="5" precision="FP16"> <!-- output_conv_state -->
               <dim>2</dim>   <!-- batch_size -->
               <dim>2048</dim> <!-- hidden_size -->
               <dim>3</dim>   <!-- kernel_size -->
           </port>
       </output>
   </layer>
