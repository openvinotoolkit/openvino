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
       _, hidden_size, seq_len = input_embeds.shape
       _, w_in_channels, _ = weight.shape
       state_len = conv_state.shape[-1]
       groups = hidden_size // w_in_channels

       input_embeds_new = torch.cat([conv_state, input_embeds], dim=-1).to(weight.dtype)
       conv_out = F.conv1d(input_embeds_new, weight, bias, padding=0, groups=groups)
       conv_out = conv_out[:, :, -seq_len:]

       new_conv_state = input_embeds_new[:, :, -state_len:]

       return conv_out, new_conv_state


**Inputs**

* **1**: ``input_embeds`` - 3D tensor of type *T* and shape ``[batch_size, hidden_size, seq_len]``. The input sequence embeddings. **Required.**

* **2**: ``conv_state`` - 3D tensor of type *T* and shape ``[batch_size, hidden_size, kernel_size]``. The cached past input values used to maintain causality across inference steps. **Required.**

* **3**: ``conv_weight`` - 3D tensor of type *T* and shape ``[out_channels, hidden_size / group_size, conv_kernel_size]``. The convolution filter weights. **Required.**

* **4**: ``conv_bias`` - 1D tensor of type *T* and shape ``[out_channels]``. The convolution bias. **Optional.**


**Outputs**

* **1**: ``output_embeds`` - 3D tensor of type *T* and shape ``[batch_size, seq_len, hidden_size]``. The output sequence embeddings after applying the causal grouped convolution.

* **2**: ``output_conv_state`` - 3D tensor of type *T* and shape ``[batch_size, hidden_size, kernel_size]``. The updated convolution state containing the last ``kernel_size`` time steps of the extended input.


**Types**

* *T*: any supported floating-point type.


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
