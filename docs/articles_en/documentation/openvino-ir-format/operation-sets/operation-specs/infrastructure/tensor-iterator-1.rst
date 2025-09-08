TensorIterator
==============


.. meta::
  :description: Learn about TensorIterator-1 - an infrastructure operation, which
                can be performed on multiple input tensors of any supported type and shape.

**Versioned name**: *TensorIterator-1*

**Category**: *Infrastructure*

**Short description**: *TensorIterator* layer performs recurrent execution of the network, which is described in the ``body``, iterating through the data.

**TensorIterator attributes**:

* **Body**:

  ``body`` is a network that will be recurrently executed. The network is described layer by layer as a typical IR network.

  * **Body attributes**:

    No attributes available.

* **Port map**:

  *port_map* is a set of rules to map input or output data tensors of ``TensorIterator`` layer onto ``body`` data tensors. The ``port_map`` entries can be ``input`` and ``output``. Each entry describes a corresponding mapping rule.

  * **Port map attributes**:

    * *external_port_id*

      * **Description**: *external_port_id* is a port ID of the ``TensorIterator`` layer.
      * **Range of values**: indexes of the *TensorIterator* outputs
      * **Type**: ``int``
      * **Default value**: None
      * **Required**: *yes*

    * *internal_layer_id*

      * **Description**: *internal_layer_id* is a *Parameter* or *Result* layer ID inside the ``body`` network to map to.
      * **Range of values**: IDs of the *Parameter* layers inside in the *TensorIterator* layer
      * **Type**: ``int``
      * **Default value**: None
      * **Required**: *yes*

    * *axis*

      * **Description**: *axis* is an axis to iterate through. It triggers the slicing of this tensor. Only if it is specified, the corresponding ``input`` or ``output`` is divided into pieces and start, end and stride attributes define how slicing is performed.
      * **Range of values**: an integer
      * **Type**: ``int``
      * **Default value**: None
      * **Required**: *no*

    * *start*

      * **Description**: *start* is an index where the iteration starts from. Negative value means counting indexes from the end. Applies only when the attribute ``axis`` is specified.
      * **Range of values**: an integer
      * **Type**: ``int``
      * **Default value**: 0
      * **Required**: *no*

    * *end*

      * **Description**: *end* is an index where iteration ends. Negative value means counting indexes from the end. Applies only when the attribute ``axis`` is specified.
      * **Range of values**: an integer
      * **Type**: ``int``
      * **Default value**: -1
      * **Required**: *no*

    * *stride*

      * **Description**: *stride* is a step of iteration. Negative value means backward iteration. Applies only when the attribute ``axis`` is specified.
      * **Range of values**: an integer
      * **Type**: ``int``
      * **Default value**: 1
      * **Required**: *no*


* **Back edges**:

  *back_edges* is a set of rules to transfer tensor values from ``body`` outputs at one iteration to ``body`` parameters at the next iteration. Back edge connects some *Result* layer in ``body`` to *Parameter* layer in the same ``body``.

  * **Back edge attributes**:

    * *from-layer*

      * **Description**: *from-layer* is a *Result* layer ID inside the ``body`` network.
      * **Range of values**: IDs of the *Result* layers inside the *TensorIterator*
      * **Type**: ``int``
      * **Default value**: None
      * **Required**: *yes*

    * *to-layer*

      * **Description**: *to-layer* is a *Parameter* layer ID inside the ``body`` network to end mapping.
      * **Range of values**: IDs of the *Parameter* layers inside the *TensorIterator*
      * **Type**: ``int``
      * **Default value**: None
      * **Required**: *yes*

**Inputs**

* **Multiple inputs**: Tensors of any type and shape supported type.

**Outputs**

* **Multiple outputs**: Results of execution of the ``body``. Tensors of any type and shape.


**Detailed description**

Similar to other layers, TensorIterator has regular sections: ``input`` and ``output``. It allows connecting TensorIterator to the rest of the IR.
TensorIterator also has several special sections: ``body``, ``port_map``, ``back_edges``. The principles of their work are described below.

How ``body`` is iterated:

*At the first iteration:* TensorIterator slices input tensors by a specified axis and iterates over all parts in a specified order. It process input tensors with arbitrary network specified as an IR network in the ``body`` section. IR is executed as no back-edges are present. Edges from ``port map`` are used to connect input ports of TensorIterator to ``Parameters`` in body.

[``inputs``] - ``Port map`` edges -> [``Parameters:body:Results``]

``Parameter`` and ``Result`` layers are part of the ``body``. ``Parameters`` are stable entry points in the ``body``. The results of the execution of the ``body`` are presented as stable ``Result`` layers. Stable means that these nodes cannot be fused.

*Next iterations:*
Back edges define which data is copied back to ``Parameters`` layers from ``Results`` layers between IR iterations in TensorIterator ``body``. That means they pass data from source layer back to target layer. Each layer that is a target for back-edge has also an incoming ``port map`` edge as an input. The values from back-edges are used instead of corresponding edges from ``port map``. After each iteration of the network, all back edges are executed.
Iterations can be considered as statically unrolled sequence: all edges that flow between two neighbor iterations are back-edges. So in the unrolled loop, each back-edge is transformed to regular edge.

... -> [``Parameters:body:Results``] - back-edges -> [``Parameters:body:Results``] - back-edges -> [``Parameters:body:Results``] - back-edges -> ...

*Calculation of results:*

If ``output`` entry in the ``Port map`` doesn't have partitioning (``axis, begin, end, strides``) attributes, then the final value of ``output`` of TensorIterator is the value of ``Result`` node from the last iteration. Otherwise the final value of ``output`` of TensorIterator is a concatenation of tensors in the ``Result`` node for all ``body`` iterations. Concatenation order is specified by ``stride`` attribute.

The last iteration:

[``Parameters:body:Results``] - ``Port map`` edges -> [``outputs``],  if partitioning attributes are not set.

if there are partitioning attributes, then an output tensor is a concatenation of tensors from all body iterations. If ``stride > 0``:

.. code-block:: cpp

    output = Concat(S[0], S[1], ..., S[N-1])

where ``Si`` is value of ``Result`` operation at i-th iteration in the tensor iterator body that corresponds to this output port. If ``stride < 0``, then output is concatenated in a reverse order:

.. code-block:: cpp

    output = Concat(S[N-1], S[N-2], ..., S[0])

**Examples**

*Example 1: a typical TensorIterator structure*

.. code-block:: xml
   :force:

    <layer type="TensorIterator" ... >
        <input> ... </input>
        <output> ... </output>
        <port_map>
            <input external_port_id="0" internal_layer_id="0" axis="1" start="-1" end="0" stride="-1"/>
            <input external_port_id="1" internal_layer_id="1"/>
            ...
            <output external_port_id="3" internal_layer_id="2" axis="1" start="-1" end="0" stride="-1"/>
            ...
        </port_map>
        <back_edges>
            <edge from-layer="1" to-layer="1"/>
            ...
        </back_edges>
        <body>
            <layers> ... </layers>
            <edges> ... </edges>
        </body>
    </layer>


*Example 2: a full TensorIterator layer*

.. code-block:: xml
   :force:

    <layer type="TensorIterator" ...>
        <input>
            <port id="0">
                <dim>1</dim>
                <dim>25</dim>
                <dim>512</dim>
            </port>
            <port id="1">
                <dim>1</dim>
                <dim>256</dim>
            </port>
            <port id="2">
                <dim>1</dim>
                <dim>256</dim>
            </port>
        </input>
        <output>
            <port id="3" precision="FP32">
                <dim>1</dim>
                <dim>25</dim>
                <dim>256</dim>
            </port>
        </output>
        <port_map>
            <input axis="1" external_port_id="0" internal_layer_id="0" start="0"/>
            <input external_port_id="1" internal_layer_id="3"/>
            <input external_port_id="2" internal_layer_id="4"/>
            <output axis="1" external_port_id="3" internal_layer_id="12"/>
        </port_map>
        <back_edges>
            <edge from-layer="8" to-layer="4"/>
            <edge from-layer="9" to-layer="3"/>
        </back_edges>
        <body>
            <layers>
                <layer id="0" type="Parameter" ...>
                    <output>
                        <port id="0" precision="FP32">
                            <dim>1</dim>
                            <dim>1</dim>
                            <dim>512</dim>
                        </port>
                    </output>
                </layer>
                <layer id="1" type="Const" ...>
                    <data offset="0" size="16"/>
                    <output>
                        <port id="1" precision="I64">
                            <dim>2</dim>
                        </port>
                    </output>
                </layer>
                <layer id="2" type="Reshape" ...>
                    <input>
                        <port id="0">
                            <dim>1</dim>
                            <dim>1</dim>
                            <dim>512</dim>
                        </port>
                        <port id="1">
                            <dim>2</dim>
                        </port>
                    </input>
                    <output>
                        <port id="2" precision="FP32">
                            <dim>1</dim>
                            <dim>512</dim>
                        </port>
                    </output>
                </layer>
                <layer id="3" type="Parameter" ...>
                    <output>
                        <port id="0" precision="FP32">
                            <dim>1</dim>
                            <dim>256</dim>
                        </port>
                    </output>
                </layer>
                <layer id="4" type="Parameter" ...>
                    <output>
                        <port id="0" precision="FP32">
                            <dim>1</dim>
                            <dim>256</dim>
                        </port>
                    </output>
                </layer>
                <layer id="5" type="Const" ...>
                    <data offset="16" size="3145728"/>
                    <output>
                        <port id="1" precision="FP32">
                            <dim>1024</dim>
                            <dim>768</dim>
                        </port>
                    </output>
                </layer>
                <layer id="6" type="Const" ...>
                    <data offset="3145744" size="4096"/>
                    <output>
                        <port id="1" precision="FP32">
                            <dim>1024</dim>
                        </port>
                    </output>
                </layer>
                <layer id="7" type="LSTMCell" ...>
                    <data hidden_size="256"/>
                    <input>
                        <port id="0">
                            <dim>1</dim>
                            <dim>512</dim>
                        </port>
                        <port id="1">
                            <dim>1</dim>
                            <dim>256</dim>
                        </port>
                        <port id="2">
                            <dim>1</dim>
                            <dim>256</dim>
                        </port>
                        <port id="3">
                            <dim>1024</dim>
                            <dim>768</dim>
                        </port>
                        <port id="4">
                            <dim>1024</dim>
                        </port>
                    </input>
                    <output>
                        <port id="5" precision="FP32">
                            <dim>1</dim>
                            <dim>256</dim>
                        </port>
                        <port id="6" precision="FP32">
                            <dim>1</dim>
                            <dim>256</dim>
                        </port>
                    </output>
                </layer>
                <layer id="8" type="Result" ...>
                    <input>
                        <port id="0">
                            <dim>1</dim>
                            <dim>256</dim>
                        </port>
                    </input>
                </layer>
                <layer id="9" type="Result" ...>
                    <input>
                        <port id="0">
                            <dim>1</dim>
                            <dim>256</dim>
                        </port>
                    </input>
                </layer>
                <layer id="10" type="Const" ...>
                    <data offset="3149840" size="24"/>
                    <output>
                        <port id="1" precision="I64">
                            <dim>3</dim>
                        </port>
                    </output>
                </layer>
                <layer id="11" type="Reshape" ...>
                    <input>
                        <port id="0">
                            <dim>1</dim>
                            <dim>256</dim>
                        </port>
                        <port id="1">
                            <dim>3</dim>
                        </port>
                    </input>
                    <output>
                        <port id="2" precision="FP32">
                            <dim>1</dim>
                            <dim>1</dim>
                            <dim>256</dim>
                        </port>
                    </output>
                </layer>
                <layer id="12" type="Result" ...>
                    <input>
                        <port id="0">
                            <dim>1</dim>
                            <dim>1</dim>
                            <dim>256</dim>
                        </port>
                    </input>
                </layer>
            </layers>
            <edges>
                <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
                <edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
                <edge from-layer="2" from-port="2" to-layer="7" to-port="0"/>
                <edge from-layer="3" from-port="0" to-layer="7" to-port="1"/>
                <edge from-layer="4" from-port="0" to-layer="7" to-port="2"/>
                <edge from-layer="5" from-port="1" to-layer="7" to-port="3"/>
                <edge from-layer="6" from-port="1" to-layer="7" to-port="4"/>
                <edge from-layer="7" from-port="6" to-layer="8" to-port="0"/>
                <edge from-layer="7" from-port="5" to-layer="9" to-port="0"/>
                <edge from-layer="7" from-port="5" to-layer="11" to-port="0"/>
                <edge from-layer="10" from-port="1" to-layer="11" to-port="1"/>
                <edge from-layer="11" from-port="2" to-layer="12" to-port="0"/>
            </edges>
        </body>
    </layer>


