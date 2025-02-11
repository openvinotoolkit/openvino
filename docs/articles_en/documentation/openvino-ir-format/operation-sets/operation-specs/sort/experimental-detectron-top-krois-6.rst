ExperimentalDetectronTopKROIs
=============================


.. meta::
  :description: Learn about ExperimentalDetectronTopKROIs-6 - a sorting and
                maximization operation, which can be performed on two required
                input tensors.

**Versioned name**: *ExperimentalDetectronTopKROIs-6*

**Category**: *Sorting and maximization*

**Short description**: The *ExperimentalDetectronTopKROIs* operation is TopK operation applied to probabilities of input
ROIs.

**Detailed description**: The operation performs probabilities descending sorting for input ROIs and returns *max_rois*
number of ROIs. Order of sorted ROIs with equal probabilities is undefined. If the number of ROIs is less than *max_rois*
then operation returns all ROIs descended sorted and the output tensor is filled with undefined values for the rest of
output tensor elements.

**Attributes**:

* *max_rois*

  * **Description**: The *max_rois* attribute specifies maximal numbers of output ROIs.
  * **Range of values**: non-negative integer number
  * **Type**: ``int``
  * **Default value**: 0
  * **Required**: *no*

**Inputs**

* **1**: A 2D tensor of type *T* with shape ``[number_of_ROIs, 4]`` describing the ROIs as 4-tuples: [x 1, y 1, x 2, y 2]. **Required.**
* **2**: A 1D tensor of type *T* with shape ``[number_of_input_ROIs]`` contains probabilities for input ROIs. **Required.**

**Outputs**

* **1**: A 2D tensor of type *T* with shape ``[max_rois, 4]`` describing *max_rois* ROIs with highest probabilities.

**Types**

* *T*: any supported floating-point type.

**Example**

.. code-block:: xml
   :force:

   <layer ... type="ExperimentalDetectronTopKROIs" version="opset6">
       <data max_rois="1000"/>
       <input>
           <port id="0">
               <dim>5000</dim>
               <dim>4</dim>
           </port>
           <port id="1">
               <dim>5000</dim>
           </port>
       </input>
       <output>
           <port id="2" precision="FP32">
               <dim>1000</dim>
               <dim>4</dim>
           </port>
       </output>
   </layer>


