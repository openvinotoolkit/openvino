Interface PartialShapeConstructor
=================================

.. code-block:: ts

   interface PartialShapeConstructor {
       new PartialShapeConstructor(shape): PartialShape;
   }

This interface contains constructor of the :doc:`PartialShape <PartialShape>` class.

* **Defined in:**
  `addon.ts:640 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L640>`__


Constructors
#####################


.. rubric:: constructor

*

   .. code-block:: ts

       new PartialShapeConstructor(shape): PartialShape

   It constructs ``PartialShape`` with a specified string.
   Skip defining the ``shape`` parameter to create an empty shape.

   * **Parameters:**

     - ``Optional``:

       .. code-block:: ts

          shape: string

     A string representation of the shape.

   * **Returns:**  :doc:`PartialShape <PartialShape>`

   - **Defined in**
     `addon.ts:646 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L646>`__

