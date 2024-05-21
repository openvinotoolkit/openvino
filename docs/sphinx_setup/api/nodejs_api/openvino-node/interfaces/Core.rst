Interface Core
==============

.. code-block:: ts

   interface Core {
       compileModel(model, deviceName, config?): Promise<CompiledModel>;
       compileModelSync(model, deviceName, config?): CompiledModel;
       getAvailableDevices(): string[];
       getProperty(propertyName): string | number | boolean;
       getProperty(deviceName, propertyName): string | number | boolean;
       importModelSync(modelStream, device): CompiledModel;
       readModel(modelPath, weightsPath?): Promise<Model>;
       readModel(modelBuffer, weightsBuffer?): Promise<Model>;
       readModelSync(modelPath, weightsPath?): Model;
       readModelSync(modelBuffer, weightsBuffer?): Model;
       setProperty(props): void;
       setProperty(deviceName, props): void;
   }


* **Defined in:**
  `addon.ts:23 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L23>`__


Methods
#####################


.. rubric:: compileModel

.. container:: m-4

   .. code-block:: ts

      compileModel(model, device, config?): Promise<CompiledModel>

   * **Parameters:**

     -  model: :doc:`Model <Model>`
     -  device: string
     - ``Optional``

       .. code-block:: ts

          config: {
              [option: string]: string;
          }

     - [option: string]:string

   * **Returns:** Promise<\ :doc:`CompiledModel <CompiledModel>` \>

   * **Defined in:**
     `addon.ts:24 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L24>`__


.. rubric:: compileModelSync

.. container:: m-4

   .. code-block:: ts

      compileModelSync(model, device, config?): CompiledModel

   * **Parameters:**

     - model: :doc:`Model <Model>`
     - device: string
     - ``Optional``

       .. code-block:: ts

          config: {
                    [option: string]: string;
           }

     - [option: string]:string

   * **Returns:** :doc:`CompiledModel <CompiledModel>`

   * **Defined in:**
     `addon.ts:29 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L29>`__


.. rubric:: getAvailableDevices

.. container:: m-4

   .. code-block:: ts

      getAvailableDevices(): string[]

   * **Returns:** string[]

   * **Defined in:**
     `addon.ts:45 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L45>`__


.. rubric:: getProperty

.. container:: m-4

   .. code-block:: ts

      getProperty(propertyName): string | number | boolean

   * **Parameters:**

     - propertyName: string

   * **Returns:**  string | number | boolean

   * **Defined in:**
     `addon.ts:57 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L57>`__

.. container:: m-4

   .. code-block:: ts

      getProperty(deviceName, propertyName): string | number | boolean

   * **Parameters:**

     - deviceName: string
     - propertyName: string

   * **Returns:**  string | number | boolean

   * **Defined in:**
     `addon.ts:58 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L58>`__


.. rubric:: importModelSync

.. container:: m-4

   .. code-block:: ts

      importModelSync(modelStream, device): CompiledModel

   * **Parameters:**

     - modelStream: Buffer
     - device: string

   * **Returns:** CompiledModel

   * **Defined in:**
     `addon.ts:39 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L39>`__

.. container:: m-4

   .. code-block:: ts

      importModelSync(modelStream: Buffer, device: string, props: { [key: string]: string | number | boolean }): CompiledModel

   .. container:: m-4

      **Parameters:**

      - modelStream: Buffer
      - device: string
      -

        .. code-block:: ts

           props: {
                    [key: string]: string | number | boolean;
           }

      **Returns:** CompiledModel

   * **Defined in:**
     `addon.ts:40 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L40>`__


.. rubric:: readModel

.. container:: m-4

   .. code-block:: ts

      readModel(modelPath, weightsPath?): Promise<Model>

   * **Parameters:**

     - modelPath: string
     - ``Optional``

       .. code-block:: ts

          weightsPath: string

   * **Returns:**  Promise<\ :doc:`Model <Model>`\ >

   * **Defined in:**
     `addon.ts:34 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L34>`__

.. container:: m-4

   .. code-block:: ts

      readModel(modelBuffer, weightsBuffer?): Promise<Model>

   * **Parameters:**

     - modelBuffer: Uint8Array
     - ``Optional``

       .. code-block:: ts

          weightsBuffer: Uint8Array

   * **Returns:**  Promise<\ :doc:`Model <Model>`\ >

   * **Defined in:**
     `addon.ts:35 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L35>`__


.. rubric:: readModelSync

.. container:: m-4

   .. code-block:: ts

      readModelSync(modelPath, weightsPath?): Model

   * **Parameters:**

     - modelPath: string
     - ``Optional``

       .. code-block:: ts

          weightsPath: string

   * **Returns:**  :doc:`Model <Model>`

   * **Defined in:**
     `addon.ts:37 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L37>`__

.. container:: m-4

   .. code-block:: ts

      readModelSync(modelBuffer, weightsBuffer?): Model

   * **Parameters:**

     - modelBuffer: Uint8Array
     - ``Optional``

       .. code-block:: ts

          weightsBuffer: Uint8Array

   * **Returns:**  :doc:`Model <Model>`

   * **Defined in:**
     `addon.ts:38 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L38>`__


.. rubric:: setProperty

.. container:: m-4

   .. code-block:: ts

      setProperty(props): void

   * **Parameters:**

     -

       .. code-block:: ts

          props: {
                   [key: string]: string | number | boolean;
          }

       - [key: string]: string | number | boolean

   * **Returns:**  void

   * **Defined in:**
     `addon.ts:52 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L52>`__

.. container:: m-4

   .. code-block:: ts

      setProperty(deviceName, props): void

   * **Parameters:**

     - deviceName: string
     -

       .. code-block:: ts

          props: {
                   [key: string]: string | number | boolean;
          }

       - [key: string]: string | number | boolean

   * **Returns:**  string | number | boolean

   * **Defined in:**
     `addon.ts:53 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L53>`__

