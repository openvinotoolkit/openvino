Interface Core
==============

Core represents an OpenVINO runtime Core entity.
User applications can create several Core class instances,
but in this case, the underlying plugins
are created multiple times and not shared between several Core instances.
It is recommended to have a single Core instance per application.

.. code-block:: ts

   interface Core {
       addExtension(libraryPath): void;
       compileModel(model, deviceName, config?): Promise<CompiledModel>;
       compileModel(modelPath, deviceName, config?): Promise<CompiledModel>;
       compileModelSync(model, deviceName, config?): CompiledModel;
       compileModelSync(modelPath, deviceName, config?): CompiledModel;
       getAvailableDevices(): string[];
       getProperty(propertyName): string | number | boolean;
       getProperty(deviceName, propertyName): string | number | boolean;
       getVersions(deviceName): {
           [deviceName: string]: {
               buildNumber: string;
               description: string;
           };
       };
       importModelSync(modelStream, device, config?): CompiledModel;
       readModel(modelPath, weightsPath?): Promise<Model>;
       readModel(modelBuffer, weightsBuffer?): Promise<Model>;
       readModelSync(modelPath, weightsPath?): Model;
       readModelSync(modelBuffer, weightsBuffer?): Model;
       setProperty(properties): void;
       setProperty(deviceName, properties): void;
   }


* **Defined in:**
  `addon.ts:32 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L32>`__


Methods
#####################


.. rubric:: addExtension

.. container:: m-4

   .. code-block:: ts

      addExtension(libraryPath): void

   Registers extensions to a Core object.

   * **Parameters:**

     - libraryPath: string

       A path to the library with ov::Extension

   * **Defined in:**
     `addon.ts:37 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L37>`__


.. rubric:: compileModel
   :name: compileModel

.. container:: m-4

   .. code-block:: ts

      compileModel(model, deviceName, config?): Promise<CompiledModel>

   Asynchronously creates a compiled model from a source :doc:`Model <Model>` object.
   You can create as many compiled models as needed and use them
   simultaneously (up to the limitation of the hardware resources).

   * **Parameters:**

     - model: :doc:`Model <Model>`

       The :doc:`Model <Model>` object acquired from :ref:`Core.readModel <readModel>`

     - deviceName: string

       The name of a device, to which the model is loaded.

     - ``Optional``

       An object with the key-value pairs
       (property name, property value): relevant only for this load operation.

       .. code-block:: ts

          config: {
              [propertyName: string]: string;
          }

     - [propertyName: string]:string

   * **Returns:** Promise<\ :doc:`CompiledModel <CompiledModel>` \>

   * **Defined in:**
     `addon.ts:48 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L48>`__


   .. code-block:: ts

      compileModel(modelPath, deviceName, config?): Promise<CompiledModel>

   Asynchronously reads a model and creates a compiled model
   from the IR/ONNX/PDPD file. This can be more efficient
   than using :ref:`Core.readModel <readModel>` + :ref:`core.compileModel(Model) <compileModel>`
   flow especially for cases when caching is enabled and a cached model is
   available. You can create as many compiled models as needed and use
   them simultaneously (up to the limitation of the hardware resources).

   * **Parameters:**

     - model: :doc:`Model <Model>`

       The path to a model.is i

     - deviceName: string

       The name of a device, to which a model is loaded.

     - ``Optional``

       .. code-block:: ts

          config: {
              [propertyName: string]: string;
          }

       An object with the key-value pairs
       (property name, property value): relevant only for this load operation.

       - [propertyName: string]:string

   * **Returns:** Promise<\ :doc:`CompiledModel <CompiledModel>` \>

   * **Defined in:**
     `addon.ts:67 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L67>`__


.. rubric:: compileModelSync

.. container:: m-4

   .. code-block:: ts

      compileModelSync(model, deviceName, config?): CompiledModel

   A synchronous version of :ref:`Core.compileModel <compileModel>`.
   It creates a compiled model from a source model object.

   * **Parameters:**

     - model: :doc:`Model <Model>`
     - deviceName: string
     - ``Optional``

       .. code-block:: ts

          config: {
                    [propertyName: string]: string;
           }

     - [propertyName: string]:string

   * **Returns:** :doc:`CompiledModel <CompiledModel>`

   * **Defined in:**
     `addon.ts:76 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L76>`__


   .. code-block:: ts

      compileModelSync(modelPath, deviceName, config?): CompiledModel

   A synchronous version of :ref:`Core.compileModel <compileModel>`.
   It reads a model and creates a compiled model from the IR/ONNX/PDPD file.

   * **Parameters:**

     - modelPath: string
     - deviceName: string
     - ``Optional``

       .. code-block:: ts

          config: {
                    [propertyName: string]: string;
           }

     - [propertyName: string]:string

   * **Returns:** :doc:`CompiledModel <CompiledModel>`

   * **Defined in:**
     `addon.ts:85 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L85>`__


.. rubric:: getAvailableDevices

.. container:: m-4

   .. code-block:: ts

      getAvailableDevices(): string[]

   It returns a list of available inference devices.
   Core objects go over all registered plugins.

   * **Returns:** string[]

     The list of devices may include any of the following: CPU, GPU.0,
     GPU.1, NPU… If there is more than one device of a specific type, they are
     enumerated with ``.#`` suffix. Such enumerated devices can later be used
     as a device name in all Core methods, like ``compile_model``, ``query_model``,
     ``set_property`` and so on.

   * **Defined in:**
     `addon.ts:99 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L99>`__


.. rubric:: getProperty

.. container:: m-4

   .. code-block:: ts

      getProperty(propertyName): string | number | boolean

   It gets the properties dedicated to device behavior.

   * **Parameters:**

     - propertyName: string

       A property name.

   * **Returns:**  string | number | boolean

   * **Defined in:**
     `addon.ts:104 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L104>`__

.. container:: m-4

   .. code-block:: ts

      getProperty(deviceName, propertyName): string | number | boolean

   It gets the properties dedicated to device behavior.

   * **Parameters:**

     - deviceName: string

       The name of a device, the properties of which you get.

     - propertyName: string

       A property name.

   * **Returns:**  string | number | boolean

   * **Defined in:**
     `addon.ts:111 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L111>`__


.. rubric:: getVersions

.. container:: m-4

   .. code-block:: ts

      getVersions(deviceName): {
          [deviceName: string]: {
              buildNumber: string;
              description: string;
          };
      }

   It returns information on the version of device plugins.

   * **Parameters:**

     - deviceName: string

       A device name to identify a plugin.

   * **Returns:**

     .. code-block::

        {
            [deviceName: string]: {
                buildNumber: string;
                description: string;
            };
        }

     * buildNumber: string
     * description: string

   * **Defined in:**
     `addon.ts:119 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L119>`__


.. rubric:: importModelSync

.. container:: m-4

   .. code-block:: ts

      importModelSync(modelStream, device, config?): CompiledModel

   It imports a previously exported compiled model.

   * **Parameters:**

     - modelStream: Buffer

       The input stream that contains a model, previously exported with the
       :ref:`CompiledModel.exportModelSync <exportModelSync>` method.

     - device: string

       The name of a device, for which you import a compiled model. Note, if the device name
       was not used to compile the original model, an exception is thrown.

     - ``Optional``

       .. code-block:: ts

          config: {
                    [key: string]: string | number | boolean;
           }

       An object with the key-value pairs (property name, property value): relevant only for this load operation.

       - [key: string]: string | number | boolean

   * **Returns:** :doc:`CompiledModel <CompiledModel>`

   * **Defined in:**
     `addon.ts:135 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L135>`__


.. rubric:: readModel
   :name: readModel

.. container:: m-4

   .. code-block:: ts

      readModel(modelPath, weightsPath?): Promise<Model>

   It reads models from the IR / ONNX / PDPD / TF and TFLite formats.

   * **Parameters:**

     - modelPath: string

       The path to a model in the IR / ONNX / PDPD / TF or TFLite format.

     - ``Optional``

       .. code-block:: ts

          weightsPath: string

       The path to a data file for the IR format (.bin): if the path is empty, it tries to
       read the bin file with the same name as xml and if the bin file with the same name
       was not found, it loads IR without weights.

       | For the ONNX format (.onnx), the weights parameter is not used.
       | For the PDPD format (.pdmodel), the weights parameter is not used.
       | For the TF format (.pb), the weights parameter is not used.
       | For the TFLite format (.tflite), the weights parameter is not used.

   * **Returns:** Promise<\ :doc:`Model <Model>`\ >

   * **Defined in:**
     `addon.ts:152 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L152>`__

.. container:: m-4

   .. code-block:: ts

      readModel(modelBuffer, weightsBuffer?): Promise<Model>

   It reads models from the IR / ONNX / PDPD / TF and TFLite formats.

   * **Parameters:**

     - modelBuffer: Uint8Array

       Binary data with a model in the IR / ONNX / PDPD / TF or TFLite format.

     - ``Optional``

       .. code-block:: ts

          weightsBuffer: Uint8Array

       Binary data with tensor data.

   * **Returns:**  Promise<\ :doc:`Model <Model>`\ >

   * **Defined in:**
     `addon.ts:160 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L160>`__


.. rubric:: readModelSync

.. container:: m-4

   .. code-block:: ts

      readModelSync(modelPath, weightsPath?): Model

   It reads models from the IR / ONNX / PDPD / TF and TFLite formats.

   * **Parameters:**

     - modelPath: string

       The path to a model in the IR / ONNX / PDPD / TF or TFLite format.

     - ``Optional``

       .. code-block:: ts

          weightsPath: string

   * **Returns:** Promise<\ :doc:`Model <Model>`\ >

   * **Defined in:**
     `addon.ts:166 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L166>`__


   .. code-block:: ts

      readModelSync(modelBuffer, weightsBuffer?): Model

   * **Parameters:**

     - modelBuffer: Uint8Array
     - ``Optional``

       .. code-block:: ts

          weightsBuffer: Uint8Array

   * **Returns:**  :doc:`Model <Model>`

   * **Defined in:**
     `addon.ts:171 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L171>`__


.. rubric:: setProperty

.. container:: m-4

   .. code-block:: ts

      setProperty(properties): void

   It sets the properties.

   * **Parameters:**

     -

       .. code-block:: ts

          properties: {
                   [key: string]: string | number | boolean;
          }

       An object with the property name - property value pairs.

       - [key: string]: string | number | boolean

   * **Returns:**  void

   * **Defined in:**
     `addon.ts:176 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L176>`__


   .. code-block:: ts

      setProperty(deviceName, properties): void

   It sets the properties for a device.

   * **Parameters:**

     - deviceName: string
     -

       .. code-block:: ts

          properties: {
                   [key: string]: string | number | boolean;
          }

       - [key: string]: string | number | boolean

   * **Returns:**  string | number | boolean

   * **Defined in:**
     `addon.ts:182 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L182>`__

