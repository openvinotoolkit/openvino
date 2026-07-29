# OpenVINO Operation Enabling Flow

1. Implement operation "shell" in the `src/core/[src|include]/op/`:
   * Implement constructor(s)
   * Implement `validate_and_infer_types` method which should support dynamic input tensor(s) (with partially dynamic shapes). For more information read [OpenVINO Shape propagation guide](./shape_propagation.md)
   * Implement `visit_attributes` method
   * Implement `clone_with_new_inputs` method. The generated operation version must be explicitly specified and be equal to the operation version being added
   * In `*.hpp` file add:
     ```cpp
     OPENVINO_OP("<Operation_name>", "opset_name", <Parent_op> /* Not needed if operation is inherited from ov::Op */);
     ```
   * To support conditional compilation add following for each Operation method in `*.cpp` file:
      ```cpp
      OV_OP_SCOPE(<operation_version>_<operation_name>_<method_name>);
      ```
   * Add shape infer unit-tests to the `src/core/tests/type_prop/`

2. Add operation to the dedicated opset file `src/core/include/openvino/opsets/opsetX_tbl.hpp`

3. Implement `evaluate` method for the operation (reference implementation) in the `openvino/core/[src|include/openvino]/op/`. Reference implementation can be called from Template plugin or from OpenVINO.
To not increase the binary size of openvino lib it should be placed in Template plugin unless you are directly asked to put it in the OpenVINO core. While adding reference implementation the following points should be considered:
   * The method should avoid using the template parameters if possible. However, for the small operations like activation functions it is acceptable.
   * The method should be instantiated for practical data types only.
   * If the method can be implemented without defining strict data types (for example, data movement operations like Concat or Split) then it should be implemented in a type-agnostic way. 

## See also
 * [OpenVINO™ Core README](../README.md)
 * [OpenVINO™ README](../../../README.md)
 * [Developer documentation](../../../docs/dev/index.md)
