# OpenVINO Paddle Frontend operator enabling Flow

1. Declare `CreatorFunction` for the Paddle operator and register it to the map in `src/op_table.cpp`.
   * The map is retrived from:
     ```cpp
     std::map<std::string, CreatorFunction> get_supported_ops() {
         return {{"arg_max", op::argmax},
                 ......
     ```
2. Implement the operator mapper `CreatorFunction` in path `src/op/`.
3. Add unit-tests * [OpenVINO™ Paddle Frontend unit test README](tests.md)

## See also
* [OpenVINO™ Paddle Frontend README](../README.md)
* [OpenVINO™ Frontend README](TODO)
