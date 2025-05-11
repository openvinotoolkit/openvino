# OpenVINO Paddle Frontend Operator Enabling Flow

1. Declare `CreatorFunction` for the Paddle operator and register it to the map in `src/op_table.cpp`.
   * The map is retrieved from:

     https://github.com/openvinotoolkit/openvino/blob/7d5e0abcaa03703de9918ece2115e6ea652c39e0/src/frontends/paddle/src/op_table.cpp#L106

2. Implement the operator mapper `CreatorFunction` in the `src/op/` path.
3. Add unit-tests. For more information, refer to the [OpenVINO™ Paddle Frontend unit-tests readme](tests.md)

## See also
* [OpenVINO™ Paddle Frontend README](../README.md)
* [OpenVINO™ Frontend README](../../README.md)
