# OpenVINO Paddle Frontend operator enabling Flow

1. Declare `CreatorFunction` for the Paddle operator and register it to the map in `src/op_table.cpp`.
   * The map is retrived from:

     https://github.com/openvinotoolkit/openvino/blob/7d5e0abcaa03703de9918ece2115e6ea652c39e0/src/frontends/paddle/src/op_table.cpp#L106

2. Implement the operator mapper `CreatorFunction` in path `src/op/`.
3. Add unit-tests [OpenVINO™ Paddle Frontend unit test README](tests.md)

## See also
* [OpenVINO™ Paddle Frontend README](../README.md)
* [OpenVINO™ Frontend README](../../README.md)
