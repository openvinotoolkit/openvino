# Plugin Review

- Check that tensor layouts, backend operation semantics, and memory reuse
  remain consistent across primitive selection and execution.
- Check threading and synchronization against the backend's execution model.
- Check that device-specific tests run on hardware supporting the changed
  precision, instruction set, or kernel.
- In new C++ operator code, prefer the `ov::op::vX::OpName` namespace style
  used by current runtime APIs.
