# The PyTorch frontend is python tooling + a core-side reader:
#   tools/torch_export.py        — torch.jit.trace -> vktorch + safetensors
#   src/core/pytorch_reader.hpp  — consumed by core tests/executables
# Nothing to compile here; see README.md.
