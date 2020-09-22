#include <inference_engine.hpp>

int main() {
using namespace InferenceEngine;
//! [part0]
InferenceEngine::Core core;
// Load CPU extension as a shared library
auto extension_ptr = make_so_pointer<InferenceEngine::IExtension>("<shared lib path>");
// Add extension to the CPU device
core.AddExtension(extension_ptr, "CPU");
//! [part0]

return 0;
}
