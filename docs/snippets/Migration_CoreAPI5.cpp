#include <inference_engine.hpp>

int main() {
using namespace InferenceEngine;
//! [part5]
plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());
//! [part5]
return 0;
}
