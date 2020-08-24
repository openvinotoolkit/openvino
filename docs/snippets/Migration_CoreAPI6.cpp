#include <inference_engine.hpp>

int main() {
using namespace InferenceEngine;
//! [part6]
core.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>(), "CPU");
//! [part6]
return 0;
}
