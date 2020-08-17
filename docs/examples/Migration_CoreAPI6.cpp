#include <inference_engine.hpp>

int main() {
using namespace InferenceEngine;
core.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>(), "CPU");

return 0;
}
