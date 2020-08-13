#include <inference_engine.hpp>

int main() {
using namespace InferenceEngine;
plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());
return 0;
}
