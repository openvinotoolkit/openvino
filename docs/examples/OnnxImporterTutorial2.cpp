#include <inference_engine.hpp>

int main() {
using namespace InferenceEngine;
const std::string op_name = "Abs";

const std::int64_t version = 12;

const std::string domain = "ai.onnx";

const bool is_abs_op_supported = ngraph::onnx_import::is_operator_supported(op_name, version, domain);



std::cout << "Abs in version 12, domain `ai.onnx`is supported: " << (is_abs_op_supported ? "true" : "false") << std::endl;

return 0;
}
