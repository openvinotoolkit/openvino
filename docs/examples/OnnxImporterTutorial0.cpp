#include <inference_engine.hpp>

int main() {
using namespace InferenceEngine;
const std::int64_t version = 12;

const std::string domain = "ai.onnx";

const std::set<std::string> supported_ops = ngraph::onnx_import::get_supported_operators(version, domain);



for(const auto& op : supported_ops)

{

    std::cout << op << std::endl;

}

return 0;
}
