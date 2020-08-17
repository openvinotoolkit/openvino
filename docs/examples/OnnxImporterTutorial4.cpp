#include <inference_engine.hpp>

int main() {
using namespace InferenceEngine;
const std::shared_ptr<ngraph::Function> ng_function = ngraph::onnx_import::import_onnx_model(resnet50_path);

return 0;
}
