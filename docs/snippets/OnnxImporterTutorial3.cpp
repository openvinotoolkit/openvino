#include <inference_engine.hpp>
#include <ngraph/ngraph.hpp>
#include "ngraph/frontend/onnx_import/onnx.hpp"
#include <iostream>

int main() {
using namespace InferenceEngine;
using namespace ngraph;
//! [part3]
const std::shared_ptr<ngraph::Function> ng_function = ngraph::onnx_import::import_onnx_model(resnet50_path);
//! [part3]
return 0;
}
