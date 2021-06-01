#include <ie_core.hpp>
#include <ngraph/ngraph.hpp>
#include "onnx_import/onnx.hpp"
#include <iostream>

int main() {
//! [part3]
const char * resnet50_path = "resnet50/model.onnx";
const std::shared_ptr<ngraph::Function> ng_function = ngraph::onnx_import::import_onnx_model(resnet50_path);
//! [part3]
return 0;
}
