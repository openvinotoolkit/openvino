#include <inference_engine.hpp>
#include <ngraph/ngraph.hpp>
#include "onnx/onnx-ml.pb.h"
#include <iostream>
#include <set>

int main() {
using namespace InferenceEngine;
//! [part0]
const std::int64_t version = 12;
const std::string domain = "ai.onnx";
const std::set<std::string> supported_ops = ngraph::onnx_import::get_supported_operators(version, domain);

for(const auto& op : supported_ops)
{
    std::cout << op << std::endl;
}
//! [part0]
return 0;
}
