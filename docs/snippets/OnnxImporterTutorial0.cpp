#include <inference_engine.hpp>
#include <iostream>
#include <ngraph/ngraph.hpp>
#include <set>

#include "onnx_import/onnx.hpp"

int main() {
    //! [part0]
    const std::int64_t version = 12;
    const std::string domain = "ai.onnx";
    const std::set<std::string> supported_ops = ngraph::onnx_import::get_supported_operators(version, domain);

    for (const auto& op : supported_ops) {
        std::cout << op << std::endl;
    }
    //! [part0]
    return 0;
}
