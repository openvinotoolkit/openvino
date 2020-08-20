#include <inference_engine.hpp>

int main() {
using namespace InferenceEngine;
//! [part0]
for (auto && op : function->get_ops())
    op->get_rt_info()["affinity"] = std::shared_ptr<ngraph::VariantWrapper<std::string>>("CPU");
//! [part0]
return 0;
}
