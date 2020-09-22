#include <inference_engine.hpp>
#include <ngraph/ngraph.hpp>
#include "hetero/hetero_plugin_config.hpp"

int main() {
using namespace InferenceEngine;
using namespace ngraph;
Core core;
auto network = core.ReadNetwork("sample.xml");
auto function = network.getFunction();
//! [part0]
for (auto && op : function->get_ops())
    op->get_rt_info()["affinity"] = std::shared_ptr<ngraph::VariantWrapper<std::string>>("CPU");
//! [part0]
return 0;
}
