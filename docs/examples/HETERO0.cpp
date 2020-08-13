#include <inference_engine.hpp>
#include "ie_plugin_config.hpp"
#include "hetero/hetero_plugin_config.hpp"

int main() {
using namespace InferenceEngine;
for (auto && op : function->get_ops())
    op->get_rt_info()["affinity"] = std::shared_ptr<ngraph::VariantWrapper<std::string>>("CPU");
return 0;
}
