#include <ie_core.hpp>
#include <ngraph/function.hpp>
#include <ngraph/variant.hpp>

int main() {
InferenceEngine::Core core;
auto network = core.ReadNetwork("sample.xml");
auto function = network.getFunction();
//! [part0]
for (auto && op : function->get_ops())
    op->get_rt_info()["affinity"] = std::make_shared<ngraph::VariantWrapper<std::string>>("CPU");
//! [part0]
return 0;
}
