#include <openvino/runtime/core.hpp>
#include <openvino/core/model.hpp>


int main() {
ov::Core core;
auto model = core.read_model("sample.xml");
//! [part0]
for (auto && op : model->get_ops())
    op->get_rt_info()["affinity"] = "CPU";
//! [part0]
return 0;
}
