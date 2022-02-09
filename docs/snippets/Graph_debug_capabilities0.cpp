#include <ie_core.hpp>
#include <openvino/core/model.hpp>
#include <openvino/pass/visualize_tree.hpp>

int main() {
using namespace InferenceEngine;
//! [part0]
std::shared_ptr<ov::Model> model;
// ...
ov::pass::VisualizeTree("after.png").run_on_model(model);     // Visualize the nGraph function to an image
//! [part0]
return 0;
}
