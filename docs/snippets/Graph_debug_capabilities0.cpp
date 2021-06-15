#include <ie_core.hpp>
#include <ngraph/function.hpp>
#include <ngraph/pass/visualize_tree.hpp>

int main() {
using namespace InferenceEngine;
//! [part0]
std::shared_ptr<ngraph::Function> nGraph;
// ...
ngraph::pass::VisualizeTree("after.png").run_on_function(nGraph);     // Visualize the nGraph function to an image
//! [part0]
return 0;
}
