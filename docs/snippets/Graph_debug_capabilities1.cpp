#include <inference_engine.hpp>
#include <ngraph/pass/visualize_tree.hpp>


int main() {
using namespace InferenceEngine;
//! [part1]
std::shared_ptr<ngraph::Function> nGraph;
// ...
CNNNetwork network(nGraph);
network.serialize("test_ir.xml", "test_ir.bin");
//! [part1]
return 0;
}
