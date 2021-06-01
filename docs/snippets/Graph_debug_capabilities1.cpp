#include <ie_core.hpp>
#include <ngraph/function.hpp>

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
