#include <ie_core.hpp>
#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset3.hpp"


int main() {
//! [part0]

using namespace std;
using namespace ngraph;

auto arg0 = make_shared<opset3::Parameter>(element::f32, Shape{7});
auto arg1 = make_shared<opset3::Parameter>(element::f32, Shape{7});
// Create an 'Add' operation with two inputs 'arg0' and 'arg1'
auto add0 = make_shared<opset3::Add>(arg0, arg1);
auto abs0 = make_shared<opset3::Abs>(add0);
// Create a node whose inputs/attributes will be specified later
auto acos0 = make_shared<opset3::Acos>();
// Create a node using opset factories
auto add1 = shared_ptr<Node>(get_opset3().create("Add"));
// Set inputs to nodes explicitly
acos0->set_argument(0, add0);
add1->set_argument(0, acos0);
add1->set_argument(1, abs0);

// Run shape inference on the nodes
NodeVector ops{arg0, arg1, add0, abs0, acos0, add1};
validate_nodes_and_infer_types(ops);

// Create a graph with one output (add1) and four inputs (arg0, arg1)
auto ng_function = make_shared<Function>(OutputVector{add1}, ParameterVector{arg0, arg1});

//! [part0]

//! [part1]
InferenceEngine::CNNNetwork net (ng_function);
//! [part1]

return 0;
}
