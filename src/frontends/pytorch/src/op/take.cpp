#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/shape_of.hpp"
#include "utils.hpp"



namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;
using namespace std;



OutputVector translate_take_op(const NodeContext& context){

    num_inputs_check(context, 2, 2, true);
    //Get input tensor and tensor indices

    Output<Node> input = context.get_input(0);
    Output<Node> indices = context.get_input(1);
    //get their inputs

    //We get information about the input tensor
    auto input_shape = input.get_partial_shape();

    if (input_shape.rank().is_static() && input_shape.rank().get_length() == 0) {
        FRONT_END_OP_CONVERSION_CHECK(false, "input tensor MUST be non-scalar");
    }
    //always flatten the tensor to 1D by using -1 
    auto new_shape = context.mark_node(
        v0::Constant::create(element::i64, Shape{1}, {-1})
    );
    
    input = context.mark_node(
        std::make_shared<v1::Reshape>(input, new_shape, false)
    );

    //the openVINO needs the indices always in i64 
    indices = context.mark_node(
        std::make_shared<v0::Convert>(indices, element::i64)
    );

    //handle negative indices
    auto input_size = context.mark_node(
        std::make_shared<v3::ShapeOf>(input, element::i64)
    );
    indices = normalize_axis(context, indices, input_size);
    //create a axis_constant = 0
    auto axis_constant = context.mark_node(
        v0::Constant::create(element::i64, Shape{}, {0})
    );

    //now apply the gather function
    auto gather = context.mark_node(
        std::make_shared<v8::Gather>(input, indices, axis_constant)
    );
    

    return {gather};
    
}   




}}}}