#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/convert.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_delete(const NodeContext& context) {
    // check the input
    num_inputs_check(context, 2, 2); 
    // Retrieve inputs
    auto input = context.get_input(0);   // container/target tensor
    auto indices = context.get_input(1); // Indices for elements to delete
    // ensure int32
    if (indices.get_element_type() != ov::element::i32) {
        indices = context.mark_node(std::make_shared<v0::Convert>(indices, ov::element::i32));
    }
    // getting the shape 
    auto input_shape = context.mark_node(std::make_shared<v3::ShapeOf>(input, ov::element::i32));

    //implementation plan ->
    //slice the tensor before the indices 
    //slice the tensor after the indices
    //concatenate the slices

    // calculate the end index for slicing after the target indices
    auto end_index = context.mark_node(std::make_shared<v1::Add>(
        indices,
        v0::Constant::create(ov::element::i32, Shape{1}, {1})
    ));

    // slice elements before the indices
    auto before_indices = context.mark_node(std::make_shared<v1::StridedSlice>(
        input,
        v0::Constant::create(ov::element::i32, Shape{1}, {0}), indices, // Begin at 0, End at indices
        v0::Constant::create(ov::element::i32, Shape{1}, {1}), // Stride of 1
        std::vector<int64_t>{1}, std::vector<int64_t>{0}  
    ));
    // slice elements after the indices
    auto after_indices = context.mark_node(std::make_shared<v1::StridedSlice>(
        input,end_index, // Begin after target, End
        input_shape,
        v0::Constant::create(ov::element::i32, Shape{1}, {1}), // Stride of 1
        std::vector<int64_t>{0}, std::vector<int64_t>{1}  
    ));
    // concat or join the slices
    auto result = context.mark_node(std::make_shared<v0::Concat>(
        OutputVector{before_indices, after_indices}, 0 // axis along which to concatenate
    ));

    return {result};
}

} // namespace op
} // namespace pytorch
} // namespace frontend
} // namespace ov
