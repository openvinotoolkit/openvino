#include "openvino/frontend/pytorch/node_context.hpp"
#include "utils.hpp"
#include "utils.hpp" 

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

    using namespace ov::op;
OutputVector translate_is_nested(const NodeContext& context){
    num_inputs_check(context, 1, 1);
    auto input = context.get_input(0);
    auto input_shape = context.mark_node(std::make_shared<v3::ShapeOf>(input, element::i32));
    if (auto shape_const = std::dynamic_pointer_cast<v0::Constant>(input_shape.get_node_shared_ptr())) {
        auto shape_vals = shape_const->cast_vector<int32_t>();
        if(shape_vals.size() <= 1){
            return {v0::Constant::create(element::boolean, Shape{}, {false})};

        }
        int32_t first_dim = shape_vals[0];
        for(auto dim : shape_vals){
            if(dim == 0){
                return {v0::Constant::create(element::boolean, Shape{}, {false})};
            }
            if(dim != first_dim){
                return {v0::Constant::create(element::boolean, Shape{}, {true})};
            }
        }
        return {v0::Constant::create(element::boolean, Shape{}, {false})};
    }


    return {v0::Constant::create(element::boolean, Shape{}, {false})};
}

} // namespace op
} // namespace pytorch
} // namespace frontend
} // namespace ov