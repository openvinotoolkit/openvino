// Support dynamic shapes and ranks, undefined types, including future support of new types, such as strings and complex numbers.
// Try to maintain the same algorithmic complexity of the decomposition. Fewer operations are usually better.
// Use the latest OpenVINO opset version for the translation.
// Use helper routines for operation checks and graph construction from utils.hpp.
// Call NodeContext::mark_mode() for each created node.

////////////////////////////////////////////////////////
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/reverse.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "utils.hpp"

namespace ov{
namespace frontend{
namespace pytorch{
namespace op{
using namespace ov::op;
OutputVector translate_rot90(const NodeContext& context){
    num_inputs_check(context,1,3);
    auto tensor = context.get_input(0);
    auto k = context.input_is_none(1) ? 1 : context.const_input<int64_t>(1);
    const auto dims = context.input_is_none(2) ? context.mark_node(v0::Constant::create(element::i64, {2}, 
            {0, 1})) : context.get_input(2);
    k %= 4;
    auto one = v0::Constant::create(element::i64,{},{1});
    auto zero = v0::Constant::create(element::i64,{},{0});
        
    auto axis0 = v0::Constant::create(element::i64,{},{0});

    auto dim0 = context.mark_node(std::make_shared<v8::Gather>(dims,zero,axis0));
    auto dim1 = context.mark_node(std::make_shared<v8::Gather>(dims,one,axis0));


    auto dim0_s = context.mark_node(std::make_shared<v0::Squeeze>(dim0));
    auto dim1_s = context.mark_node(std::make_shared<v0::Squeeze>(dim1));

        if(k==1){
            auto transpose = context.mark_node(std::make_shared<v1::Transpose>(tensor,dims));
            auto flipped = context.mark_node(std::make_shared<v1::Reverse>(transpose,dim1_s,v1::Reverse::Mode::INDEX));
            return {flipped};
        }
        if(k==2){
            auto flipped = context.mark_node(std::make_shared<v1::Reverse>(tensor,dims,v1::Reverse::Mode::INDEX
            ));
            return {flipped};
        }
        if(k==3){
            auto flipped = context.mark_node(std::make_shared<v1::Reverse>(tensor,dim0_s,v1::Reverse::Mode::INDEX));
            auto transpose = context.mark_node(std::make_shared<v1::Transpose>(flipped,dims));
            return {transpose};
        }
    return {tensor};
};
} // namespace op;
} // namespace pytorch
} // namespace frontend
} // namespace ov