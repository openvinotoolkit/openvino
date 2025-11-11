#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/frontend/manager.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/lstm_cell.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

// bracha leah
// Translation for aten::lstm_cell
// Supports both variants:
//   aten::lstm_cell(input, h0, c0, w_ih, w_hh, b_ih, b_hh)
//   aten::lstm_cell(input, h0, c0, w_ih, w_hh)
OutputVector translate_lstm_cell(const NodeContext& context) {
    const size_t num_inputs = context.get_input_size();
    PYTORCH_OP_CONVERSION_CHECK(num_inputs == 5 || num_inputs == 7,
        "aten::lstm_cell expects 5 or 7 inputs.");

    ov::pass::NodeRegistry rg;

    // Mandatory inputs
    const auto x = context.get_input(0);   // input at current timestep
    const auto h0 = context.get_input(1);  // previous hidden state
    const auto c0 = context.get_input(2);  // previous cell state
    const auto w_ih = context.get_input(3);
    const auto w_hh = context.get_input(4);

    // Determine hidden size from weight_hh shape
    const auto& w_hh_pshape = w_hh.get_partial_shape();
    PYTORCH_OP_CONVERSION_CHECK(
        w_hh_pshape.rank().is_static() && w_hh_pshape[1].is_static(),
        "LSTMCell: cannot determine hidden size from weight_hh shape.");
    const auto hidden_size = w_hh_pshape[1].get_length();

    Output<Node> bias;
    if (num_inputs == 7) {
        // b_ih + b_hh
        const auto b_ih = context.get_input(5);
        const auto b_hh = context.get_input(6);
        const auto bias_sum = rg.make<v1::Add>(b_ih, b_hh);
        bias = rg.make<v1::ConvertLike>(bias_sum, x);
    } else {
        // no bias provided — create zeros
        const auto bias_shape = Shape{4 * static_cast<size_t>(hidden_size)};
        bias = v0::Constant::create(element::f32, bias_shape, {0});
        bias = rg.make<v1::ConvertLike>(bias, x);
    }

    // Unsqueeze weights to fit [num_directions, ...] format
    const auto axis_const = v0::Constant::create(element::i32, Shape{}, {0});
    const auto w_ih_unsq = rg.make<v0::Unsqueeze>(w_ih, axis_const);
    const auto w_hh_unsq = rg.make<v0::Unsqueeze>(w_hh, axis_const);
    const auto bias_unsq = rg.make<v0::Unsqueeze>(bias, axis_const);

    // Create the LSTMCell node
    const auto lstm_cell = rg.make<v0::LSTMCell>(
        x, h0, c0, w_ih_unsq, w_hh_unsq, bias_unsq, static_cast<int64_t>(hidden_size));

    // Outputs: (h_t, c_t)
    context.mark_nodes(rg.get());
    return {lstm_cell->output(0), lstm_cell->output(1)};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
