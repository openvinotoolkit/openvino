#include <openvino/opsets/opset7.hpp>
#include <openvino/opsets/opset8.hpp>

namespace ov {
namespace frontend {
namespace pytorch {

Output<Node> make_optional_bias(Output<Node> base_op,
                                const NodeContext& context,
                                size_t bias_input_idx,
                                std::vector<int> unsqueeze_dims = {}) {
    using namespace ngraph;
    using std::make_shared;

    if (!context.input_is_none(bias_input_idx)) {
        auto bias = context.get_input(bias_input_idx);
        if (!unsqueeze_dims.empty()) {
            auto indices = opset7::Constant::create(element::i32, {unsqueeze_dims.size()}, unsqueeze_dims);
            context.mark_node(indices);
            bias = make_shared<opset7::Unsqueeze>(bias, indices);
            context.mark_output(bias);
        }
        return make_shared<opset7::Add>(context.mark_output(base_op), bias);
    } else {
        return base_op;
    }
}

std::shared_ptr<ov::Node> get_rank_node(ov::Output<ov::Node> node) {
    auto shape = std::make_shared<opset8::ShapeOf>(node);
    return std::make_shared<opset8::ShapeOf>(shape);
}

Output<Node> reshape_kernel_for_group(const NodeContext& context,
                                      Output<Node> input,
                                      Output<Node> kernel,
                                      int64_t groups) {
    using namespace ngraph;
    using std::make_shared;

    auto in_shape = std::make_shared<opset8::ShapeOf>(input);
    auto c_in_idx = opset8::Constant::create(element::i64, Shape{}, {1});
    auto axis_0 = opset8::Constant::create(element::i64, Shape{}, {0});
    auto in_shape_1 = make_shared<opset8::Gather>(in_shape, c_in_idx, axis_0);
    auto in_shape_1_uns = make_shared<opset8::Unsqueeze>(in_shape_1, axis_0);
    auto groups_const = opset8::Constant::create(element::i64, Shape{1}, {groups});
    auto c_in_value = make_shared<opset8::Divide>(in_shape_1_uns, groups_const);

    auto kernel_shape = std::make_shared<opset8::ShapeOf>(kernel);
    auto c_out_idx = opset8::Constant::create(element::i64, Shape{}, {0});
    auto kernel_shape_0 = make_shared<opset8::Gather>(kernel_shape, c_out_idx, axis_0);
    auto kernel_shape_0_uns = make_shared<opset8::Unsqueeze>(kernel_shape_0, axis_0);
    auto c_out_value = make_shared<opset8::Divide>(kernel_shape_0_uns, groups_const);

    auto start = opset8::Constant::create(element::i64, Shape{1}, {2});
    auto stop = opset8::Constant::create(element::i64, Shape{1}, {std::numeric_limits<int64_t>::max()});
    auto step = opset8::Constant::create(element::i64, Shape{1}, {1});
    auto remaining_shape = make_shared<opset8::Slice>(kernel_shape, start, stop, step);

    auto new_kernel_shape =
        make_shared<opset8::Concat>(OutputVector{groups_const, c_out_value, c_in_value, remaining_shape}, 0);
    context.mark_nodes({in_shape,
                        c_in_idx,
                        axis_0,
                        in_shape_1,
                        in_shape_1_uns,
                        groups_const,
                        c_in_value,
                        kernel_shape,
                        c_out_idx,
                        kernel_shape_0,
                        kernel_shape_0_uns,
                        c_out_value,
                        start,
                        stop,
                        step,
                        remaining_shape,
                        new_kernel_shape});
    return make_shared<opset8::Reshape>(kernel, new_kernel_shape, false);
}

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
