#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_sum.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_smooth_l1_loss(const NodeContext& node) {
    // Get inputs: input (predictions) and target (ground truth)
    auto input = node.get_input(0);
    auto target = node.get_input(1);
    
    // Get attributes with defaults matching PyTorch
    auto reduction = node.get_attribute<std::string>("reduction", "mean");
    auto beta = node.get_attribute<float>("beta", 1.0);
    
    // Create constants with input's element type
    auto beta_const = v0::Constant::create(input.get_element_type(), Shape{}, {beta});
    auto half = v0::Constant::create(input.get_element_type(), Shape{}, {0.5f});
    auto beta_reciprocal = v0::Constant::create(input.get_element_type(), Shape{}, {1.0f / beta});
    
    // Calculate |input - target|
    auto diff = std::make_shared<v1::Subtract>(input, target);
    auto abs_diff = std::make_shared<v0::Abs>(diff);
    
    // Calculate smooth L1 loss:
    // L = 0.5 * (x)^2 / beta,     if |x| < beta
    // L = |x| - 0.5 * beta,       if |x| >= beta
    
    // Create mask for |x| < beta
    auto less_mask = std::make_shared<v1::Less>(abs_diff, beta_const);
    
    // Calculate quadratic term: 0.5 * (x)^2 / beta
    auto squared_diff = std::make_shared<v1::Multiply>(diff, diff);
    auto quad_term = std::make_shared<v1::Multiply>(
        std::make_shared<v1::Multiply>(half, squared_diff),
        beta_reciprocal);
    
    // Calculate linear term: |x| - 0.5 * beta
    auto half_beta = std::make_shared<v1::Multiply>(half, beta_const);
    auto linear_term = std::make_shared<v1::Subtract>(abs_diff, half_beta);
    
    // Select between quadratic and linear terms based on mask
    auto loss = std::make_shared<v1::Select>(less_mask, quad_term, linear_term);
    
    // Apply reduction
    if (reduction == "none") {
        return {loss};
    } else if (reduction == "mean") {
        // Create axes for reduction (reduce all dimensions)
        std::vector<int64_t> axes;
        for (size_t i = 0; i < input.get_shape().size(); ++i) {
            axes.push_back(i);
        }
        auto axes_const = v0::Constant::create(element::i64, Shape{axes.size()}, axes);
        return {std::make_shared<v1::ReduceMean>(loss, axes_const, false)};
    } else { // sum
        // Create axes for reduction (reduce all dimensions)
        std::vector<int64_t> axes;
        for (size_t i = 0; i < input.get_shape().size(); ++i) {
            axes.push_back(i);
        }
        auto axes_const = v0::Constant::create(element::i64, Shape{axes.size()}, axes);
        return {std::make_shared<v1::ReduceSum>(loss, axes_const, false)};
    }
}

OutputVector translate_smooth_l1_loss_fx(const NodeContext& node) {
    return translate_smooth_l1_loss(node);
}

} // namespace op
} // namespace pytorch
} // namespace frontend
} // namespace ov 
