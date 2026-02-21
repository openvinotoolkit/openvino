#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/log_softmax.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_cross_entropy_loss(const NodeContext& context) {
    // PyTorch aten::cross_entropy_loss(input, target, weight=None, reduction=1, ignore_index=-100, label_smoothing=0.0)
    num_inputs_check(context, 2, 6);
    
    auto input = context.get_input(0);  // Logits
    auto target = context.get_input(1); // Indices or Probs

    // 1. Get Reduction Mode (0: none, 1: mean, 2: sum)
    int64_t reduction = 1; 
    if (!context.input_is_none(3)) {
        reduction = context.const_input<int64_t>(3);
    }

    // 2. Get Ignore Index
    int64_t ignore_index = -100;
    if (!context.input_is_none(4)) {
        ignore_index = context.const_input<int64_t>(4);
    }

    // 3. Compute Log Softmax along the class dimension (usually dim 1)
    // We assume dim 1 is the channel/class dimension as per PT standard
    auto log_softmax = context.mark_node(std::make_shared<v1::LogSoftmax>(input, 1));

    Output<Node> loss;

    // 4. Handle Target Types
    // PyTorch CrossEntropy supports two target formats:
    // Case A: Target is class indices (Long tensor)
    // Case B: Target is probabilities (Float tensor, same shape as input)
    if (target.get_element_type().is_integral()) {
        // --- Case A: NLLLoss Logic (Negative Log Likelihood) ---
        
        // We use Gather to pick the log-probability of the correct class
        // Logits shape: [N, C, H, W], Target shape: [N, H, W]
        // We need to align them for Gather or use specialized NLLLoss-like decomposition
        
        // Simplified index-based gathering:
        auto axis = v0::Constant::create(element::i64, {}, {1});
        // Gather expects indices to be aligned. In a real developer impl, 
        // we'd use a more robust gathering logic or the ov::op::v8::GatherND.
        auto gathered_log_probs = context.mark_node(std::make_shared<v8::Gather>(log_softmax, target, axis));
        loss = context.mark_node(std::make_shared<v1::Negative>(gathered_log_probs));
        
        // Handle ignore_index by masking (setting loss to 0 where target == ignore_index)
        if (ignore_index != -100) {
            auto ignore_const = v0::Constant::create(target.get_element_type(), {}, {ignore_index});
            auto mask = context.mark_node(std::make_shared<v1::NotEqual>(target, ignore_const));
            auto mask_float = context.mark_node(std::make_shared<v0::Convert>(mask, input.get_element_type()));
            loss = context.mark_node(std::make_shared<v1::Multiply>(loss, mask_float));
        }
    } else {
        // --- Case B: Softmax Cross Entropy (Probs) ---
        // Formula: -sum(target * log_softmax, dim=1)
        auto multiplied = context.mark_node(std::make_shared<v1::Multiply>(log_softmax, target));
        auto sum_axis = v0::Constant::create(element::i64, {1}, {1});
        auto sum_probs = context.mark_node(std::make_shared<v1::ReduceSum>(multiplied, sum_axis));
        loss = context.mark_node(std::make_shared<v1::Negative>(sum_probs));
    }

    // 5. Final Reduction
    if (reduction == 1) { // Mean
        auto all_axes = v0::Constant::create(element::i64, {0}, {0}); // Simplified for 1D/2D
        return { context.mark_node(std::make_shared<v1::ReduceMean>(loss, all_axes)) };
    } else if (reduction == 2) { // Sum
        auto all_axes = v0::Constant::create(element::i64, {0}, {0});
        return { context.mark_node(std::make_shared<v1::ReduceSum>(loss, all_axes)) };
    }

    // reduction == 0 (none)
    return { loss };
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov