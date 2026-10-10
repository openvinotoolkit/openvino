// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "keep_precision_sensitive_bf16.hpp"

#include <deque>
#include <memory>
#include <unordered_set>
#include <utility>

#include "intel_gpu/runtime/debug_configuration.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/binary_elementwise_arithmetic.hpp"
#include "openvino/op/util/broadcast_base.hpp"
#include "openvino/op/util/unary_elementwise_arithmetic.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/rms.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v8 = ov::op::v8;
namespace v13 = ov::op::v13;
namespace op_util = ov::op::util;

namespace ov::intel_gpu {

namespace bf16_detail {

constexpr size_t max_attention_walk_depth = 16;
constexpr size_t max_attention_walk_nodes = 512;

static bool is_attention_input_glue(const ov::Node* node) {
    return ov::is_type_any_of<v1::Reshape,
                              v1::Transpose,
                              v0::Squeeze,
                              v0::Unsqueeze,
                              v0::Concat,
                              v8::Slice,
                              v1::StridedSlice,
                              v1::VariadicSplit,
                              v1::Split,
                              v0::Convert,
                              v0::FakeQuantize,
                              v0::Tile,
                              op_util::BroadcastBase,
                              op_util::BinaryElementwiseArithmetic,
                              op_util::UnaryElementwiseArithmetic>(node);
}

// True when the value produced by `root` reaches the query (input 0) or the key (input 1) of a
// ScaledDotProductAttention through glue ops only. Everything that mixes the hidden dimension
// (MatMul above all) stops the walk, which is what separates the QK-normalization from the
// normalizations that feed the query/key projections.
static bool feeds_attention_query_or_key(const std::shared_ptr<ov::Node>& root) {
    std::unordered_set<const ov::Node*> visited{root.get()};
    std::deque<std::pair<const ov::Node*, size_t>> queue{{root.get(), 0}};

    while (!queue.empty() && visited.size() <= max_attention_walk_nodes) {
        const auto *const node = queue.front().first;
        const auto depth = queue.front().second;
        queue.pop_front();
        if (depth >= max_attention_walk_depth) {
            continue;
        }
        for (const auto& output : node->outputs()) {
            for (const auto& consumer_input : output.get_target_inputs()) {
                auto* consumer = consumer_input.get_node();
                if (ov::is_type<v13::ScaledDotProductAttention>(consumer)) {
                    if (consumer_input.get_index() < 2) {
                        return true;
                    }
                    continue;
                }
                if (is_attention_input_glue(consumer) && visited.insert(consumer).second) {
                    queue.emplace_back(consumer, depth + 1);
                }
            }
        }
    }
    return false;
}

// Marks the RMS normalization of the attention query/key tensors to be kept in f32. Only the
// fused ov::op::internal::RMS form is recognized (see the class doc comment in the .hpp for why).
class MarkAttentionQKNormToKeepInMixedPrecision : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::intel_gpu::MarkAttentionQKNormToKeepInMixedPrecision");

    MarkAttentionQKNormToKeepInMixedPrecision(const ov::element::Type& target, size_t& marked_count) {
        using namespace ov::pass::pattern;

        // ov::pass::RMSFusion has already folded the normalization into a single op by the time
        // this pass runs, so matching the fused form is enough.
        auto rms = wrap_type<ov::op::internal::RMS>();

        ov::matcher_pass_callback callback = [target, &marked_count](Matcher& m) {
            const auto& root = m.get_match_root();
            if (!feeds_attention_query_or_key(root)) {
                return false;
            }
            ov::disable_conversion(root, target);
            ++marked_count;
            return true;
        };

        register_matcher(std::make_shared<Matcher>(rms, "MarkAttentionQKNormToKeepInMixedPrecision"), callback);
    }
};

// LPT and ov::pass::KeepDequantizationPrecision request their subgraphs to be kept in f32 with a
// hardcoded element::f16 key. Mirror those marks onto the actual compression target so that the
// f32 -> target ConvertPrecision below honours them.
class MirrorKeepMarksToTarget : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::intel_gpu::MirrorKeepMarksToTarget");

    explicit MirrorKeepMarksToTarget(const ov::element::Type& target) : m_target(target) {}

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override {
        bool is_changed = false;
        for (const auto& node : model->get_ops()) {
            if (ov::is_conversion_disabled(node, ov::element::f16) && !ov::is_conversion_disabled(node, m_target)) {
                ov::disable_conversion(node, m_target);
                is_changed = true;
            }
        }
        return is_changed;
    }

private:
    ov::element::Type m_target;
};

}  // namespace bf16_detail

bool KeepPrecisionSensitiveSubgraphsForBF16::run_on_model(const std::shared_ptr<ov::Model>& model) {
    using namespace bf16_detail;

    const auto target = ov::element::bf16;

    // Declared before the manager that captures a reference to it.
    size_t qk_norm_marked = 0;

    ov::pass::Manager manager(get_pass_config(), "KeepPrecisionSensitiveSubgraphsForBF16");
    manager.set_per_pass_validation(false);

    manager.register_pass<MirrorKeepMarksToTarget>(target);
    manager.register_pass<MarkAttentionQKNormToKeepInMixedPrecision>(target, qk_norm_marked);
    // The marks set above are read by the f32 -> bf16 ov::pass::ConvertPrecision registered right
    // after this pass: it runs ov::pass::MarkSugraphsToKeepInMixedPrecision and
    // ov::pass::AlignMixedFP32FP16Types itself (both are parameterized on the compression target,
    // not hardcoded to f16) to propagate the marks and to insert the f32 <-> bf16 boundary
    // Converts. Registering those two passes here as well would run them twice, leaving a
    // duplicate identity Convert at every boundary (harmless numerically -- CommonOptimizations'
    // NopElimination folds it away -- but pointless graph noise).

    const bool is_changed = manager.run_passes(model);

    if (qk_norm_marked == 0) {
        size_t attention_count = 0;
        size_t rms_count = 0;
        for (const auto& node : model->get_ops()) {
            attention_count += ov::is_type<v13::ScaledDotProductAttention>(node) ? 1 : 0;
            rms_count += ov::is_type<ov::op::internal::RMS>(node) ? 1 : 0;
        }
        if (attention_count > 0) {
            GPU_DEBUG_COUT << "[KeepPrecisionSensitiveSubgraphsForBF16] found no attention QK-normalization to keep "
                              "in f32 although the model has "
                           << attention_count << " ScaledDotProductAttention and " << rms_count
                           << " ov::op::internal::RMS node(s) -- RMSFusion may have been skipped for this "
                              "model/device, or the QK-normalization does not use ov::op::internal::RMS. The bf16 "
                              "accuracy regression this pass exists to fix is not being protected against."
                           << std::endl;
        }
    }

    return is_changed;
}

}  // namespace ov::intel_gpu
