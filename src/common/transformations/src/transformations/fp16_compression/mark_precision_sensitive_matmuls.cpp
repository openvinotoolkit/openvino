// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/fp16_compression/mark_precision_sensitive_matmuls.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/broadcast_base.hpp"
#include "openvino/op/util/precision_sensitive_attribute.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;
using namespace pass;

namespace {
std::shared_ptr<Node> propagate_through_ops =
    pattern::wrap_type<ov::op::v0::Squeeze,
                       ov::op::v0::Unsqueeze,
                       ov::op::v1::Reshape,
                       ov::op::v1::Transpose,
                       op::util::BroadcastBase,
                       op::util::BinaryElementwiseArithmetic,
                       op::util::UnaryElementwiseArithmetic,
                       ov::op::v1::StridedSlice,
                       ov::op::v8::Slice,
                       ov::op::v1::VariadicSplit,
                       ov::op::v1::Softmax,
                       ov::op::v8::Softmax,
                       ov::op::v1::Split,
                       ov::op::v0::Concat,
                       ov::op::v0::Convert,  // TODO: check interaction of Convert and Const
                       ov::op::v0::Constant>();
}

void mark_softmax_convert_path(const std::shared_ptr<Node>& node) {
    node->get_rt_info().emplace("softmax_convert", true);
}
bool is_softmax_convert_path(const std::shared_ptr<const Node>& node) {
    return node->get_rt_info().count("softmax_convert");
}

void erase_softmax_convert_path(const std::shared_ptr<Node>& node) {
    auto& rt_info = node->get_rt_info();
    rt_info.erase("softmax_convert");
}

// Marks MatMuls which have Convert on their inputs
class InitMarkMatmuls : public MatcherPass {
public:
    OPENVINO_RTTI("InitMarkMatmuls", "0");

    InitMarkMatmuls() {
        MATCHER_SCOPE(InitMarkMatmuls);
        auto input_1 = pattern::any_input();
        auto input_2 = pattern::any_input();
        auto convert_pattern = pattern::wrap_type<op::v0::Convert>();
        auto matmul_pattern_1 = pattern::wrap_type<op::v0::MatMul>({input_1, convert_pattern});
        auto matmul_pattern_2 = pattern::wrap_type<op::v0::MatMul>({convert_pattern, input_1});
        auto matmul_pattern = std::make_shared<pattern::op::Or>(OutputVector{matmul_pattern_1, matmul_pattern_2});

        matcher_pass_callback callback = [=](pattern::Matcher& m) {
            const auto& node = m.get_match_root();
            if (!node)
                return false;

            for (const auto& output : node->outputs()) {
                for (const auto& out_inputs : output.get_target_inputs()) {
                    if (is_softmax_convert_path(out_inputs.get_node()->shared_from_this())) {
                        disable_fp16_compression(node);
                        return true;
                    }
                }
            }

            return false;
        };

        auto m = make_shared<pattern::Matcher>(matmul_pattern, matcher_name);
        register_matcher(m, callback);
    }
};

// propagate down MatMuls which have Convert on their inputs
class PropagateDownMarkMatmuls : public MatcherPass {
public:
    OPENVINO_RTTI("PropagateDownMarkMatmuls", "0");
    PropagateDownMarkMatmuls() {
        MATCHER_SCOPE(PropagateDownMarkMatmuls);

        matcher_pass_callback callback = [=](pattern::Matcher& m) {
            const auto& node = m.get_match_root();
            if (!node)
                return false;
            if (!is_softmax_convert_path(node))
                return false;

            bool is_changed = false;
            for (const auto& in_node : node->input_values()) {
                if (!in_node.get_element_type().is_real())
                    continue;

                if (fp16_compression_is_disabled(in_node.get_node_shared_ptr())) {
                    disable_fp16_compression(node);
                    is_changed = true;
                    break;
                }
            }
            return is_changed;
        };
        auto m = make_shared<pattern::Matcher>(propagate_through_ops, matcher_name);
        register_matcher(m, callback);
    }
};

class InitMarkSoftMax : public MatcherPass {
public:
    OPENVINO_RTTI("InitMarkSoftMax", "0");

    InitMarkSoftMax() {
        MATCHER_SCOPE(InitMarkSoftMax);

        auto input_1 = pattern::any_input();
        auto softmax_v1 = pattern::wrap_type<op::v1::Softmax>();
        auto softmax_v8 = pattern::wrap_type<op::v8::Softmax>();
        auto softmax_all_ver = std::make_shared<pattern::op::Or>(OutputVector{softmax_v1, softmax_v8});
        auto convert_pattern = pattern::wrap_type<op::v0::Convert>({softmax_all_ver});

        matcher_pass_callback callback = [=](pattern::Matcher& m) {
            const auto& node = m.get_match_root();
            if (!node)
                return false;

            auto softmax_v1 = as_type_ptr<ov::op::v1::Softmax>(node->input_value(0).get_node_shared_ptr());
            auto softmax_v8 = as_type_ptr<ov::op::v8::Softmax>(node->input_value(0).get_node_shared_ptr());
            if (softmax_v1 || softmax_v8) {
                mark_softmax_convert_path(node->input_value(0).get_node_shared_ptr());
                printf("MATCHED\n");
                std::cout << node->get_friendly_name() << std::endl;
                return true;
            }

            return false;
        };

        auto m = make_shared<pattern::Matcher>(convert_pattern, matcher_name);
        register_matcher(m, callback);
    }
};

class PropagateUpMarkSoftMaxConvert : public pass::MatcherPass {
public:
    OPENVINO_RTTI("PropagateUpMarkSoftMaxConvert", "0");
    PropagateUpMarkSoftMaxConvert() {
        MATCHER_SCOPE(PropagateUpMarkSoftMaxConvert);

        matcher_pass_callback callback = [=](pattern::Matcher& m) {
            // TODO: check if Converts are necessary in the list fof propagate_through_ops pattern
            const auto& node = m.get_match_root();
            bool has_marked_output = false;
            for (const auto& output : node->outputs()) {
                for (const auto& out_inputs : output.get_target_inputs()) {
                    if (out_inputs.get_element_type().is_real() &&
                            is_softmax_convert_path(out_inputs.get_node()->shared_from_this())) {
                        has_marked_output = true;
                    }
                }
            }

            if (!has_marked_output)
                return false;

            mark_softmax_convert_path(node);
            return true;
        };

        auto m = make_shared<pattern::Matcher>(propagate_through_ops, matcher_name);
        register_matcher(m, callback);
    }
};

bool ov::pass::MarkPrecisionSensitiveMatmuls::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(MarkPrecisionSensitiveMatmuls);

    Manager manager(get_pass_config());
    REGISTER_PASS(manager, InitMarkSoftMax)

    auto propagate_up = manager.register_pass<BackwardGraphRewrite>();
    ADD_MATCHER(propagate_up, PropagateUpMarkSoftMaxConvert)

    REGISTER_PASS(manager, InitMarkMatmuls)
    REGISTER_PASS(manager, PropagateDownMarkMatmuls)
    manager.run_passes(model);

    return true;
}
