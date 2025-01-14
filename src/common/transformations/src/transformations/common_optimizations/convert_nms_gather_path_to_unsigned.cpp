// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "transformations/common_optimizations/convert_nms_gather_path_to_unsigned.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/if.hpp"
#include "openvino/op/non_max_suppression.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/broadcast_base.hpp"
#include "openvino/op/util/gather_base.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/nms_selected_indices.hpp"

using namespace std;

namespace ov {
namespace pass {
class InitNMSPath : public pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("InitNMSPath");
    InitNMSPath() {
        MATCHER_SCOPE(InitNMSPath);
        auto nms_pattern = pattern::wrap_type<ov::op::v1::NonMaxSuppression,
                                              ov::op::v3::NonMaxSuppression,
                                              ov::op::v5::NonMaxSuppression,
                                              ov::op::v9::NonMaxSuppression>();
        matcher_pass_callback callback = [=](pattern::Matcher& m) {
            const auto& out_nodes = m.get_match_root()->output(0).get_target_inputs();
            for (const auto& out_node : out_nodes) {
                ov::set_nms_selected_indices(out_node.get_node());
            }
            return true;
        };
        auto m = make_shared<pattern::Matcher>(nms_pattern, matcher_name);
        register_matcher(m, callback);
    }
};
class PropagateNMSPath : public pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PropagateNMSPath");
    PropagateNMSPath() {
        MATCHER_SCOPE(PropagateNMSPath);
        auto node_pattern = pattern::wrap_type<ov::op::v0::Squeeze,
                                               ov::op::v0::Unsqueeze,
                                               ov::op::v1::Reshape,
                                               op::util::BroadcastBase,
                                               ov::op::v1::StridedSlice,
                                               ov::op::v8::Slice,
                                               ov::op::v1::VariadicSplit,
                                               op::util::GatherBase,
                                               ov::op::v0::Concat,
                                               ov::op::v0::Convert,
                                               ov::op::v8::If>();
        matcher_pass_callback callback = [=](pattern::Matcher& m) {
            auto propagate_path = [](const ov::OutputVector& input_nodes, ov::Node* target_node) {
                if (any_of(input_nodes.begin(), input_nodes.end(), [](const Output<Node>& output) {
                        return ov::has_nms_selected_indices(output.get_node());
                    })) {
                    ov::set_nms_selected_indices(target_node);
                }
            };
            auto handle_params = [&propagate_path](std::shared_ptr<ov::op::util::MultiSubGraphOp> node,
                                                   std::shared_ptr<ov::Model> body,
                                                   int body_index) {
                const auto& params = body->get_parameters();
                for (auto input_desc : node->get_input_descriptions(body_index)) {
                    auto param = params[input_desc->m_body_parameter_index];
                    auto input_node = node->input(input_desc->m_input_index).get_source_output();
                    propagate_path({input_node}, param.get());
                }
            };
            auto handle_results = [&propagate_path](std::shared_ptr<ov::op::util::MultiSubGraphOp> node,
                                                    std::shared_ptr<ov::Model> body,
                                                    int body_index) {
                const auto& results = body->get_results();
                for (auto output_desc : node->get_output_descriptions(body_index)) {
                    auto result = results[output_desc->m_body_value_index];
                    const auto& result_inputs = result->input_values();
                    auto output_node = node->output(output_desc->m_output_index).get_node();
                    propagate_path(result_inputs, output_node);
                }
            };

            auto node = m.get_match_root();
            if (ov::is_type<ov::op::util::MultiSubGraphOp>(node)) {
                auto multi_subgraph_op = ov::as_type_ptr<ov::op::util::MultiSubGraphOp>(node);
                const auto& models = multi_subgraph_op->get_functions();

                for (size_t body_idx = 0; body_idx < models.size(); ++body_idx) {
                    handle_params(multi_subgraph_op, models[body_idx], static_cast<int>(body_idx));
                    ov::pass::Manager manager("PropagateNMSPath");
                    manager.register_pass<ov::pass::PropagateNMSPath>();
                    manager.run_passes(models[body_idx]);
                    handle_results(multi_subgraph_op, models[body_idx], static_cast<int>(body_idx));
                }
            } else {
                const auto& inputs = node->input_values();
                propagate_path(inputs, node.get());
            }
            return false;
        };
        auto m = make_shared<pattern::Matcher>(node_pattern, matcher_name);
        register_matcher(m, callback);
    }
};
class UpdateConvertGather : public pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("UpdateConvertGather");
    UpdateConvertGather() {
        MATCHER_SCOPE(UpdateConvertGather);
        auto node_pattern = pattern::wrap_type<op::util::GatherBase>();
        matcher_pass_callback callback = [=](pattern::Matcher& m) {
            auto gather = m.get_match_root();
            auto indices = gather->input_value(1);
            if (!ov::has_nms_selected_indices(indices.get_node()))
                return false;
            gather->get_rt_info()["dontReverseIndices"] = true;
            auto out_type = (indices.get_element_type() == element::i64 ? element::u64 : element::u32);
            auto existing_convert = ov::as_type_ptr<ov::op::v0::Convert>(indices.get_node_shared_ptr());
            if (existing_convert && indices.get_target_inputs().size() == 1) {
                existing_convert->set_convert_element_type(out_type);
                existing_convert->validate_and_infer_types();
            } else {
                auto new_convert_to_unsigned = make_shared<ov::op::v0::Convert>(indices, out_type);
                gather->input(1).replace_source_output(new_convert_to_unsigned);
                copy_runtime_info(gather, new_convert_to_unsigned);
            }
            return true;
        };
        auto m = make_shared<pattern::Matcher>(node_pattern, matcher_name);
        register_matcher(m, callback);
    }
};
}  // namespace pass
}  // namespace ov

ov::pass::ConvertNmsGatherPathToUnsigned::ConvertNmsGatherPathToUnsigned() {
    ADD_MATCHER_FOR_THIS(InitNMSPath)
    ADD_MATCHER_FOR_THIS(PropagateNMSPath)
    ADD_MATCHER_FOR_THIS(UpdateConvertGather)
}
