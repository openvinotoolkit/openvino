// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "remove_empty_kv_inputs.hpp"

#include "openvino/core/graph_util.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/op/util/node_util.hpp"
#include "openvino/opsets/opset13.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/validate.hpp"

namespace opp = ov::pass::pattern;

namespace {

// specific function that match subgraph appeared as result of lpt transformations
auto match_down_up_convert_subgraph_after_lpt = [](const ov::Output<ov::Node>& input) {
    auto upconvert = opp::wrap_type<ov::op::v0::Convert>({input}, opp::type_matches(ov::element::f32));

    auto upscale = opp::wrap_type<ov::op::v0::Constant>(opp::rank_equals(0));
    auto upmul = opp::wrap_type<ov::op::v1::Multiply>({upconvert, upscale});

    auto downscale = opp::wrap_type<ov::op::v0::Constant>(opp::rank_equals(0));
    auto downmul = opp::wrap_type<ov::op::v1::Multiply>({upmul, downscale});

    auto downconvert =
        opp::wrap_type<ov::op::v0::Convert>({downmul},
                                            opp::type_matches_any({ov::element::f8e4m3, ov::element::f8e5m2}));

    return downconvert;
};

// diagnostics warnings on OPENVINO_MATCHER_PASS_RTTI() definition: visibility hidden
#ifdef __GNUC__
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wattributes"
#endif

class RemoveEmptyKVTensors : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::LLMCompiledModel::RemoveEmptyKVTensors");

    struct Context {
        std::vector<std::shared_ptr<ov::opset13::Parameter>> old_params;
        using Ref = std::reference_wrapper<Context>;
    };

    RemoveEmptyKVTensors(Context::Ref ctx) {
        auto param = opp::wrap_type<ov::op::v0::Parameter>();
        auto param_or =
            std::make_shared<opp::op::Or>(ov::OutputVector{param, match_down_up_convert_subgraph_after_lpt(param)});

        auto concat = opp::wrap_type<ov::op::v0::Concat>({param_or, opp::any_input()});

        auto callback = [=](opp::Matcher& m) {
            auto& node_to_output = m.get_pattern_value_map();
            auto matched_param = ov::as_type_ptr<ov::op::v0::Parameter>(node_to_output.at(param).get_node_shared_ptr());
            auto matched_node_concat = node_to_output.at(concat).get_node_shared_ptr();

            ctx.get().old_params.push_back(matched_param);

            // Use concat's first input source node to find ShapeOf users.
            // This works universally for both plain parameter and down_up_convert subgraph cases,
            // because in the subgraph case matched_param->get_users() would return the Convert
            // node (first node of the subgraph), not the ShapeOf.
            auto concat_input0_node = matched_node_concat->input(0).get_source_output().get_node_shared_ptr();
            auto users = concat_input0_node->get_users();

            // In subgraph case the parameter itself may also have a ShapeOf user,
            // so check both the concat input node and the parameter.
            if (concat_input0_node != matched_param) {
                auto param_users = matched_param->get_users();
                users.insert(users.end(), param_users.begin(), param_users.end());
            }

            // Find and replace ShapeOf nodes with constants
            for (auto& user : users) {
                if (ov::is_type<ov::op::v3::ShapeOf>(user)) {
                    auto cst_node =
                        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, matched_param->get_shape());
                    ov::replace_node(user, cst_node);
                }
            }

            // Redirect second concat input to every node which reads from concat
            auto curr_kv_tensor = matched_node_concat->input(1).get_source_output();
            for (auto target_input : matched_node_concat->output(0u).get_target_inputs()) {
                target_input.replace_source_output(curr_kv_tensor);
            }

            return true;
        };
        register_matcher(std::make_shared<opp::Matcher>(concat, "RemoveEmptyKVTensors"), std::move(callback));
    }
};

#ifdef __GNUC__
#    pragma GCC diagnostic pop
#endif

bool remove_empty_kv_inputs(std::shared_ptr<ov::Model> model) {
    ov::pass::GraphRewrite rewr;
    RemoveEmptyKVTensors::Context ctx;
    rewr.add_matcher<RemoveEmptyKVTensors>(std::ref(ctx));
    rewr.run_on_model(model);
    for (auto old_param : ctx.old_params) {
        model->remove_parameter(old_param);
    }
    ov::pass::Validate().run_on_model(model);
    // NB: if old_params is not empty - pass has been applied
    return !ctx.old_params.empty();
}

}  // namespace

namespace ov::npuw {

bool RemoveEmptyKVInputs::run_on_model(const std::shared_ptr<ov::Model>& model) {
    return remove_empty_kv_inputs(model);
}

}  // namespace ov::npuw
