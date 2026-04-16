// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "add_position_ids_param.hpp"

#include "openvino/op/ops.hpp"
#include "openvino/openvino.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/multi_matcher.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/validate.hpp"

namespace opp = ov::pass::pattern;

namespace {

// diagnostics warnings on OPENVINO_MATCHER_PASS_RTTI() definition: visibility hidden
#ifdef __GNUC__
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wattributes"
#endif

void set_node_name(std::shared_ptr<ov::Node> node, const std::string& name) {
    node->set_friendly_name(name);
    node->get_output_tensor(0).set_names({name});
}

using NodePair = std::pair<std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>>;

class PositionIdsMatcher : public ov::pass::MultiMatcher {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::PositionIdsMatcher");
    explicit PositionIdsMatcher(ov::ParameterVector& new_params) {
        auto range = opp::wrap_type<ov::op::v4::Range>();
        auto unsqueeze_axes = opp::wrap_type<ov::op::v0::Constant>();
        auto unsqueeze = opp::wrap_type<ov::op::v0::Unsqueeze>({range, unsqueeze_axes});

        auto unsqueeze1_axes = opp::wrap_type<ov::op::v0::Constant>();
        auto unsqueeze1 = opp::wrap_type<ov::op::v0::Unsqueeze>({unsqueeze, unsqueeze1_axes});

        auto convert = opp::optional<ov::op::v0::Convert>({unsqueeze1});
        auto matmul = opp::wrap_type<ov::op::v0::MatMul>({opp::any_input(), convert});
        auto transpose = opp::wrap_type<ov::op::v1::Transpose>({matmul, opp::any_input()});

        auto concat = opp::wrap_type<ov::op::v0::Concat>({transpose, transpose});
        auto cos = opp::wrap_type<ov::op::v0::Cos>(concat);
        auto sin = opp::wrap_type<ov::op::v0::Sin>(concat);

        ov::pass::MultiMatcher::Callback callback = [=, &new_params](const auto& m) {
            auto& pattern_to_output = m.at(cos).front();

            auto range_node = pattern_to_output.at(range).get_node_shared_ptr();
            auto position_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1,-1});
            set_node_name(position_ids, "position_ids");
            auto position_ids_squeezed = std::make_shared<ov::op::v0::Squeeze>(position_ids, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0}));

            OPENVINO_ASSERT(range_node->get_output_size() == 1, "Range node should have exactly one output");
            auto range_consumers = range_node->get_output_target_inputs(0);
            for (auto&& consumer : range_consumers) {
                consumer.replace_source_output(position_ids_squeezed->output(0));
            }

            new_params.push_back(position_ids);
            return true;
        };

        register_patterns({sin, cos}, std::move(callback));
    }
};

#ifdef __GNUC__
#    pragma GCC diagnostic pop
#endif
}  // anonymous namespace

bool ov::npuw::AddPositionIdsParam::run_on_model(const std::shared_ptr<ov::Model>& model) {
    ov::ParameterVector new_parameters;
    ov::pass::Manager manager("add-position-ids-param");
    manager.set_per_pass_validation(false);
    manager.register_pass<PositionIdsMatcher>(new_parameters);
    OPENVINO_ASSERT(manager.run_passes(model), "Failed to find position_ids subgraph in the model");

    model->add_parameters(new_parameters);
    model->validate_nodes_and_infer_types();
    return true;
}
