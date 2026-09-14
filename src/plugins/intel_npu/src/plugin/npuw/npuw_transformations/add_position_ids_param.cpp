// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "add_position_ids_param.hpp"

#include <unordered_set>

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

// Creates a fresh `position_ids` [batch, seq] parameter plus a batch-squeezed [seq] view of it.
std::pair<std::shared_ptr<ov::op::v0::Parameter>, std::shared_ptr<ov::op::v0::Squeeze>> make_position_ids() {
    auto position_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1, -1});
    set_node_name(position_ids, "position_ids");
    auto position_ids_squeezed =
        std::make_shared<ov::op::v0::Squeeze>(position_ids,
                                              ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0}));
    return {position_ids, position_ids_squeezed};
}

// TODO: Consolidate with similar pattern in prepare_embedding_model.cpp
class HardcodedPositionIdsMatcher : public ov::pass::MultiMatcher {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::HardcodedPositionIdsMatcher");
    explicit HardcodedPositionIdsMatcher(ov::ParameterVector& new_params) {
        // The oldest exports shared one Range between RoPE, Causal Mask and the Gated Short
        // Convolution Block indexing, and offset it inside Range itself:
        //
        //                                          -> Convert -> RoPE
        //                                         |
        //                  -> Unsqueeze -> Unsqueeze
        //                 |                       |
        //   Range --------                         -> Unsqueeze -> LessEqual (Causal Mask)
        //                 |
        //                  -> Clamp -> ScatterNDUpdate (Gated Short Convolution Block)
        //
        // transformers>=5.4 aranges from zero and adds the past length on top of it instead, gives
        // RoPE and Causal Mask a Range each, and drops the Clamp path altogether, leaving no
        // branching at all:
        //
        //   Range -> Add -> Unsqueeze -> Unsqueeze -> Convert -> RoPE
        //
        auto range = opp::wrap_type<ov::op::v4::Range>();
        auto position = opp::optional<ov::op::v1::Add>({range->output(0), opp::any_input()});
        auto unsqueeze_axes = opp::wrap_type<ov::op::v0::Constant>();
        auto unsqueeze = opp::wrap_type<ov::op::v0::Unsqueeze>({position, unsqueeze_axes});

        auto unsqueeze1_axes = opp::wrap_type<ov::op::v0::Constant>();
        auto unsqueeze1 = opp::wrap_type<ov::op::v0::Unsqueeze>({unsqueeze, unsqueeze1_axes});

        // FIXME: Convert probably needs to be made optional in future as in similar transformation from
        // prepare_embedding_model.cpp
        auto convert = opp::wrap_type<ov::op::v0::Convert>({unsqueeze1});
        auto matmul = opp::wrap_type<ov::op::v0::MatMul>({opp::any_input(), convert});
        auto transpose = opp::wrap_type<ov::op::v1::Transpose>({matmul, opp::any_input()});

        auto concat = opp::wrap_type<ov::op::v0::Concat>({transpose, transpose});
        auto cos = opp::wrap_type<ov::op::v0::Cos>(concat);
        auto sin = opp::wrap_type<ov::op::v0::Sin>(concat);

        ov::pass::MultiMatcher::Callback callback = [=, &new_params](const auto& m) {
            // NOTE: Range that mimics `position_ids` is consumed by RoPE operation, but might as well be used for
            //       Causal Mask creation (via LessEqual, ex.: transformers==5.0.0) and Gated Short Convolution Block's
            //       ScatterNDUpdate operation (ex.: transformers==4.57.6). The rewrite below only rewires the RoPE
            //       path and ScatterNDUpdate path (if exists) to use new `position_ids` parameter.
            //       For static shapes case, it is not right to use actual `position_ids` for the second argument of
            //       LessEqual operation (=Q range) in creation of the causal mask. This setup will only allow positions
            //       from the left till the real current positions in the sequence (inclusively), while our current
            //       items are lied at the right end of the static `input_ids` after a whole window of padding. Thus,
            //       the Range should be preserved for Causal Mask creation.
            auto& pattern_to_output = m.at(cos).front();

            auto range_node = pattern_to_output.at(range).get_node_shared_ptr();
            // Point of branching:
            auto unsqueeze1_node = pattern_to_output.at(unsqueeze1).get_node_shared_ptr();
            auto convert_node = pattern_to_output.at(convert).get_node_shared_ptr();

            // Create `position_ids` parameter
            auto [position_ids, position_ids_squeezed] = make_position_ids();

            // Create new Unsqueeze node for point of branching
            auto unsqueeze1_node_copy =
                unsqueeze1_node->clone_with_new_inputs({position_ids, unsqueeze1_node->input_value(1)});
            convert_node->input(0).replace_source_output(unsqueeze1_node_copy->output(0));

            // FIXME: For Gated Short Convolution Block, there is ScatterNDUpdate that also consumes generated
            // positions in old IRs.
            //        It seems to right to use the newly created `position_ids` for it as well, however, real tests show
            //        no difference against usage of hardcoded QRange: both are similarly accurate.
            OPENVINO_ASSERT(range_node->get_output_size() == 1, "Range node should have exactly one output");
            auto range_consumers = range_node->get_output_target_inputs(0);
            for (auto&& consumer : range_consumers) {
                // Only the Clamp path is remained to rewire.
                if (consumer.get_node()->get_type_name() == std::string("Clamp")) {
                    consumer.replace_source_output(position_ids_squeezed->output(0));
                }
            }

            new_params.push_back(position_ids);
            return true;
        };

        register_patterns({sin, cos}, std::move(callback));
    }
};

// Gather-based (cache-table) RoPE. Instead of computing cos/sin, ONNX GroupQueryAttention-style exports
// precompute cos/sin cache tables and index into them with the position ids. The `position_ids` therefore
// drives a Gather rather than a Cos/Sin, and the branch has no MatMul/Transpose/Concat at all:
//
// Range -> [Convert] -> Unsqueeze -> [Add(past_len)] -> Squeeze -> [Maximum] -> [Minimum] --> Gather (cos)
//                                                                                         --> Gather (sin)
//
// The Maximum/Minimum pair is a clip of the absolute position into the cache bounds; both the
// clip and the past-length Add are optional across exports.
class GatheredPositionIdsMatcher : public ov::pass::MultiMatcher {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::GatheredPositionIdsMatcher");
    explicit GatheredPositionIdsMatcher(ov::ParameterVector& new_params) {
        auto range = opp::wrap_type<ov::op::v4::Range>();
        auto convert = opp::optional<ov::op::v0::Convert>({range->output(0)});
        auto unsqueeze = opp::wrap_type<ov::op::v0::Unsqueeze>({convert, opp::any_input()});
        auto add = opp::optional<ov::op::v1::Add>({unsqueeze->output(0), opp::any_input()});
        auto squeeze = opp::wrap_type<ov::op::v0::Squeeze>({add, opp::any_input()});
        auto clip_max = opp::optional<ov::op::v1::Maximum>({squeeze->output(0), opp::any_input()});
        auto clip_min = opp::optional<ov::op::v1::Minimum>({clip_max->output(0), opp::any_input()});
        // A single Gather root matches both the cos-cache and sin-cache lookups (identical structure);
        // they share the same `squeeze`, so one `position_ids` parameter is created regardless.
        auto gather = opp::wrap_type<ov::op::v8::Gather>({opp::any_input(), clip_min, opp::any_input()});

        ov::pass::MultiMatcher::Callback callback = [=, &new_params](const auto& m) {
            // The cos-cache and sin-cache Gathers share the same position-producing `Squeeze`; some exports
            // may also repeat it per layer. All of them carry the same logical positions, so create a single
            // `position_ids` parameter and rewire every distinct `Squeeze` to it. Dedupe by the matched Squeeze
            // node to avoid double-rewiring (and to avoid emitting colliding same-named parameters).
            //
            // The gathered index is 1-D [seq]; squeeze the batch dim off `position_ids` to match. This
            // bypasses the hardcoded Range/Add(past_len) while preserving the downstream clip and Gathers.
            std::unordered_set<ov::Node*> handled_squeezes;
            auto [position_ids, position_ids_squeezed] = make_position_ids();
            for (auto&& match : m.at(gather)) {
                auto squeeze_node = match.at(squeeze).get_node_shared_ptr();
                if (!handled_squeezes.insert(squeeze_node.get()).second) {
                    continue;
                }
                OPENVINO_ASSERT(squeeze_node->get_output_size() == 1, "Squeeze node should have exactly one output");
                for (auto&& consumer : squeeze_node->get_output_target_inputs(0)) {
                    consumer.replace_source_output(position_ids_squeezed->output(0));
                }
            }

            new_params.push_back(position_ids);
        };

        register_patterns({gather}, std::move(callback));
    }
};

#ifdef __GNUC__
#    pragma GCC diagnostic pop
#endif
}  // anonymous namespace

bool ov::npuw::AddPositionIdsParam::run_on_model(const std::shared_ptr<ov::Model>& model) {
    ov::ParameterVector new_parameters;
    {
        ov::pass::Manager manager("add-position-ids-param");
        manager.set_per_pass_validation(false);
        manager.register_pass<HardcodedPositionIdsMatcher>(new_parameters);
        manager.run_passes(model);
    }
    if (new_parameters.empty()) {
        // The Cos/Sin- and Gather-based RoPE flavors are mutually exclusive within a model: run the
        // Gather matcher only if the former didn't fire, so a single `position_ids` is ever introduced.
        ov::pass::Manager manager("add-position-ids-param-gathered");
        manager.set_per_pass_validation(false);
        manager.register_pass<GatheredPositionIdsMatcher>(new_parameters);
        manager.run_passes(model);
    }

    model->add_parameters(new_parameters);
    model->validate_nodes_and_infer_types();
    return true;
}
