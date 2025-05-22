// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/add.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"
#include "transformations_visibility.hpp"

#include "openvino/op/range.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/einsum.hpp"

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/einsum.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/utils/utils.hpp"

static ov::Output<ov::Node> insert_identity(const ov::Output<ov::Node>& in_node) {
    auto axis_1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto identity_1 = std::make_shared<ov::op::v0::Unsqueeze>(in_node, axis_1);
    return std::make_shared<ov::op::v15::Squeeze>(identity_1, axis_1);
}

using ResultVector = std::vector<std::shared_ptr<ov::op::v0::Result>>;
namespace ov {
namespace pass {

class TRANSFORMATIONS_API PositionIDsReplacer;
class TRANSFORMATIONS_API PositionIDsReplacerQwen;

}  // namespace pass
}  // namespace ov

class ov::pass::PositionIDsReplacer : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PositionIDsReplacer");
    explicit PositionIDsReplacer(const Output<Node>& position_ids);
};

/**
 * @brief Qwen model expects data processing in order, the "position ids" input is detached and
 * is not explicitly used in the model. The model uses implicitly defined "position ids" based
 * on the past KV cache size.
 *
 * To use this model in Continuous batching mode, we need to apply position_ids and
 * use the corresponding rotary_emb_cos/rotary_emb_sin.
 * For this, we replace
 *      rotary_emb_cos/rotary_emb_sin -> Slice -> Slice
 * With
 *      rotary_emb_cos/rotary_emb_sin -> Gather(by position_ids)
 * Which enables applying RoPE for each token independently of their order in the input tensor.
 */
class ov::pass::PositionIDsReplacerQwen : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PositionIDsReplacerQwen");
    explicit PositionIDsReplacerQwen(const Output<Node>& position_ids);
};

class ReplaceRoPERangeWithPositionIds : public ov::pass::MatcherPass {

public:
    ReplaceRoPERangeWithPositionIds(const std::shared_ptr<ov::Node>& position_ids, int& layer_index1, const std::shared_ptr<ov::Node>& concat) {
        using namespace ov::op;
        using namespace ov;
        using namespace ov::pass::pattern;
        // 1. Pattern: Range -> (optional Convert/Reshape) -> Einsum with 'i,j->ij'
        auto range_pattern = wrap_type<v4::Range>({any_input(), any_input(), any_input()});
        auto einsum_pattern = wrap_type<v7::Einsum>({range_pattern, any_input()});
 
        matcher_pass_callback callback = [=, &position_ids, &layer_index1](Matcher& m) {
            std::cout << "ReplaceRoPERangeWithPositionIds start" << std::endl;
            auto pattern_map = m.get_pattern_value_map();
            auto einsum = std::dynamic_pointer_cast<v7::Einsum>(m.get_match_root());
            if (!einsum)
                return false;

            std::cout << "1" << std::endl;
            std::string equation = einsum->get_equation();
            if (equation != "i,j->ij")
                return false;
 
            auto range = pattern_map[range_pattern];

 
            if (einsum->input(0).get_element_type() != concat->output(0).get_element_type()) {
                std::cout << "inside" << std::endl;
                auto cast = std::make_shared<v0::Convert>(concat, einsum->input(0).get_element_type());
                cast->set_friendly_name("position_ids_cast");
                einsum->input(0).replace_source_output(cast->output(0));
                std::cout << "REPLACED" << std::endl;
            } else {
                std::cout << "else" << std::endl;
                einsum->input(0).replace_source_output(concat);
            }
 
            std::cout << "ReplaceRoPERangeWithPositionIds finish" << std::endl;
            return true;
        };

        // matcher_pass_callback callback = [=, &position_ids, &dbg_results, &layer_index1](Matcher& m) {
        //     // std::cout << "ReplaceRoPERangeWithPositionIds start" << std::endl;
        //     auto pattern_map = m.get_pattern_value_map();
        //     auto einsum = std::dynamic_pointer_cast<v7::Einsum>(m.get_match_root());
        //     if (!einsum)
        //         return false;

        //     std::string equation = einsum->get_equation();
        //     if (equation != "i,j->ij")
        //         return false;
 
        //     auto range = pattern_map[range_pattern];
        //     auto dbg_result = std::make_shared<v0::Result>(range);
        //     dbg_result->get_output_tensor(0).set_names({"range_result_" + std::to_string(layer_index1)});
        //     dbg_results.push_back(dbg_result);
 
        //     // std::cout << "ReplaceRoPERangeWithPositionIds finish" << std::endl;
        //     return true;
        // };
 
 
        auto m = std::make_shared<Matcher>(einsum_pattern, "ReplaceRoPERangeWithPositionIds");
        this->register_matcher(m, callback);
    }
};

class CustomModelPass : public ov::pass::ModelPass {
public:
    CustomModelPass(const std::shared_ptr<ov::Node>& position_ids) : m_position_ids(position_ids) {}
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override {
        int layer_index1 = 0;

        auto var_info = ov::op::util::VariableInfo{ov::PartialShape{-1}, ov::element::i64, "aaaaa" + std::to_string(layer_index1++)};
        auto var = std::make_shared<ov::op::util::Variable>(var_info);
        auto read_value = std::make_shared<ov::op::v6::ReadValue>(ov::op::v0::Constant::create(ov::element::i64, ov::Shape{0}, {}), var);
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{read_value->output(0), m_position_ids->output(0)}, 0);
        auto assign = std::make_shared<ov::op::v6::Assign>(concat, var);

        ov::pass::Manager manager;
        manager.set_per_pass_validation(false);
        manager.register_pass<ReplaceRoPERangeWithPositionIds>(m_position_ids, layer_index1, concat);
        manager.run_passes(model);
        model->add_variables({var});
        return true;
    }
    std::shared_ptr<ov::Node> m_position_ids;
};
