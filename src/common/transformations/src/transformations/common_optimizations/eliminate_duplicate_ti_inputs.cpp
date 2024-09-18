// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/eliminate_duplicate_ti_inputs.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/tensor_iterator.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using namespace ov::op::util;

ov::pass::EliminateDuplicateTIInputs::EliminateDuplicateTIInputs() {
    MATCHER_SCOPE(EliminateDuplicateTIInputs);
    auto ti = pattern::wrap_type<ov::op::v0::TensorIterator>();
    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto ti = ov::as_type_ptr<ov::op::v0::TensorIterator>(m.get_match_root());
        if (ti == nullptr) {
            return false;
        }

        std::vector<std::shared_ptr<SubGraphOp::InputDescription>> should_stay;
        std::map<std::shared_ptr<SubGraphOp::InputDescription>,
                 std::vector<std::shared_ptr<SubGraphOp::InputDescription>>>
            need_to_eliminate;
        auto input_descs = ti->get_input_descriptions();
        for (auto& key : input_descs) {
            auto is_equal = [&](const std::shared_ptr<SubGraphOp::InputDescription>& input) -> bool {
                if (ti->input_value(input->m_input_index) == ti->input_value(key->m_input_index)) {
                    auto invariant_l = std::dynamic_pointer_cast<SubGraphOp::InvariantInputDescription>(input);
                    auto invariant_r = std::dynamic_pointer_cast<SubGraphOp::InvariantInputDescription>(key);
                    if (invariant_l && invariant_r) {
                        return true;
                    }

                    auto slice_l = std::dynamic_pointer_cast<SubGraphOp::SliceInputDescription>(input);
                    auto slice_r = std::dynamic_pointer_cast<SubGraphOp::SliceInputDescription>(key);

                    if (slice_l && slice_r) {
                        return slice_l->m_axis == slice_r->m_axis && slice_l->m_start == slice_r->m_start &&
                               slice_l->m_end == slice_r->m_end && slice_l->m_part_size == slice_r->m_part_size &&
                               slice_l->m_stride == slice_r->m_stride;
                    }

                    auto merged_l = std::dynamic_pointer_cast<SubGraphOp::MergedInputDescription>(input);
                    auto merged_r = std::dynamic_pointer_cast<SubGraphOp::MergedInputDescription>(key);

                    if (merged_l && merged_r) {
                        return merged_l->m_body_value_index == merged_r->m_body_value_index;
                    }
                }
                return false;
            };
            auto it = std::find_if(should_stay.begin(), should_stay.end(), is_equal);
            if (it == should_stay.end()) {
                should_stay.push_back(key);
            } else {
                need_to_eliminate[*it].push_back(key);
            }
        }

        if (need_to_eliminate.empty()) {
            return false;
        }

        const auto& body = ti->get_function();
        // re-connect outputs of duplicate Parameters
        auto parameters = body->get_parameters();
        for (const auto& it : need_to_eliminate) {
            for (const auto& redundant : it.second) {
                parameters[redundant->m_body_parameter_index]->output(0).replace(
                    parameters[it.first->m_body_parameter_index]);
            }
        }

        // Create new TI
        auto new_ti = std::make_shared<ov::op::v0::TensorIterator>();
        new_ti->set_output_descriptions(0, ti->get_output_descriptions());
        ov::ParameterVector new_params;
        for (const auto& remain : should_stay) {
            auto par = body->get_parameters()[remain->m_body_parameter_index];
            new_params.push_back(par);
        }
        auto new_body = std::make_shared<ov::Model>(body->get_results(), new_params);
        new_ti->set_body(new_body);

        for (const auto& remain : should_stay) {
            auto par = body->get_parameters()[remain->m_body_parameter_index];
            auto in = ti->input_value(remain->m_input_index);
            if (auto invariant = std::dynamic_pointer_cast<SubGraphOp::InvariantInputDescription>(remain)) {
                new_ti->set_invariant_input(par, in);
            } else if (auto merged = std::dynamic_pointer_cast<SubGraphOp::MergedInputDescription>(remain)) {
                auto results = body->get_results();
                new_ti->set_merged_input(par, in, results[merged->m_body_value_index]);
            } else if (auto slice = std::dynamic_pointer_cast<SubGraphOp::SliceInputDescription>(remain)) {
                new_ti->set_sliced_input(par,
                                         in,
                                         slice->m_start,
                                         slice->m_stride,
                                         slice->m_part_size,
                                         slice->m_end,
                                         slice->m_axis);
            }
        }
        new_ti->validate_and_infer_types();

        copy_runtime_info(ti, new_ti);
        replace_node(ti, new_ti);
        new_ti->set_friendly_name(ti->get_friendly_name());
        return true;
    };
    auto m = std::make_shared<pattern::Matcher>(ti, matcher_name);
    this->register_matcher(m, callback);
}
