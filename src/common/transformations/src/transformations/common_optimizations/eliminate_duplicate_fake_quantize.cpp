// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/eliminate_duplicate_fake_quantize.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

using ov::pass::pattern::any_input;
using ov::pass::pattern::Matcher;
using ov::pass::pattern::wrap_type;

namespace v0 = ov::op::v0;

ov::pass::EliminateDuplicateFakeQuantize::EliminateDuplicateFakeQuantize() {
    MATCHER_SCOPE(EliminateDuplicateFakeQuantize);
    
    // Pattern: any_input -> FQ1 -> FQ2
    auto input_pattern = any_input();
    auto fq1_pattern = wrap_type<v0::FakeQuantize>(
        {input_pattern, any_input(), any_input(), any_input(), any_input()},
        ov::pass::pattern::consumers_count(1));
    auto fq2_pattern = wrap_type<v0::FakeQuantize>(
        {fq1_pattern, any_input(), any_input(), any_input(), any_input()});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        std::cout << "EliminateDuplicateFakeQuantize callback triggered" << std::endl;
        const auto& pattern_value_map = m.get_pattern_value_map();
        
        auto fq1 = ov::as_type_ptr<v0::FakeQuantize>(
            pattern_value_map.at(fq1_pattern).get_node_shared_ptr());
        auto fq2 = ov::as_type_ptr<v0::FakeQuantize>(
            pattern_value_map.at(fq2_pattern).get_node_shared_ptr());
        
        if (!fq1 || !fq2)
            return false;
        
        // Check if both FakeQuantize have the same number of levels
        if (fq1->get_levels() != fq2->get_levels())
            return false;
        
        // Get FQ1's output range (indices 3 and 4)
        auto fq1_output_low = fq1->input_value(3);
        auto fq1_output_high = fq1->input_value(4);
        
        // Get FQ2's input range (indices 1 and 2) and output range (indices 3 and 4)
        auto fq2_input_low = fq2->input_value(1);
        auto fq2_input_high = fq2->input_value(2);
        auto fq2_output_low = fq2->input_value(3);
        auto fq2_output_high = fq2->input_value(4);
        
        // Try to get constant values for range checking
        auto fq1_ol_const = ov::as_type_ptr<v0::Constant>(fq1_output_low.get_node_shared_ptr());
        auto fq1_oh_const = ov::as_type_ptr<v0::Constant>(fq1_output_high.get_node_shared_ptr());
        auto fq2_il_const = ov::as_type_ptr<v0::Constant>(fq2_input_low.get_node_shared_ptr());
        auto fq2_ih_const = ov::as_type_ptr<v0::Constant>(fq2_input_high.get_node_shared_ptr());
        
        // Check if FQ1's output range and FQ2's input range are constants
        // If they are, we can check for compatibility
        bool ranges_compatible = true;
        if (fq1_ol_const && fq1_oh_const && fq2_il_const && fq2_ih_const) {
            auto fq1_ol_val = fq1_ol_const->cast_vector<float>();
            auto fq1_oh_val = fq1_oh_const->cast_vector<float>();
            auto fq2_il_val = fq2_il_const->cast_vector<float>();
            auto fq2_ih_val = fq2_ih_const->cast_vector<float>();
            
            // Check if all vectors have the same size
            if (fq1_ol_val.size() != fq1_oh_val.size() ||
                fq2_il_val.size() != fq2_ih_val.size() ||
                fq1_ol_val.size() != fq2_il_val.size()) {
                return false;
            }
            
            // Check if ranges are compatible (exact match or FQ2 can clip FQ1's output)
            const float eps = 1e-6f;
            for (size_t i = 0; i < fq1_ol_val.size(); ++i) {
                // Check if ranges exactly match
                bool exact_match = (std::abs(fq1_ol_val[i] - fq2_il_val[i]) < eps &&
                                   std::abs(fq1_oh_val[i] - fq2_ih_val[i]) < eps);
                
                // Check if FQ1 output is within FQ2 input range (subset)
                bool is_subset = (fq1_ol_val[i] >= fq2_il_val[i] - eps &&
                                 fq1_oh_val[i] <= fq2_ih_val[i] + eps);
                
                if (!exact_match && !is_subset) {
                    // Ranges don't match and FQ2 will clip - this might be intentional
                    // Only merge if the mismatch is small (< 5% difference)
                    float range1 = fq1_oh_val[i] - fq1_ol_val[i];
                    float range2 = fq2_ih_val[i] - fq2_il_val[i];
                    float diff = std::abs(range1 - range2);
                    if (range1 > 0 && diff / range1 > 0.05f) {
                        ranges_compatible = false;
                        break;
                    }
                }
            }
        }
        
        if (!ranges_compatible)
            return false;
        
        // Create merged FakeQuantize:
        // - Use FQ1's input (index 0) and input range (indices 1, 2)
        // - Use FQ2's output range (indices 3, 4)
        // - Use same levels as both FQs
        auto merged_fq = std::make_shared<v0::FakeQuantize>(
            fq1->input_value(0),     // Original input to FQ1
            fq1->input_value(1),     // FQ1's input_low
            fq1->input_value(2),     // FQ1's input_high
            fq2_output_low,           // FQ2's output_low
            fq2_output_high,          // FQ2's output_high
            fq1->get_levels()         // Same levels
        );
        
        merged_fq->set_friendly_name(fq2->get_friendly_name());
        ov::copy_runtime_info({fq1, fq2}, merged_fq);
        ov::replace_node(fq2, merged_fq);
        
        return true;
    };

    auto m = std::make_shared<Matcher>(fq2_pattern, matcher_name);
    register_matcher(m, callback);
}
