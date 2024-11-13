// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/mark_dequantization_subgraph.hpp"

#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/fp16_compression/mark_decompression_convert_constant_folding.hpp"
#include "transformations/rt_info/dequantization_node.hpp"
#include "transformations/rt_info/disable_constant_folding.hpp"
#include "transformations/rt_info/keep_const_precision.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;
using namespace ov::op;
using namespace ov::pass::pattern;

/**
 * @ingroup ov_transformation_common_api
 * @brief TBA
 */
class TRANSFORMATIONS_API MarkDequantization : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("MarkDequantization", "0");
    explicit MarkDequantization(const element::TypeVector& precisions,
                                bool fold_subtract_const = false,
                                bool fold_multiply_const = true);
};

/**
 * @ingroup ov_transformation_common_api
 * @brief TBA
 */
class TRANSFORMATIONS_API KeepConstsPrecision : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("KeepConstsPrecision", "0");
    explicit KeepConstsPrecision(const element::TypeVector& precisions);
};

namespace {

bool check_precision(const ov::element::Type_t type_to_check, const ov::element::TypeVector& precisions) {
    return std::find(precisions.begin(), precisions.end(), type_to_check) != precisions.end();
};

using RTInfoSetter = std::function<void(const std::shared_ptr<ov::Node>& node)>;
void set_rt_info(const PatternValueMap& pt_map,
                 const RTInfoSetter& rt_info_setter,
                 const NodeVector& pattern_nodes,
                 const ov::element::TypeVector& precisions) {
    for (const auto& pattern_node : pattern_nodes) {
        if (pt_map.count(pattern_node)) {
            auto node = pt_map.at(pattern_node).get_node_shared_ptr();
            if (ov::as_type_ptr<v0::Convert>(node) && !check_precision(node->get_input_element_type(0), precisions)) {
                continue;
            }
            rt_info_setter(node);
        }
    }
};
}  // namespace

MarkDequantization::MarkDequantization(const element::TypeVector& precisions,
                                       const bool fold_subtract_const,
                                       const bool fold_multiply_const) {
    // Dequantization subgraph may have two forms: with and without Subtract
    //
    //    Input                                 Input
    //      |                                     |
    //   Convert  zero point           OR       Convert   scale
    //       \     /                               \      /
    //       Subtract   scale                      Multiply
    //           \      /
    //           Multiply
    //
    auto input_pattern = any_input();
    auto convert_pattern = wrap_type<v0::Convert>({input_pattern}, consumers_count(1));

    // zero points:
    auto zp_pattern = any_input();
    auto zp_convert_pattern = optional<v0::Convert>(zp_pattern);
    auto subtract_pattern = optional<v1::Subtract>({convert_pattern, zp_convert_pattern});

    // scale:
    auto scale_pattern = any_input();
    auto scale_convert_pattern = optional<v0::Convert>(scale_pattern);
    auto multiply_pattern = wrap_type<v1::Multiply>({subtract_pattern, scale_convert_pattern});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) -> bool {
        const auto& pt_map = m.get_pattern_value_map();
        auto convert = pt_map.at(convert_pattern);
        const auto multiply = m.get_match_root();

        if (transformation_callback(multiply)) {
            return false;
        }

        // Multiply and Subtract have to be marked as dq
        set_rt_info(pt_map, mark_as_dequantization_node, {subtract_pattern, multiply_pattern}, {/* not applicable */});

        // Convert might be presented on scales, zp and data_input.
        // Depending on the transformation arguments they have to be marked/unmarked with disable_cf rt_info.
        NodeVector converts_to_mark = {convert_pattern};
        NodeVector converts_to_unmark = {};
        if (fold_subtract_const) {
            converts_to_unmark.push_back(zp_convert_pattern);
        } else {
            converts_to_mark.push_back(zp_convert_pattern);
        }

        if (fold_multiply_const) {
            converts_to_unmark.push_back(scale_convert_pattern);
        } else {
            converts_to_mark.push_back(scale_convert_pattern);
        }

        set_rt_info(pt_map, disable_constant_folding, converts_to_mark, precisions);
        set_rt_info(pt_map, enable_constant_folding, converts_to_unmark, precisions);
        return false;
    };

    auto m = std::make_shared<Matcher>(multiply_pattern, "MarkDQ");
    this->register_matcher(m, callback);
}

KeepConstsPrecision::KeepConstsPrecision(const element::TypeVector& precisions) {
    // Dequantization subgraph may have two forms: with and without Subtract
    //
    //    Input                                 Input
    //      |                                     |
    //   Convert  zero point           OR       Convert   scale
    //       \     /                               \      /
    //       Subtract   scale                      Multiply
    //           \      /
    //           Multiply
    //
    auto input_pattern = any_input();
    auto convert_pattern = wrap_type<v0::Convert>({input_pattern}, consumers_count(1));

    // zero points:
    auto zp_pattern = any_input();
    auto zp_convert_pattern = optional<v0::Convert>(zp_pattern);
    auto subtract_pattern = optional<v1::Subtract>({convert_pattern, zp_convert_pattern});

    // scale:
    auto scale_pattern = any_input();
    auto scale_convert_pattern = optional<v0::Convert>(scale_pattern);
    auto multiply_pattern = wrap_type<v1::Multiply>({subtract_pattern, scale_convert_pattern});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) -> bool {
        const auto& pt_map = m.get_pattern_value_map();
        const auto multiply = m.get_match_root();

        if (transformation_callback(multiply)) {
            return false;
        }

        NodeVector keep_const_precisions = {};
        for (const auto& pattern_node : {input_pattern, zp_pattern, scale_pattern}) {
            if (pt_map.count(pattern_node)) {
                auto node = pt_map.at(pattern_node).get_node_shared_ptr();
                if (ov::as_type_ptr<v0::Constant>(node)) {
                    ov::enable_keep_const_precision(node);
                }
            }
        }
        return false;
    };

    auto m = std::make_shared<Matcher>(multiply_pattern, "KeepConstsPrecision");
    this->register_matcher(m, callback);
}

bool pass::MarkDequantizationAndDecompression::run_on_model(const std::shared_ptr<ov::Model>& m) {
    ov::pass::Manager manager("MarkDequantizationAndDecompressionManager");
    manager.register_pass<MarkDequantization>(m_precisions, m_fold_subtract_const, m_fold_multiply_const);
    manager.register_pass<DisableDecompressionConvertConstantFolding>();
    manager.register_pass<ConstantFolding>();
    manager.register_pass<KeepConstsPrecision>(m_precisions);
    manager.run_passes(m);

    return false;
}
