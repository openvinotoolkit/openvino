// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/file_util.hpp"
#include "utils/model.hpp"

namespace ov {
namespace util {

std::string get_model_type(const std::shared_ptr<ov::Model>& model) {
    if (ov::util::is_dynamic_model(model)) {
        return "dynamic";
    }
    return "static";
}

std::map<std::string, ov::conformance::InputInfo>
get_input_info_by_model(const std::shared_ptr<ov::Model>& model) {
    std::map<std::string, ov::conformance::InputInfo> in_info;
    for (const auto& node : model->get_ordered_ops()) {
        ov::conformance::InputInfo::Range ranges(ov::conformance::DEFAULT_MIN_VALUE, ov::conformance::DEFAULT_MAX_VALUE);
        bool is_const = false;
        if (ov::op::util::is_constant(node)) {
            std::shared_ptr<ov::op::v0::Constant> constant = std::dynamic_pointer_cast<ov::op::v0::Constant>(node);
            auto const_ranges = get_const_ranges(constant,
                                                 constant->get_default_output().get_element_type());
            ranges = const_ranges;
        } else if (!ov::op::util::is_parameter(node)) {
            continue;
        }
        auto partial_shape = node->get_default_output().get_partial_shape();
        in_info.insert({node->get_friendly_name(),
                        ov::conformance::InputInfo(partial_shape, ranges.min, ranges.max, is_const)});
    }
    return in_info;
}

std::map<std::string, ov::conformance::InputInfo>
align_input_info(const std::shared_ptr<ov::Model>& model,
                 const std::shared_ptr<ov::Model>& model_ref,
                 const std::map<std::string, ov::conformance::InputInfo>& in_info,
                 const std::map<std::string, ov::conformance::InputInfo>& in_info_ref,
                 const std::unordered_map<std::string, std::string> &matched_op) {
    std::map<std::string, ov::conformance::InputInfo> updated_input_info = in_info_ref;
    for (const auto& op : model->get_ordered_ops()) {
        const auto op_name = op->get_friendly_name();
        if (!in_info.count(op_name)) {
            continue;
        }
        if (matched_op.count(op_name)) {
            updated_input_info[matched_op.at(op_name)] = in_info.at(op_name);
        }
    }
    return updated_input_info;
}

}  // namespace util
}  // namespace ov