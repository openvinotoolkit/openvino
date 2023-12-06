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
                 const std::map<std::string, std::string> &matched_op) {
    bool is_update_required = !matched_op.empty();
    if (!is_update_required) {
        for (const auto& ref_item : in_info_ref) {
            if (!in_info.count(ref_item.first)) {
                is_update_required = true;
                break;
            } else if (in_info.at(ref_item.first).is_const != ref_item.second.is_const) {
                throw std::runtime_error("Impossible to update input info!!!");
            }
        }
    }

    std::map<std::string, ov::conformance::InputInfo> updated_input_info = in_info_ref;
    if (is_update_required) {
        // align matched model names
        const auto& ref_model_ops = model_ref->get_ordered_ops();
        const auto& model_ops = model->get_ordered_ops();
        size_t ref_ordered_ops_size = ref_model_ops.size();
        size_t ordered_ops_size = model_ops.size();
        if (ref_ordered_ops_size != ordered_ops_size && matched_op.empty()) {
            throw std::runtime_error("Matched models can not be compared according different op numbers!");
        }
        for (size_t i = 0; i < ordered_ops_size; ++i) {
            auto model_op_name = model_ops[i]->get_friendly_name();
            if (!in_info.count(model_op_name)) {
                continue;
            }
            if (!matched_op.empty()) {
                if (!matched_op.count(model_op_name)) {
                    continue;
                }
            }
            auto model_ref_op_name = matched_op.empty() ? ref_model_ops[i]->get_friendly_name() : matched_op.at(model_op_name);

            const auto& in_info_item = in_info.at(model_op_name);
            const auto& ref_in_info_item = in_info_ref.at(model_ref_op_name);
            if (in_info_item.is_const != ref_in_info_item.is_const) {
                throw std::runtime_error("Impossible to update input info!!!");
            }
            updated_input_info[model_ref_op_name] = in_info_item;
        }
    }
    return updated_input_info;
}

}  // namespace util
}  // namespace ov