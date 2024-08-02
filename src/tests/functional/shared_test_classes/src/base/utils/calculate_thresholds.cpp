// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/utils/calculate_thresholds.hpp"

#include "openvino/core/node.hpp"
#include "openvino/core/model.hpp"

#include "openvino/op/convert.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"

namespace ov {
namespace test {
namespace utils {

inline std::pair<double, double>
calculate_thresholds_by_whole_model(const std::shared_ptr<ov::Model>& model) {
    double max_abs_threshold = DISABLE_THRESHOLD, max_rel_threshold = DISABLE_THRESHOLD;

    // check all operations except convert to generate correct values
    for (const auto& op : model->get_ordered_ops()) {
        if (std::dynamic_pointer_cast<ov::op::v0::Convert>(op)) {
            continue;
        }
        // check the default threshold for operations
        const auto it = custom_op_thresholds.find(op->get_type_info());
        if (it != custom_op_thresholds.end()) {
            if (it->second.abs_threshold > max_abs_threshold) {
                max_abs_threshold = it->second.abs_threshold;
            }
            if (it->second.rel_threshold > max_rel_threshold) {
                max_rel_threshold = it->second.rel_threshold;
            }
        }
        for (const auto& in : op->inputs()) {
            const auto elem_type = in.get_element_type();
            const auto abs_value = ov::test::utils::tensor_comparation::calculate_default_abs_threshold(elem_type);
            if (abs_value > max_abs_threshold) {
                max_abs_threshold = abs_value;
            }
            const auto rel_value = ov::test::utils::tensor_comparation::calculate_default_rel_threshold(elem_type);
            if (rel_value > max_rel_threshold) {
                max_rel_threshold = rel_value;
            }
        }
    }
    return {max_abs_threshold, max_rel_threshold};
}

inline std::pair<double, double>
calculate_thresholds_by_model_results(const std::shared_ptr<ov::Model>& model,
                                      const std::shared_ptr<ov::Model>& ref_model,
                                      const ov::element::Type& inference_precision) {
    double max_abs_threshold = DISABLE_THRESHOLD, max_rel_threshold = DISABLE_THRESHOLD;

    const auto results = model->get_results();
    const auto ref_results = ref_model->get_results();
    if (results.size() != ref_results.size()) {
        throw std::runtime_error("Model and ref_model should have the same output number! Impossible to calculate default threshold!");
    }
    for (size_t out_idx = 0; out_idx < results.size(); ++out_idx) {
        const auto elem_type = results[out_idx]->get_element_type();
        const auto ref_elem_type = ref_results[out_idx]->get_element_type();
        const auto abs_value = ov::test::utils::tensor_comparation::calculate_default_abs_threshold(elem_type, ref_elem_type, inference_precision);
        if (abs_value > max_abs_threshold) {
            max_abs_threshold = abs_value;
        }
        const auto rel_value = ov::test::utils::tensor_comparation::calculate_default_rel_threshold(elem_type, ref_elem_type, inference_precision);
        if (rel_value > max_rel_threshold) {
            max_rel_threshold = rel_value;
        }
    }
    return { max_abs_threshold, max_rel_threshold };
}

std::pair<double, double>
calculate_thresholds_by_model(const std::shared_ptr<ov::Model>& model,
                              const std::shared_ptr<ov::Model>& ref_model,
                              const ov::element::Type& inference_precision) {
    double model_max_abs_threshold = DISABLE_THRESHOLD, model_max_rel_threshold = DISABLE_THRESHOLD,
           out_max_abs_threshold = DISABLE_THRESHOLD, out_max_rel_threshold = DISABLE_THRESHOLD;
    std::tie(model_max_abs_threshold, model_max_rel_threshold) = ov::test::utils::calculate_thresholds_by_whole_model(model);
    if (ref_model) {
        std::tie(out_max_abs_threshold, out_max_rel_threshold) =
            ov::test::utils::calculate_thresholds_by_model_results(model, ref_model, inference_precision);
    }
    return { std::max(model_max_abs_threshold, out_max_abs_threshold),
             std::max(model_max_rel_threshold, out_max_rel_threshold) };
}

} // namespace utils
} // namespace test
} // namespace ov
