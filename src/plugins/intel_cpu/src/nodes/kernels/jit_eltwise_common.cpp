// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_eltwise_common.hpp"

namespace ov::intel_cpu {

static void set_intersection(const std::set<std::vector<element::Type>>& precisions1,
                             const std::set<std::vector<element::Type>>& precisions2,
                             std::set<std::vector<element::Type>>& intersection) {
    std::map<element::Type, size_t> intersection_types;

    for (const auto& it1_precisions : precisions1) {
        for (const auto& it2 : precisions2) {
            // all element types are equal
            if (it1_precisions[0] == it2[0]) {
                // first precisions size is used
                intersection_types.emplace(it1_precisions[0], it1_precisions.size());
            }
        }
    }

    for (auto& intersection_type : intersection_types) {
        intersection.insert(std::vector<element::Type>(intersection_type.second, intersection_type.first));
    }
}

ov::element::Type eltwise_precision_helper::get_precision(const size_t inputs_number,
                                                          const ov::element::Type (&src_prc)[MAX_ELTWISE_INPUTS],
                                                          const std::vector<EltwiseData>& eltwise_data,
                                                          const std::vector<element::Type>& exec_precisions_priority) {
    ov::element::Type exec_prc = ov::element::dynamic;

    std::set<std::vector<element::Type>> supported_precision_intersection =
        get_supported_precisions(eltwise_data.front().algo);

    // for element-wise operations all inputs must to have the same precisions
    auto has_same_precision = [](const std::vector<element::Type>& precisions) {
        return std::all_of(precisions.begin(), precisions.end(), [&precisions](const element::Type precision) {
            return precision == precisions[0];
        });
    };

    assert(std::all_of(supported_precision_intersection.begin(),
                       supported_precision_intersection.end(),
                       has_same_precision));

    for (size_t i = 1; i < eltwise_data.size(); ++i) {
        std::set<std::vector<element::Type>> prcs = get_supported_precisions(eltwise_data[i].algo);
        std::set<std::vector<element::Type>> prcs_intersect = {};

        OPENVINO_ASSERT(std::all_of(prcs.begin(), prcs.end(), has_same_precision),
                        "for element-wise nodes all precisions have to be equal");

        set_intersection(supported_precision_intersection, prcs, prcs_intersect);

        supported_precision_intersection = prcs_intersect;
    }

    for (const auto prc : exec_precisions_priority) {
        if (std::any_of(supported_precision_intersection.begin(),
                        supported_precision_intersection.end(),
                        [&prc, &src_prc](const std::vector<element::Type>& precisions) {
                            return (std::find(precisions.begin(), precisions.end(), prc) != precisions.end()) &&
                                   (src_prc[0] == prc);
                        })) {
            exec_prc = prc;
            break;
        }
    }

    for (size_t i = 0; i < inputs_number; i++) {
        if (src_prc[i] != exec_prc) {
            exec_prc = ov::element::f32;
            break;
        }
    }

    if (exec_prc == ov::element::dynamic) {
        OPENVINO_THROW("Eltwise jitter failed to specify execution precision for Eltwise node");
    }

    return exec_prc;
}

}  // namespace ov::intel_cpu
