// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_eltwise_common.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <map>
#include <set>
#include <vector>

#include "nodes/executors/eltwise_config.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"

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

    std::set<std::vector<element::Type>> supported_precisions = get_supported_precisions(eltwise_data.front().algo);

    // for element-wise operations all inputs must to have the same precisions
    auto has_same_precision = [](const std::vector<element::Type>& precisions) {
        return std::all_of(precisions.begin(), precisions.end(), [&precisions](const element::Type precision) {
            return precision == precisions[0];
        });
    };

    assert(std::all_of(supported_precisions.begin(), supported_precisions.end(), has_same_precision));

    for (size_t i = 1; i < eltwise_data.size(); ++i) {
        std::set<std::vector<element::Type>> prcs = get_supported_precisions(eltwise_data[i].algo);
        std::set<std::vector<element::Type>> prcs_intersect = {};

        OPENVINO_ASSERT(std::all_of(prcs.begin(), prcs.end(), has_same_precision),
                        "for element-wise nodes all precisions have to be equal");

        set_intersection(supported_precisions, prcs, prcs_intersect);

        supported_precisions = prcs_intersect;
    }

    // To select the most suitable precision from inputs are mixed-precision
    // Preference is given to higher bitwidth, and for equal bitwidth, to real (floating point) types.
    const auto input_precision = [&] {
        auto selected_type = src_prc[0];  // Start with the first input's precision
        for (size_t i = 1; i < inputs_number; i++) {
            if (selected_type.bitwidth() > src_prc[i].bitwidth()) {
                continue;
            }
            // If bitwidths are equal and selected_type is real (floating point), keep it
            if (selected_type.bitwidth() == src_prc[i].bitwidth() && selected_type.is_real()) {
                continue;
            }
            // Otherwise, update selected_type to the current input's precision
            selected_type = src_prc[i];
        }
        return selected_type;
    }();

    for (const auto prc : exec_precisions_priority) {
        if (input_precision != prc) {
            continue;
        }
        if (std::any_of(supported_precisions.begin(),
                        supported_precisions.end(),
                        [&prc](const std::vector<element::Type>& precisions) {
                            // L56 has the check that all precisions are equal
                            // So only check the first one
                            return (precisions[0] == prc);
                        })) {
            exec_prc = prc;
            break;
        }
    }

    if (exec_prc == ov::element::dynamic) {
        // Fallback to fp32 if no other precision is found
        exec_prc = ov::element::f32;
    }

    return exec_prc;
}

void jit_uni_eltwise_kernel::operator()(const jit_eltwise_call_args_ptrs* const_args,
                                        const jit_eltwise_call_args_indexes* indexes) const {
    assert(ker_);
    ker_(const_args, indexes);
}
}  // namespace ov::intel_cpu
