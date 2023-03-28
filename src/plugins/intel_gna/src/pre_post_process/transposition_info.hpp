// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <vector>

#include "openvino/core/model.hpp"

namespace ov {
namespace intel_gna {
namespace pre_post_processing {

struct TranspositionInfo {
    bool transpose;
    size_t num_transpose_rows;
    size_t num_transpose_columns;
};

using TranspositionInfoMap = std::map<std::string, std::vector<TranspositionInfo>>;

/*
 * Converts TranspositionInfo struct to ngraph function.
 * This method creates ngraph function with Transpose layer.
 */
std::shared_ptr<ov::Model> ToProcessModel(const TranspositionInfo& t_info);
/*
 * Converts several TranspositionInfo structures to ngraph function.
 * This method creates ngraph function with Gather layer.
 */
std::shared_ptr<ov::Model> ToProcessModel(const std::vector<TranspositionInfo>& transposes);

/*
 * Converts transposition maps to ngraph model, which will be ran on CPU as pre/post-processing step.
 * This conversion is needed to support the exported models version <= 2.8 (OV < 2023.0)
 * @return
 */
template <class T1, class T2>
void ConvertTransposeMapToModel(T1& transposes, T2& nodes) {
    for (auto&& node : nodes) {
        auto t_it = transposes.find(node.name);
        if (t_it != transposes.end() && !t_it->second.empty()) {
            node.pre_post_process_model = ToProcessModel(t_it->second);
        }
    }
};

static inline bool FoundPartToTranspose(const std::vector<TranspositionInfo>& transposes) {
    auto part_to_transpose =
        std::find_if(std::begin(transposes), std::end(transposes), [](const TranspositionInfo& t_info) {
            return t_info.transpose;
        });
    return part_to_transpose != std::end(transposes);
}

}  // namespace pre_post_processing
}  // namespace intel_gna
}  // namespace ov
