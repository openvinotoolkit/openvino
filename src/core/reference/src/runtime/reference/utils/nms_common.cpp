// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/runtime/reference/utils/nms_common.hpp"

#include <algorithm>
#include <cstring>
#include <numeric>

#include "ngraph/check.hpp"

namespace ngraph {
namespace runtime {
namespace reference {
namespace nms_common {
void nms_common_postprocessing(void* prois,
                               void* pscores,
                               void* pselected_num,
                               const ngraph::element::Type& output_type,
                               const std::vector<float>& selected_outputs,
                               const std::vector<int64_t>& selected_indices,
                               const std::vector<int64_t>& valid_outputs,
                               const ngraph::element::Type& selected_outputs_type) {
    int64_t total_num = std::accumulate(valid_outputs.begin(), valid_outputs.end(), int64_t(0));

    switch (selected_outputs_type) {
    case element::Type_t::bf16: {
        bfloat16* ptr = static_cast<bfloat16*>(prois);
        for (auto i = 0; i < total_num * 6; ++i) {
            ptr[i] = bfloat16(selected_outputs[i]);
        }
    } break;
    case element::Type_t::f16: {
        float16* ptr = static_cast<float16*>(prois);
        for (auto i = 0; i < total_num * 6; ++i) {
            ptr[i] = float16(selected_outputs[i]);
        }
    } break;
    case element::Type_t::f32: {
        float* ptr = static_cast<float*>(prois);
        memcpy(ptr, selected_outputs.data(), total_num * sizeof(float) * 6);
    } break;
    default:
        NGRAPH_UNREACHABLE("unsupported element type, should be [bf16, f16, f32]");
    }

    if (pscores) {
        if (output_type == ngraph::element::i64) {
            int64_t* indices_ptr = static_cast<int64_t*>(pscores);
            memcpy(indices_ptr, selected_indices.data(), total_num * sizeof(int64_t));
        } else {
            int32_t* indices_ptr = static_cast<int32_t*>(pscores);
            for (int64_t i = 0; i < total_num; ++i) {
                indices_ptr[i] = static_cast<int32_t>(selected_indices[i]);
            }
        }
    }

    if (pselected_num) {
        if (output_type == ngraph::element::i64) {
            int64_t* valid_outputs_ptr = static_cast<int64_t*>(pselected_num);
            std::copy(valid_outputs.begin(), valid_outputs.end(), valid_outputs_ptr);
        } else {
            int32_t* valid_outputs_ptr = static_cast<int32_t*>(pselected_num);
            for (size_t i = 0; i < valid_outputs.size(); ++i) {
                valid_outputs_ptr[i] = static_cast<int32_t>(valid_outputs[i]);
            }
        }
    }
}
}  // namespace nms_common
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
