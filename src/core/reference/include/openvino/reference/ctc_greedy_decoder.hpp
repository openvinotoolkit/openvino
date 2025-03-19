// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <limits>
#include <vector>

#include "openvino/reference/utils/coordinate_index.hpp"

namespace ov {
namespace reference {
template <typename T>
void ctc_greedy_decoder(const T* data,
                        const T* sequence_masks,
                        T* out,
                        const Shape& data_shape,
                        const Shape& sequence_masks_shape,
                        const Shape& out_shape,
                        const bool ctc_merge_repeated) {
    const auto max_seq_len = data_shape[0];
    const auto batch_size = data_shape[1];
    const auto class_count = data_shape[2];
    const uint64_t blank_index = class_count - 1;

    // final sequences don't have to fill the whole output, elements that don't store
    // information are set to -1

    std::vector<T> tmp_out(shape_size(out_shape));
    std::fill(tmp_out.begin(), tmp_out.end(), static_cast<T>(-1));

    for (unsigned int batch_ind = 0; batch_ind < batch_size; batch_ind++) {
        T previous_class_index = static_cast<T>(-1);
        auto out_index = coordinate_index({batch_ind, 0, 0, 0}, out_shape);
        for (unsigned int seq_ind = 0; seq_ind < max_seq_len; seq_ind++) {
            auto data_index = coordinate_index({seq_ind, batch_ind, 0}, data_shape);
            auto mask_index = coordinate_index({seq_ind, batch_ind}, sequence_masks_shape);

            if (sequence_masks[mask_index] == T{0}) {
                break;
            }

            auto class_index = data + data_index;
            auto class_max_element = std::max_element(class_index, class_index + class_count);
            T max_class_ind = static_cast<T>(std::distance(class_index, class_max_element));

            if (!(previous_class_index == max_class_ind && ctc_merge_repeated) &&
                static_cast<uint64_t>(max_class_ind) < blank_index) {
                tmp_out[out_index++] = max_class_ind;
            }
            previous_class_index = max_class_ind;
        }
    }
    std::copy(tmp_out.begin(), tmp_out.end(), out);
}
}  // namespace reference
}  // namespace ov
