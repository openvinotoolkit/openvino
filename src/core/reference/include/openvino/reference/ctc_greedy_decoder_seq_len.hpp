// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <limits>
#include <vector>

#include "openvino/reference/utils/coordinate_transform.hpp"
namespace ov {
namespace reference {
template <typename TF, typename TI, typename TCI, typename TSL>
void ctc_greedy_decoder_seq_len(const TF* data,
                                const TI* sequence_length,
                                const TI* blank_index,
                                TCI* out1,
                                TSL* out2,
                                const Shape& data_shape,
                                const Shape& out_shape,
                                const bool ctc_merge_repeated) {
    const auto batch_size = data_shape[0];
    const auto seq_len_max = data_shape[1];
    const auto class_count = data_shape[2];
    std::fill_n(out1, shape_size(out_shape), TCI(-1));

    for (std::size_t batch_ind = 0; batch_ind < batch_size; ++batch_ind) {
        TI previous_class_index = static_cast<TI>(-1);
        auto out_index = batch_ind * seq_len_max;
        auto seq_len = static_cast<std::size_t>(sequence_length[batch_ind]);
        for (std::size_t seq_ind = 0; seq_ind < seq_len; seq_ind++) {
            auto data_index = batch_ind * seq_len_max * class_count + seq_ind * class_count;
            auto class_index = data + data_index;
            auto class_max_element = std::max_element(class_index, class_index + class_count);
            const auto max_class_ind = std::distance(class_index, class_max_element);
            if (max_class_ind != blank_index[0] && !(ctc_merge_repeated && previous_class_index == max_class_ind)) {
                out1[out_index++] = static_cast<TCI>(max_class_ind);
            }
            previous_class_index = static_cast<TI>(max_class_ind);
        }
        out2[batch_ind] = static_cast<TSL>(out_index - batch_ind * seq_len_max);
    }
}
}  // namespace reference
}  // namespace ov
