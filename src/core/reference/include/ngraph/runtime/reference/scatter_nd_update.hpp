// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstring>
#include <numeric>

#include "ngraph/coordinate.hpp"
#include "ngraph/shape.hpp"
#include "utils/span.hpp"

namespace ngraph {
namespace runtime {
namespace reference {
template <typename dataType, typename indicesType>
void scatterNdUpdate(const dataType* const inputData,
                     const indicesType* const indices,
                     const dataType* const updates,
                     dataType* const outBuf,
                     const Shape& dataShape,
                     const Shape& indicesShape,
                     const Shape& updatesShape) {
    const auto update_chunk_shape = span(dataShape).drop_front(indicesShape.back());
    const auto update_el_number = shape_size(update_chunk_shape);

    std::memcpy(outBuf, inputData, sizeof(dataType) * shape_size(dataShape));

    const auto input_data_dim_pading = [&] {
        std::vector<size_t> padding(dataShape.size(), 1);
        for (size_t i = dataShape.size() - 1; i != 0; --i) {
            padding[i - 1] = padding[i] * dataShape[i];
        };
        return padding;
    }();

    const auto num_of_updates = shape_size(span(indicesShape).drop_back(1));
    for (size_t i = 0; i != num_of_updates; ++i) {
        const auto indices_coord = indices + i * indicesShape.back();
        const auto coord = span(indices_coord, indicesShape.back());
        const auto out_index = std::inner_product(begin(coord), end(coord), begin(input_data_dim_pading), uint64_t(0));

        const auto update_data = updates + i * update_el_number;
        const auto update_mem_size = update_el_number * sizeof(dataType);
        std::memcpy(outBuf + out_index, update_data, update_mem_size);
    }
}
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
