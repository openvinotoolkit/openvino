// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstring>
#include <numeric>

#include "add.hpp"
#include "and.hpp"
#include "maximum.hpp"
#include "minimum.hpp"
#include "multiply.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "or.hpp"
#include "subtract.hpp"
#include "utils/span.hpp"
#include "xor.hpp"

namespace ov {
namespace reference {
using Reduction = ov::op::v14::ScatterNDUpdate::Reduction;
template <typename T>
void reduction_functor_for(T* arg0, const T* arg1, size_t count, const Reduction reduction_type) {
    switch (reduction_type) {
    case Reduction::MAX:
        return maximum<T>(arg0, arg1, arg0, count);
    case Reduction::MIN:
        return minimum<T>(arg0, arg1, arg0, count);
    case Reduction::PROD:
        return multiply<T>(arg0, arg1, arg0, count);
    case Reduction::SUM:
        return add<T>(arg0, arg1, arg0, count);
    case Reduction::SUB:
        return subtract<T>(arg0, arg1, arg0, count);
    default:
        OPENVINO_THROW("No functor available for this type of reduction");
    }
}

template <>
void reduction_functor_for(char* arg0, const char* arg1, size_t count, const Reduction reduction_type) {
    switch (reduction_type) {
    case Reduction::MIN:
    case Reduction::PROD:
        return logical_and<char>(arg0, arg1, arg0, count);
    case Reduction::SUM:
    case Reduction::MAX:
        return logical_or<char>(arg0, arg1, arg0, count);
    case Reduction::SUB:
        return logical_xor<char>(arg0, arg1, arg0, count);
    default:
        OPENVINO_THROW("No functor available for this type of reduction");
    }
}

template <typename dataType, typename indicesType>
void scatterNdUpdate(const dataType* const inputData,
                     const indicesType* const indices,
                     const dataType* const updates,
                     dataType* const outBuf,
                     const Shape& dataShape,
                     const Shape& indicesShape,
                     const Shape& updatesShape,
                     const Reduction reduction_type = Reduction::NONE) {
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

    std::vector<indicesType> indicesCopy(indices, indices + shape_size(indicesShape));

    const auto num_of_updates = shape_size(span(indicesShape).drop_back(1));
    for (size_t i = 0; i != num_of_updates; ++i) {
        const auto indices_coord = indicesCopy.data() + i * indicesShape.back();
        const auto coord = span(indices_coord, indicesShape.back());

        // Negative value for indices means counting backwards from the end.
        int j = 0;
        for (auto& c : coord) {
            if (c < 0) {
                c += static_cast<indicesType>(dataShape[j]);
            }
            j++;
        }

        const auto out_index = std::inner_product(begin(coord), end(coord), begin(input_data_dim_pading), uint64_t(0));

        const auto update_data = updates + i * update_el_number;
        OPENVINO_ASSERT(out_index >= 0 && out_index + update_el_number <= shape_size(dataShape),
                        "Index is out of bounds");
        if (reduction_type == Reduction::NONE) {
            const auto update_mem_size = update_el_number * sizeof(dataType);
            std::memcpy(outBuf + out_index, update_data, update_mem_size);
        } else {
            reduction_functor_for<dataType>(outBuf + out_index, update_data, update_el_number, reduction_type);
        }
    }
}
}  // namespace reference
}  // namespace ov
