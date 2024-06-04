// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstring>
#include <numeric>

#include "openvino/core/shape.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "openvino/reference/add.hpp"
#include "openvino/reference/and.hpp"
#include "openvino/reference/maximum.hpp"
#include "openvino/reference/minimum.hpp"
#include "openvino/reference/multiply.hpp"
#include "openvino/reference/or.hpp"
#include "openvino/reference/subtract.hpp"
#include "openvino/reference/xor.hpp"
#include "utils/span.hpp"

namespace ov {
namespace reference {
namespace scatter_nd_update {
template <typename T>
using reduction_function = T (*)(const T, const T);

template <typename T,
          typename std::enable_if<!std::is_same<typename std::decay<T>::type, char>::value>::type* = nullptr>
reduction_function<T> reduction_functor_for(const ov::op::v15::ScatterNDUpdate::Reduction reduction_type) {
    using U = typename std::decay<T>::type;
    switch (reduction_type) {
    case ov::op::v15::ScatterNDUpdate::Reduction::MAX:
        return func::max<U>;
    case ov::op::v15::ScatterNDUpdate::Reduction::MIN:
        return func::min<U>;
    case ov::op::v15::ScatterNDUpdate::Reduction::PROD:
        return func::multiply<U>;
    case ov::op::v15::ScatterNDUpdate::Reduction::SUM:
        return func::add<U>;
    case ov::op::v15::ScatterNDUpdate::Reduction::SUB:
        return func::subtract<U>;
    case ov::op::v15::ScatterNDUpdate::Reduction::NONE:
    default:
        return nullptr;
    }
}

template <typename T, typename std::enable_if<std::is_same<typename std::decay<T>::type, char>::value>::type* = nullptr>
reduction_function<T> reduction_functor_for(const ov::op::v15::ScatterNDUpdate::Reduction reduction_type) {
    using U = typename std::decay<T>::type;
    switch (reduction_type) {
    case ov::op::v15::ScatterNDUpdate::Reduction::MIN:
    case ov::op::v15::ScatterNDUpdate::Reduction::PROD:
        return func::logical_and<U>;
    case ov::op::v15::ScatterNDUpdate::Reduction::SUM:
    case ov::op::v15::ScatterNDUpdate::Reduction::MAX:
        return func::logical_or<U>;
    case ov::op::v15::ScatterNDUpdate::Reduction::SUB:
        return func::logical_xor<U>;
    case ov::op::v15::ScatterNDUpdate::Reduction::NONE:
    default:
        return nullptr;
    }
}
}  // namespace scatter_nd_update
template <typename dataType, typename indicesType>
void scatterNdUpdate(
    const dataType* const inputData,
    const indicesType* const indices,
    const dataType* const updates,
    dataType* const outBuf,
    const Shape& dataShape,
    const Shape& indicesShape,
    const Shape& updatesShape,
    const ov::op::v15::ScatterNDUpdate::Reduction reduction_type = ov::op::v15::ScatterNDUpdate::Reduction::NONE) {
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
    const auto reduction = scatter_nd_update::reduction_functor_for<dataType>(reduction_type);
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
        if (reduction) {
            std::transform(outBuf + out_index,
                           outBuf + out_index + update_el_number,
                           update_data,
                           outBuf + out_index,
                           reduction);
        } else {
            std::memcpy(outBuf + out_index, update_data, update_el_number * sizeof(dataType));
        }
    }
}
}  // namespace reference
}  // namespace ov
