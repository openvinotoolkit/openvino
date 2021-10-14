// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/op/tile.hpp>

namespace ov {
namespace op {
namespace v0 {
template <class T>
void inline set_dynamic_shape(T& shape){
    shape = T{};
}

template<>
void inline set_dynamic_shape(PartialShape& shape){
    shape = PartialShape::dynamic();
}

template <class T>
void shape_infer(Tile* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 2 && output_shapes.size() == 1);
    const auto& arg_shape = input_shapes[0];
    const auto& repeats_shape = input_shapes[1];
    auto& output_shape = output_shapes[0];
    using DimType = typename std::iterator_traits<typename T::iterator>::value_type;

    std::vector<DimType> repeats_value(repeats_shape);
    if (!repeats_value.empty() && arg_shape.rank().is_static()) {
        std::vector<DimType> data_shape(arg_shape);
        auto data_rank = data_shape.size();
        auto repeats_rank = repeats_value.size();
        auto output_rank = std::max(data_rank, repeats_rank);

        // expand data shape and repeats to output rank
        data_shape.insert(data_shape.begin(), output_rank - data_rank, 1);
        repeats_value.insert(repeats_value.begin(), output_rank - repeats_rank, 1);

        output_shape.resize(output_rank);
        for (size_t i = 0; i < output_rank; i++)
            output_shape[i] = data_shape[i] * repeats_value[i];

    } else {
        set_dynamic_shape(output_shape);
    }
}
}  // namespace v0
}  // namespace op
}  // namespace ov