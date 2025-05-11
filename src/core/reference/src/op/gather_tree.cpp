// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/gather_tree.hpp"

#include <stdio.h>

#include <cmath>
#include <numeric>

#include "openvino/core/except.hpp"
#include "openvino/reference/utils/coordinate_transform.hpp"

using namespace ov;

static size_t _asIndex(const char* source, const element::Type& element_type) {
    // According to the GatherTree op specification only I32 and FP32 precisions are supported.
    switch (element_type) {
    case element::Type_t::f16: {
        ov::float16 tmpBuff = 0.f;
        memcpy(&tmpBuff, source, sizeof(ov::float16));
        return static_cast<size_t>(tmpBuff);
    }
    case element::Type_t::f32: {
        float tmpBuff = 0.f;
        memcpy(&tmpBuff, source, sizeof(float));
        return static_cast<size_t>(tmpBuff);
    }
    case element::Type_t::i32: {
        int32_t tmpBuff = 0;
        memcpy(&tmpBuff, source, sizeof(int32_t));
        return static_cast<size_t>(tmpBuff);
    }
    default: {
        OPENVINO_THROW("Unsupported input data type: ", element_type.to_string());
    }
    }
}

// This is an implementation of the algorithm from the tensorflow 1.5 sources.
void reference::gather_tree(const char* step_ids,
                            const char* parent_ids,
                            const char* max_seq_len,
                            const char* end_token,
                            char* out,
                            const Shape& step_ids_shape,
                            const Shape& parent_ids_shape,
                            const Shape& max_seq_len_shape,
                            const Shape& end_token_shape,
                            const element::Type& element_type) {
    if (step_ids_shape != parent_ids_shape) {
        OPENVINO_THROW("step_ids shape and parent_ids shape must be the same");
    }
    if (step_ids_shape.size() != 3) {
        OPENVINO_THROW("step_ids must be a 3-tensor");
    }
    if (!is_vector(max_seq_len_shape)) {
        OPENVINO_THROW("max_seq_len must be a vector");
    }
    if (!is_scalar(end_token_shape)) {
        OPENVINO_THROW("end_token must be a scalar");
    }

    const size_t max_time = step_ids_shape.at(0);
    const size_t batch_size = step_ids_shape.at(1);
    const size_t beam_width = step_ids_shape.at(2);

    const size_t elem_size = element_type.size();

    if (max_seq_len_shape.front() != batch_size) {
        OPENVINO_THROW("max_seq_len must have size of BATCH_SIZE");
    }

    const auto in_strides = row_major_strides(step_ids_shape);
    CoordinateTransformBasic cordinate_transform(step_ids_shape);

    for (const auto& coord : cordinate_transform) {
        const auto out_idx = std::inner_product(coord.begin(), coord.end(), in_strides.begin(), uint64_t(0));
        memcpy(out + out_idx * elem_size, end_token, elem_size);
    }

    for (size_t batch = 0; batch < batch_size; ++batch) {
        for (size_t beam = 0; beam < beam_width; ++beam) {
            const size_t max_seq_in_beam = std::min(max_time, _asIndex(max_seq_len + batch * elem_size, element_type));

            if (max_seq_in_beam == 0) {
                continue;
            }

            const auto coord = Coordinate({max_seq_in_beam - 1, batch, beam});
            const auto offset =
                std::inner_product(coord.begin(), coord.end(), in_strides.begin(), uint64_t(0)) * elem_size;
            memcpy(out + offset, step_ids + offset, elem_size);

            size_t parent = _asIndex(parent_ids + offset, element_type);

            for (size_t level = max_seq_in_beam - 1; level-- > 0;) {
                const auto coord_beam = Coordinate({level, batch, beam});
                const auto out_idx =
                    std::inner_product(coord_beam.begin(), coord_beam.end(), in_strides.begin(), uint64_t(0));

                const auto coord_parent = Coordinate({level, batch, parent});
                const auto step_ids_idx =
                    std::inner_product(coord_parent.begin(), coord_parent.end(), in_strides.begin(), uint64_t(0));

                memcpy(out + out_idx * elem_size, step_ids + step_ids_idx * elem_size, elem_size);

                parent = _asIndex(parent_ids + step_ids_idx * elem_size, element_type);
            }

            bool finished = false;
            for (size_t time = 0; time < max_seq_in_beam; ++time) {
                const auto out_coord = Coordinate({time, batch, beam});
                const auto out_idx =
                    std::inner_product(out_coord.begin(), out_coord.end(), in_strides.begin(), uint64_t(0));
                if (finished) {
                    memcpy(out + out_idx * elem_size, end_token, elem_size);
                } else if (_asIndex(out + out_idx * elem_size, element_type) == _asIndex(end_token, element_type)) {
                    finished = true;
                }
            }
        }
    }
}
