// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/tensorflow_lite/sparsity_info.hpp"

#include <limits>

#include "schema_generated.h"

bool ov::frontend::tensorflow_lite::SparsityInfo::is_copyable() const {
    return false;
}

// Overflow-safe multiplication for size calculations.
// Throws on overflow to prevent heap buffer allocation with a truncated size,
// which would lead to out-of-bounds writes during sparse data deserialization.
static size_t safe_multiply(size_t a, size_t b) {
    FRONT_END_GENERAL_CHECK(b == 0 || a <= std::numeric_limits<size_t>::max() / b,
                            "Integer overflow in sparse tensor size calculation: ",
                            a,
                            " * ",
                            b);
    return a * b;
}

// Unpacks sparse data. Supports only case when sparse tensor has only one sparse dimension [DENSE, ..., DENSE, SPARSE]
// TensorFlow Lite uses a specific format for storing sparse data (TACO):
// It uses three 1D arrays/1D tensors/vectors:
// values - stored in model's buffers
// segments/positions - list of row's start positions
// indices - list of value indexes in a row (size is equal to values)
// Algorithm is next, and it should be easily modified later for 2D sparse matrices (that's why it uses a 2D approach):
// 1. Get a first segment position, set idx = 0
// In a cycle
// 2. Get next segment position
// 3. Get an element_count by a difference between current segment position and last_segment position.
//    If diff between last and current segment is 0 it means an empty row (contains only default values).
//    In cycle for each element in a segment:
//    4. Calculate a row_offset using idx-th index by indices and element_size (fp16/fp32/etc...)
//    5. Put an idx-th value to a found target dest + row_offset
//    6. Move to a next row
template <typename T, typename U>
static void read_sparse_data(uint8_t* dest,
                             const size_t total_size,
                             const uint8_t* values,
                             const size_t values_size,
                             const size_t row_size,
                             const size_t element_size,
                             const size_t sparse_dim_size,
                             const size_t num_dense_rows,
                             const ::flatbuffers::Vector<T>* indices,
                             const ::flatbuffers::Vector<U>* segments) {
    FRONT_END_GENERAL_CHECK(segments && segments->size() >= 2,
                            "Sparse segments vector must have at least 2 elements, got ",
                            (segments ? segments->size() : 0));
    FRONT_END_GENERAL_CHECK(indices, "Sparse indices vector is null");
    FRONT_END_GENERAL_CHECK(static_cast<size_t>(segments->size()) == num_dense_rows + 1,
                            "Sparse segments vector size ",
                            segments->size(),
                            " does not match expected number of dense rows + 1 (",
                            num_dense_rows + 1,
                            ")");

    U last_segment = *segments->begin();
    FRONT_END_GENERAL_CHECK(static_cast<int64_t>(last_segment) >= 0, "Sparse segment value must be non-negative");
    size_t idx = 0;
    size_t dest_row_offset = 0;
    for (auto segment = segments->begin() + 1; segment != segments->end(); last_segment = *segment, ++segment) {
        FRONT_END_GENERAL_CHECK(dest_row_offset < total_size,
                                "Dense row offset ",
                                dest_row_offset,
                                " exceeds buffer size ",
                                total_size);
        FRONT_END_GENERAL_CHECK(static_cast<int64_t>(*segment) >= 0, "Sparse segment value must be non-negative");
        FRONT_END_GENERAL_CHECK(*segment >= last_segment, "Sparse segments must be monotonically non-decreasing");
        size_t element_count = static_cast<size_t>(*segment - last_segment);
        for (size_t i = 0; i < element_count; ++i, ++idx) {
            FRONT_END_GENERAL_CHECK(idx < static_cast<size_t>(indices->size()),
                                    "Sparse index position ",
                                    idx,
                                    " exceeds indices vector size ",
                                    indices->size());
            auto index_value = (*indices)[static_cast<flatbuffers::uoffset_t>(idx)];
            FRONT_END_GENERAL_CHECK(static_cast<int64_t>(index_value) >= 0,
                                    "Sparse index value must be non-negative, got ",
                                    static_cast<int64_t>(index_value));
            FRONT_END_GENERAL_CHECK(static_cast<size_t>(index_value) < sparse_dim_size,
                                    "Sparse index value ",
                                    static_cast<int64_t>(index_value),
                                    " is out of bounds for dimension size ",
                                    sparse_dim_size);
            auto row_offset = safe_multiply(static_cast<size_t>(index_value), element_size);
            auto value_offset = safe_multiply(idx, element_size);
            FRONT_END_GENERAL_CHECK(
                row_offset <= total_size - dest_row_offset && element_size <= total_size - dest_row_offset - row_offset,
                "Sparse write at offset ",
                dest_row_offset + row_offset,
                " would exceed dense buffer of size ",
                total_size);
            FRONT_END_GENERAL_CHECK(
                values_size == 0 || (values_size >= element_size && value_offset <= values_size - element_size),
                "Sparse read at offset ",
                value_offset,
                " would exceed values buffer of size ",
                values_size);
            memcpy(dest + dest_row_offset + row_offset, values + value_offset, element_size);
        }
        FRONT_END_GENERAL_CHECK(dest_row_offset <= std::numeric_limits<size_t>::max() - row_size,
                                "Overflow in dense row offset accumulation");
        dest_row_offset += row_size;
    }
}

void* ov::frontend::tensorflow_lite::SparsityInfo::densify() {
    FRONT_END_GENERAL_CHECK(m_values, "Values are not found");
    size_t sparse_idx = 0;
    for (; sparse_idx < m_dim_format.size(); ++sparse_idx) {
        if (m_dim_format[sparse_idx] == ::tflite::DimensionType_SPARSE_CSR)
            break;
    }
    FRONT_END_GENERAL_CHECK(sparse_idx < m_dim_format.size(), "Sparse dimension isn't found for sparse tensor");
    FRONT_END_GENERAL_CHECK(sparse_idx == (m_dim_format.size() - 1),
                            "Supports only sparse tensor with sparse dimension as a last dimension");

    size_t total_size = m_target_type.size(),  // Size of data in bytes
        row_size = total_size;                 // Size of data row in bytes
    size_t sparse_dim_size = 0;
    for (size_t dim = 0; dim < m_shape.size(); ++dim) {
        FRONT_END_GENERAL_CHECK(m_shape[dim] >= 0,
                                "Sparse tensor shape dimension must be non-negative, got ",
                                m_shape[dim]);
        total_size = safe_multiply(total_size, static_cast<size_t>(m_shape[dim]));
        switch (m_dim_format[dim]) {
        case ::tflite::DimensionType_DENSE:
            break;
        case ::tflite::DimensionType_SPARSE_CSR:
            row_size = safe_multiply(row_size, static_cast<size_t>(m_shape[dim]));
            sparse_dim_size = static_cast<size_t>(m_shape[dim]);
            break;
        default:
            FRONT_END_THROW("Unsupported dimension type found");
            break;
        }
    }
    FRONT_END_GENERAL_CHECK(total_size > 0, "Wrong sparse segment size found");
    FRONT_END_GENERAL_CHECK(sparse_dim_size > 0, "Sparse dimension size must be positive");
    const size_t num_dense_rows = total_size / row_size;

    m_data.resize(total_size);
    memset(m_data.data(), 0, total_size);
    switch (m_data_desc[sparse_idx].indices_type) {
    case ::tflite::SparseIndexVector_Uint8Vector:
        switch (m_data_desc[sparse_idx].segments_type) {
        case ::tflite::SparseIndexVector_Uint8Vector:
            read_sparse_data(m_data.data(),
                             total_size,
                             m_values,
                             m_values_size,
                             row_size,
                             m_target_type.size(),
                             sparse_dim_size,
                             num_dense_rows,
                             static_cast<const ::tflite::Uint8Vector*>(m_data_desc[sparse_idx].indices)->values(),
                             static_cast<const ::tflite::Uint8Vector*>(m_data_desc[sparse_idx].segments)->values());
            break;
        case ::tflite::SparseIndexVector_Uint16Vector:
            read_sparse_data(m_data.data(),
                             total_size,
                             m_values,
                             m_values_size,
                             row_size,
                             m_target_type.size(),
                             sparse_dim_size,
                             num_dense_rows,
                             static_cast<const ::tflite::Uint8Vector*>(m_data_desc[sparse_idx].indices)->values(),
                             static_cast<const ::tflite::Uint16Vector*>(m_data_desc[sparse_idx].segments)->values());
            break;
        case ::tflite::SparseIndexVector_Int32Vector:
            read_sparse_data(m_data.data(),
                             total_size,
                             m_values,
                             m_values_size,
                             row_size,
                             m_target_type.size(),
                             sparse_dim_size,
                             num_dense_rows,
                             static_cast<const ::tflite::Uint8Vector*>(m_data_desc[sparse_idx].indices)->values(),
                             static_cast<const ::tflite::Int32Vector*>(m_data_desc[sparse_idx].segments)->values());
            break;
        default:
            FRONT_END_THROW("Unsupported vector type");
            break;
        }
        break;
    case ::tflite::SparseIndexVector_Uint16Vector:
        switch (m_data_desc[sparse_idx].segments_type) {
        case ::tflite::SparseIndexVector_Uint8Vector:
            read_sparse_data(m_data.data(),
                             total_size,
                             m_values,
                             m_values_size,
                             row_size,
                             m_target_type.size(),
                             sparse_dim_size,
                             num_dense_rows,
                             static_cast<const ::tflite::Uint16Vector*>(m_data_desc[sparse_idx].indices)->values(),
                             static_cast<const ::tflite::Uint8Vector*>(m_data_desc[sparse_idx].segments)->values());
            break;
        case ::tflite::SparseIndexVector_Uint16Vector:
            read_sparse_data(m_data.data(),
                             total_size,
                             m_values,
                             m_values_size,
                             row_size,
                             m_target_type.size(),
                             sparse_dim_size,
                             num_dense_rows,
                             static_cast<const ::tflite::Uint16Vector*>(m_data_desc[sparse_idx].indices)->values(),
                             static_cast<const ::tflite::Uint16Vector*>(m_data_desc[sparse_idx].segments)->values());
            break;
        case ::tflite::SparseIndexVector_Int32Vector:
            read_sparse_data(m_data.data(),
                             total_size,
                             m_values,
                             m_values_size,
                             row_size,
                             m_target_type.size(),
                             sparse_dim_size,
                             num_dense_rows,
                             static_cast<const ::tflite::Uint16Vector*>(m_data_desc[sparse_idx].indices)->values(),
                             static_cast<const ::tflite::Int32Vector*>(m_data_desc[sparse_idx].segments)->values());
            break;
        default:
            FRONT_END_THROW("Unsupported vector type");
            break;
        }
        break;
    case ::tflite::SparseIndexVector_Int32Vector:
        switch (m_data_desc[sparse_idx].segments_type) {
        case ::tflite::SparseIndexVector_Uint8Vector:
            read_sparse_data(m_data.data(),
                             total_size,
                             m_values,
                             m_values_size,
                             row_size,
                             m_target_type.size(),
                             sparse_dim_size,
                             num_dense_rows,
                             static_cast<const ::tflite::Int32Vector*>(m_data_desc[sparse_idx].indices)->values(),
                             static_cast<const ::tflite::Uint8Vector*>(m_data_desc[sparse_idx].segments)->values());
            break;
        case ::tflite::SparseIndexVector_Uint16Vector:
            read_sparse_data(m_data.data(),
                             total_size,
                             m_values,
                             m_values_size,
                             row_size,
                             m_target_type.size(),
                             sparse_dim_size,
                             num_dense_rows,
                             static_cast<const ::tflite::Int32Vector*>(m_data_desc[sparse_idx].indices)->values(),
                             static_cast<const ::tflite::Uint16Vector*>(m_data_desc[sparse_idx].segments)->values());
            break;
        case ::tflite::SparseIndexVector_Int32Vector:
            read_sparse_data(m_data.data(),
                             total_size,
                             m_values,
                             m_values_size,
                             row_size,
                             m_target_type.size(),
                             sparse_dim_size,
                             num_dense_rows,
                             static_cast<const ::tflite::Int32Vector*>(m_data_desc[sparse_idx].indices)->values(),
                             static_cast<const ::tflite::Int32Vector*>(m_data_desc[sparse_idx].segments)->values());
            break;
        default:
            FRONT_END_THROW("Unsupported vector type");
            break;
        }
        break;
    default:
        FRONT_END_THROW("Unsupported vector type");
        break;
    }
    return m_data.data();
}
