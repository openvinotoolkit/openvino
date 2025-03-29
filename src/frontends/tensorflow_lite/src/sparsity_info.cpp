// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/tensorflow_lite/sparsity_info.hpp"

#include "schema_generated.h"

bool ov::frontend::tensorflow_lite::SparsityInfo::is_copyable() const {
    return false;
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
                             uint8_t* dest_end,
                             const uint8_t* values,
                             const size_t row_size,
                             const size_t element_size,
                             const ::flatbuffers::Vector<T>* indices,
                             const ::flatbuffers::Vector<U>* segments) {
    U last_segment = *segments->begin();
    size_t idx = 0;
    for (auto segment = segments->begin() + 1; segment != segments->end(); last_segment = *segment, ++segment) {
        FRONT_END_GENERAL_CHECK(dest < dest_end, "Dense data is out of bounds");
        size_t element_count = *segment - last_segment;
        for (size_t i = 0; i < element_count; ++i, ++idx) {
            auto row_offset = (*indices)[static_cast<flatbuffers::uoffset_t>(idx)] * element_size;
            auto value_offset = idx * element_size;
            memcpy(static_cast<uint8_t*>(static_cast<void*>(dest)) + row_offset, values + value_offset, element_size);
        }
        dest += row_size;
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
    for (size_t dim = 0; dim < m_shape.size(); ++dim) {
        total_size *= m_shape[dim];
        switch (m_dim_format[dim]) {
        case ::tflite::DimensionType_DENSE:
            break;
        case ::tflite::DimensionType_SPARSE_CSR:
            row_size *= m_shape[dim];
            break;
        default:
            FRONT_END_THROW("Unsupported dimension type found");
            break;
        }
    }
    FRONT_END_GENERAL_CHECK(total_size > 0, "Wrong sparse segment size found");

    m_data.resize(total_size);
    memset(m_data.data(), 0, total_size);
    switch (m_data_desc[sparse_idx].indices_type) {
    case ::tflite::SparseIndexVector_Uint8Vector:
        switch (m_data_desc[sparse_idx].segments_type) {
        case ::tflite::SparseIndexVector_Uint8Vector:
            read_sparse_data(m_data.data(),
                             m_data.data() + total_size,
                             m_values,
                             row_size,
                             m_target_type.size(),
                             static_cast<const ::tflite::Uint8Vector*>(m_data_desc[sparse_idx].indices)->values(),
                             static_cast<const ::tflite::Uint8Vector*>(m_data_desc[sparse_idx].segments)->values());
            break;
        case ::tflite::SparseIndexVector_Uint16Vector:
            read_sparse_data(m_data.data(),
                             m_data.data() + total_size,
                             m_values,
                             row_size,
                             m_target_type.size(),
                             static_cast<const ::tflite::Uint8Vector*>(m_data_desc[sparse_idx].indices)->values(),
                             static_cast<const ::tflite::Uint16Vector*>(m_data_desc[sparse_idx].segments)->values());
            break;
        case ::tflite::SparseIndexVector_Int32Vector:
            read_sparse_data(m_data.data(),
                             m_data.data() + total_size,
                             m_values,
                             row_size,
                             m_target_type.size(),
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
                             m_data.data() + total_size,
                             m_values,
                             row_size,
                             m_target_type.size(),
                             static_cast<const ::tflite::Uint16Vector*>(m_data_desc[sparse_idx].indices)->values(),
                             static_cast<const ::tflite::Uint8Vector*>(m_data_desc[sparse_idx].segments)->values());
            break;
        case ::tflite::SparseIndexVector_Uint16Vector:
            read_sparse_data(m_data.data(),
                             m_data.data() + total_size,
                             m_values,
                             row_size,
                             m_target_type.size(),
                             static_cast<const ::tflite::Uint16Vector*>(m_data_desc[sparse_idx].indices)->values(),
                             static_cast<const ::tflite::Uint16Vector*>(m_data_desc[sparse_idx].segments)->values());
            break;
        case ::tflite::SparseIndexVector_Int32Vector:
            read_sparse_data(m_data.data(),
                             m_data.data() + total_size,
                             m_values,
                             row_size,
                             m_target_type.size(),
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
                             m_data.data() + total_size,
                             m_values,
                             row_size,
                             m_target_type.size(),
                             static_cast<const ::tflite::Int32Vector*>(m_data_desc[sparse_idx].indices)->values(),
                             static_cast<const ::tflite::Uint8Vector*>(m_data_desc[sparse_idx].segments)->values());
            break;
        case ::tflite::SparseIndexVector_Uint16Vector:
            read_sparse_data(m_data.data(),
                             m_data.data() + total_size,
                             m_values,
                             row_size,
                             m_target_type.size(),
                             static_cast<const ::tflite::Int32Vector*>(m_data_desc[sparse_idx].indices)->values(),
                             static_cast<const ::tflite::Uint16Vector*>(m_data_desc[sparse_idx].segments)->values());
            break;
        case ::tflite::SparseIndexVector_Int32Vector:
            read_sparse_data(m_data.data(),
                             m_data.data() + total_size,
                             m_values,
                             row_size,
                             m_target_type.size(),
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
