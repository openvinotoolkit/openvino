// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sparsity_info.hpp"

#include "schema_generated.h"

bool ov::frontend::tensorflow_lite::SparsityInfo::is_copyable() const {
    return false;
}

// Unpacks sparse data. Supports only case when sparse tensor has only one sparse dimension [DENSE, ..., DENSE, SPARSE]
template <typename T, typename U>
static void read_sparse_data(uint8_t* dest,
                             uint8_t* dest_end,
                             const size_t row_size,
                             const ::flatbuffers::Vector<T>* indices,
                             const ::flatbuffers::Vector<U>* segments) {
    uint8_t* data = dest - row_size;  // row size will be increased at first step
    T last_idx = ~static_cast<T>(0);
    for (auto idx = indices->begin(); idx != indices->end(); ++idx) {
        if (*idx <= last_idx) {
            data += row_size;
        }
        FRONT_END_GENERAL_CHECK(data + *idx < dest_end, "Dense data is out of bounds");
        static_cast<U*>(static_cast<void*>(data))[*idx] = segments->Get(*idx);
        last_idx = *idx;
    }
}

void* ov::frontend::tensorflow_lite::SparsityInfo::densify() {
    size_t sparse_idx = 0;
    for (; sparse_idx < m_dim_format.size(); ++sparse_idx) {
        if (m_dim_format[sparse_idx] == ::tflite::DimensionType_SPARSE_CSR)
            break;
    }
    FRONT_END_GENERAL_CHECK(sparse_idx < m_dim_format.size(), "Sparse dimension isn't found for sparse tensor");
    FRONT_END_GENERAL_CHECK(sparse_idx == (m_dim_format.size() - 1),
                            "Supports only sparse tensor with sparse dimension as a last dimension");

    size_t total_size = 0,  // Size of data in bytes
        row_size = 0;       // Size of data row in bytes
    switch (m_data_desc[sparse_idx].segments_type) {
    case ::tflite::SparseIndexVector_Uint8Vector:
        total_size = 1;
        break;
    case ::tflite::SparseIndexVector_Uint16Vector:
        total_size = 2;
        break;
    case ::tflite::SparseIndexVector_Int32Vector:
        total_size = 4;
        break;
    }
    row_size = total_size;  // Byte size is same
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
                             row_size,
                             static_cast<const ::tflite::Uint8Vector*>(m_data_desc[sparse_idx].indices)->values(),
                             static_cast<const ::tflite::Uint8Vector*>(m_data_desc[sparse_idx].segments)->values());
            break;
        case ::tflite::SparseIndexVector_Uint16Vector:
            read_sparse_data(m_data.data(),
                             m_data.data() + total_size,
                             row_size,
                             static_cast<const ::tflite::Uint8Vector*>(m_data_desc[sparse_idx].indices)->values(),
                             static_cast<const ::tflite::Uint16Vector*>(m_data_desc[sparse_idx].segments)->values());
            break;
        case ::tflite::SparseIndexVector_Int32Vector:
            read_sparse_data(m_data.data(),
                             m_data.data() + total_size,
                             row_size,
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
                             row_size,
                             static_cast<const ::tflite::Uint16Vector*>(m_data_desc[sparse_idx].indices)->values(),
                             static_cast<const ::tflite::Uint8Vector*>(m_data_desc[sparse_idx].segments)->values());
            break;
        case ::tflite::SparseIndexVector_Uint16Vector:
            read_sparse_data(m_data.data(),
                             m_data.data() + total_size,
                             row_size,
                             static_cast<const ::tflite::Uint16Vector*>(m_data_desc[sparse_idx].indices)->values(),
                             static_cast<const ::tflite::Uint16Vector*>(m_data_desc[sparse_idx].segments)->values());
            break;
        case ::tflite::SparseIndexVector_Int32Vector:
            read_sparse_data(m_data.data(),
                             m_data.data() + total_size,
                             row_size,
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
                             row_size,
                             static_cast<const ::tflite::Int32Vector*>(m_data_desc[sparse_idx].indices)->values(),
                             static_cast<const ::tflite::Uint8Vector*>(m_data_desc[sparse_idx].segments)->values());
            break;
        case ::tflite::SparseIndexVector_Uint16Vector:
            read_sparse_data(m_data.data(),
                             m_data.data() + total_size,
                             row_size,
                             static_cast<const ::tflite::Int32Vector*>(m_data_desc[sparse_idx].indices)->values(),
                             static_cast<const ::tflite::Uint16Vector*>(m_data_desc[sparse_idx].segments)->values());
            break;
        case ::tflite::SparseIndexVector_Int32Vector:
            read_sparse_data(m_data.data(),
                             m_data.data() + total_size,
                             row_size,
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