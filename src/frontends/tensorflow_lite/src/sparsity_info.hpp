// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <string>
#include <vector>

#include "common_op_table.hpp"
#include "decoder_map.hpp"
#include "openvino/frontend/tensorflow_lite/visibility.hpp"

namespace ov {
namespace frontend {
namespace tensorflow_lite {

class SparsityInfo : public ov::RuntimeAttribute {
public:
    struct SparsityDataDesc {
        uint8_t segments_type;
        const void* segments;
        uint8_t indices_type;
        const void* indices;
    };

    OPENVINO_RTTI("SparsityInfo");
    SparsityInfo() = default;
    explicit SparsityInfo(const std::vector<int32_t>& shape,
                          const std::vector<int32_t>& traversal_order,
                          const std::vector<int32_t>& block_map,
                          const std::vector<uint16_t>& dim_format,
                          const std::vector<SparsityDataDesc>& data_desc)
        : m_shape(shape),
          m_traversal_order(traversal_order),
          m_block_map(block_map),
          m_dim_format(dim_format),
          m_data_desc(data_desc) {
        enable();
    }

    bool is_copyable() const override;

    // Getters and setters for private attributes
    const std::vector<int32_t>& get_shape() const {
        return m_shape;
    }
    void set_shape(const std::vector<int32_t>& shape) {
        m_shape = shape;
    }
    const std::vector<int32_t>& get_traversal_order() const {
        return m_traversal_order;
    }
    void set_traversal_order(const std::vector<int32_t>& traversal_order) {
        m_traversal_order = traversal_order;
    }
    const std::vector<int32_t>& get_block_map() const {
        return m_block_map;
    }
    void set_block_map(const std::vector<int32_t>& block_map) {
        m_block_map = block_map;
    }
    const std::vector<uint16_t>& get_dim_format() const {
        return m_dim_format;
    }
    void set_dim_format(const std::vector<uint16_t>& dim_format) {
        m_dim_format = dim_format;
    }
    const std::vector<SparsityDataDesc>& get_data_desc() const {
        return m_data_desc;
    }
    void set_data_desc(const std::vector<SparsityDataDesc>& data_desc) {
        m_data_desc = data_desc;
    }
    bool is_disabled() const {
        return m_disabled;
    }
    void disable() {
        m_disabled = true;
    }
    void enable() {
        // We dont count on data_types and data_ptr in case other data is absent
        m_disabled = (m_shape.size() == 0 || m_traversal_order.size() == 0 || m_block_map.size() == 0 ||
                      m_dim_format.size() == 0);
    }
    // Unpack sparse tensor and returns dense data
    void* dense_data() {
        if (m_disabled)
            FRONT_END_THROW("Try to get dense data from unsupported tensor");
        if (m_data.size() > 0)
            return m_data.data();
        else
            return densify();
    }

private:
    std::vector<int32_t> m_shape;               // Target shape of the tensor
    std::vector<int32_t> m_traversal_order;     // Sparse tensor traversal order
    std::vector<int32_t> m_block_map;           // Block map of the tensor
    std::vector<uint16_t> m_dim_format;         // List of dimension's format
    std::vector<SparsityDataDesc> m_data_desc;  // Tensor data descriptors
    std::vector<uint8_t> m_data;                // Dense data
    bool m_disabled;

    // Unpack sparse tensor and return pointer on unpacked data
    void* densify();
};
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
