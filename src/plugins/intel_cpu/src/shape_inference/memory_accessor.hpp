// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <vector>

#include "cpu_memory.h"
#include "openvino/core/shape.hpp"
#include "tensor_data_accessor.hpp"

namespace ov::intel_cpu {
/**
 * @brief cpu memory accessor implementing ov::ITensorAccessor to get data as tensor from cpu container.
 */
class MemoryAccessor : public ov::ITensorAccessor {
    using container_type = std::unordered_map<size_t, MemoryPtr>;

public:
    MemoryAccessor(const container_type& ptrs, const std::vector<int64_t>& ranks) : m_ptrs{ptrs}, m_ranks(ranks) {}

    ~MemoryAccessor() = default;

    ov::Tensor operator()(size_t port) const override {
        const auto t_iter = m_ptrs.find(port);
        if (t_iter != m_ptrs.cend()) {
            auto memPtr = t_iter->second;
            // use scalar shape {} instead of {1} if required by shapeInference
            const auto shape = (m_ranks[port] != 0) ? ov::Shape(memPtr->getStaticDims()) : ov::Shape();
            return {memPtr->getDesc().getPrecision(), shape, memPtr->getData()};
        }
        return {};
    }

private:
    const container_type& m_ptrs;  //!< Pointer to cpu memory pointers with op data.
    const std::vector<int64_t>& m_ranks;
};
}  // namespace ov::intel_cpu
