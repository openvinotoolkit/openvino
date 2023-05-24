// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <vector>
#include "cpu_memory.h"
#include "openvino/core/shape.hpp"
#include "tensor_data_accessor.hpp"
#include <ie_ngraph_utils.hpp>

namespace ov {
namespace intel_cpu {
/**
 * @brief cpu memory accessor implementing ov::ITensorAccessor to get data as tensor from cpu container.
 */
struct MemoryAccessor : public ov::ITensorAccessor {
    using container_type = std::unordered_map<size_t, MemoryPtr>;

    explicit MemoryAccessor(const container_type& ptrs, const std::vector<int64_t>& ranks)
        : m_ptrs{ptrs}, m_ranks(ranks) {}


    ~MemoryAccessor() {
    }

    ov::Tensor operator()(size_t port) const override {
        const auto t_iter = m_ptrs.find(port);
        if (t_iter != m_ptrs.cend()) {
            auto memPtr = t_iter->second;
            // use scalar shape {} instead of {1} if required by shapeInference
            ov::Shape shape;
            if (m_ranks[port] != 0) {
                shape = ov::Shape(memPtr->getStaticDims());
            }
            return {InferenceEngine::details::convertPrecision(memPtr->getDesc().getPrecision()),
                    shape,
                    memPtr->GetPtr()
                   };
        } else {
            return {};
        }
    }

    const container_type m_ptrs;              //!< Pointer to cpu memory pointers with op data.
    std::vector<int64_t> m_ranks;
};
}  // namespace intel_cpu
}  // namespace ov

