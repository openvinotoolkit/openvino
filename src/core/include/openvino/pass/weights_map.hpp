// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <filesystem>
#include <functional>
#include <string>
#include <variant>

#include "openvino/core/model.hpp"
#include "openvino/opsets/opset.hpp"
#include "openvino/pass/pass.hpp"
#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/runtime/string_aligned_buffer.hpp"

namespace ov {
namespace pass {

typedef std::variant<std::shared_ptr<ov::StringAlignedBuffer>,
                     std::shared_ptr<ov::SharedStringAlignedBuffer>,
                     std::shared_ptr<ov::AlignedBuffer>>
    WeightsVariant;

class OPENVINO_API WeightsMap {
public:
    WeightsMap() {}
    WeightsMap(const WeightsMap&) = delete;
    WeightsMap& operator=(const WeightsMap&) = delete;
    ~WeightsMap() {
        m_offsetConstMap.clear();
    }

    void add_weights(int64_t offset, const WeightsVariant& weights) {
        m_offsetConstMap[offset] = weights;
        std::pair<int64_t, WeightsVariant> pair(offset, weights);
        m_offsetConstMapVector.push_back(pair);
    }

    void get_weights(int64_t offset, WeightsVariant& weights) const {
        auto it = m_offsetConstMap.find(offset);
        if (it != m_offsetConstMap.end()) {
            weights = it->second;
        } else {
            throw std::runtime_error("Weights not found for the given offset");
        }
    }

    size_t size() {
        return sizeof(m_offsetConstMapVector) +
               sizeof(std::pair<int64_t, WeightsVariant>) * m_offsetConstMapVector.capacity();
    }

private:
    std::unordered_map<int64_t, WeightsVariant> m_offsetConstMap;
    std::vector<std::pair<int64_t, WeightsVariant>> m_offsetConstMapVector;
};

}  // namespace pass
}  // namespace ov
