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
    }

    bool get_weights(int64_t offset, WeightsVariant& weights) const {
        auto it = m_offsetConstMap.find(offset);
        if (it != m_offsetConstMap.end()) {
            weights = it->second;
            return true;
        } else {
            std::cout << "Weights not found for the given offset" << std::endl;
            for (const auto& pair : m_offsetConstMap) {
                std::cout << "key: " << pair.first << " ";  // pair.first 是键
            }
            std::cout << std::endl << "all key dumped!" << std::endl;
            return false;
        }
    }

    size_t size() {
        return m_offsetConstMap.size();
    }

private:
    std::unordered_map<int64_t, WeightsVariant> m_offsetConstMap;
};

}  // namespace pass
}  // namespace ov
