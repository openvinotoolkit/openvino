// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "weights_prefetch.hpp"

#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "openvino/core/weight_sharing_util.hpp"
#include "openvino/op/constant.hpp"

namespace ov::intel_cpu {

void WeightsPrefetch::registerConstants(const std::vector<std::shared_ptr<const ov::op::v0::Constant>>& constants) {
    const std::lock_guard<std::mutex> lock(m_mutex);
    m_constants.reserve(m_constants.size() + constants.size());
    for (const auto& constant : constants) {
        if (m_registered.insert(constant).second) {
            m_constants.emplace_back(constant);
        }
    }
}

void WeightsPrefetch::prefetchOnce() {
    std::call_once(m_once, [this] {
        std::vector<ConstantRef> constants;
        {
            const std::lock_guard<std::mutex> lock(m_mutex);
            constants = std::move(m_constants);
            m_constants.clear();
            m_registered.clear();
        }

        for (const auto& observer : constants) {
            if (const auto constant = observer.lock()) {
                ov::wsh::Extension::hint_prefetch_async(*constant);
            }
        }
    });
}

}  // namespace ov::intel_cpu
