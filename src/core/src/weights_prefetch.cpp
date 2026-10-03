// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/weights_prefetch.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/weight_sharing_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/embedding_segments_sum.hpp"
#include "openvino/op/gather_elements.hpp"
#include "openvino/op/moe.hpp"
#include "openvino/op/util/embeddingbag_offsets_base.hpp"
#include "openvino/op/util/embeddingbag_packed_base.hpp"
#include "openvino/op/util/gather_base.hpp"
#include "openvino/op/util/gather_nd_base.hpp"
#include "ov_ops/gather_compressed.hpp"
#include "ov_ops/gather_matmul.hpp"

namespace ov::weight_sharing {

namespace {

/** @brief Tells whether @p node addresses the data of its input @p port by index. */
bool reads_input_by_index(const ov::Node* node, const size_t port) {
    if (port == 0) {
        // GatherCompressed derives from v8::Gather but declares no parent in its type info,
        // so GatherBase alone does not match it.
        return ov::is_type_any_of<op::util::GatherBase,
                                  op::internal::GatherCompressed,
                                  op::util::GatherNDBase,
                                  op::v6::GatherElements,
                                  op::util::EmbeddingBagOffsetsBase,
                                  op::util::EmbeddingBagPackedBase,
                                  op::v3::EmbeddingSegmentsSum>(node);
    }
    // GatherMatmul selects the expert weights matrix to multiply by; GatherMatmulCompressed derives
    // from it in its type info as well.
    if (port == 1) {
        return ov::is_type<op::internal::GatherMatmul>(node);
    }
    // MOE routes each token to the top-k experts only, its ports 3..8 hold the per-expert weights
    // and biases. MOECompressed derives from it in its type info as well.
    return port >= 3 && ov::is_type<op::internal::MOE>(node);
}

/** @brief Tells whether the whole buffer of @p constant is expected to be read at inference time. */
bool is_read_entirely(const ov::op::v0::Constant& constant) {
    // A constant with no consumer is a model output, so its data is read in full as well.
    const auto& consumers = constant.get_output_target_inputs(0);
    return consumers.empty() || std::any_of(consumers.begin(), consumers.end(), [](const Input<Node>& consumer) {
               return !reads_input_by_index(consumer.get_node(), consumer.get_index());
           });
}

}  // namespace

void WeightsPrefetch::register_constants(const ov::Model& model) {
    register_constants(model.get_ordered_ops());
}

void WeightsPrefetch::register_constants(const std::vector<std::shared_ptr<ov::Node>>& ops) {
    std::vector<ConstantRef> candidates;
    for (const auto& op : ops) {
        if (auto constant = ov::as_type_ptr<const op::v0::Constant>(op); constant && is_read_entirely(*constant)) {
            candidates.emplace_back(std::move(constant));
        }
    }

    const std::lock_guard<std::mutex> lock(m_mutex);
    m_constants.reserve(m_constants.size() + candidates.size());
    for (auto& candidate : candidates) {
        if (m_registered.insert(candidate).second) {
            m_constants.emplace_back(std::move(candidate));
        }
    }
}

void WeightsPrefetch::prefetch_once() {
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
                Extension::hint_prefetch(*constant);
            }
        }
    });
}

}  // namespace ov::weight_sharing
