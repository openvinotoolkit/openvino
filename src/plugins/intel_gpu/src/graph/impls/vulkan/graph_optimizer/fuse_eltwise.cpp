// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fuse_eltwise.hpp"

#include <algorithm>
#include <array>
#include <map>
#include <utility>
#include <vector>

#include "../eltwise_fusion_policy.hpp"
#include "activation_inst.h"
#include "eltwise_inst.h"
#include "intel_gpu/graph/program.hpp"
#include "openvino/core/except.hpp"
#include "quantize_inst.h"
#include "reorder_inst.h"

namespace cldnn::vulkan {
namespace {

bool can_merge(const program_node& producer, const program_node* peer) {
    if (producer.get_users().size() != 1) {
        return false;
    }
    if (peer == nullptr) {
        return true;
    }
    return std::none_of(producer.get_dependencies().begin(), producer.get_dependencies().end(), [peer](const auto& dependency) {
        return dependency.first == peer;
    });
}

}  // namespace

void fuse_eltwise::run(program& program) const {
    std::map<primitive_id, std::vector<std::pair<primitive_id, size_t>>> fusing_history;
    bool processing_order_changed = false;
    bool fused_consumer = false;

    do {
        fused_consumer = false;
        std::vector<primitive_id> consumer_ids;
        consumer_ids.reserve(program.get_processing_order().size());
        for (const auto* node : program.get_processing_order()) {
            consumer_ids.push_back(node->id());
        }

        for (const auto& consumer_id : consumer_ids) {
            if (!program.has_node(consumer_id)) {
                continue;
            }
            auto& consumer = *program.get_node_ptr(consumer_id);
            const auto candidate = select_candidate(program, consumer);
            if (!candidate.has_value()) {
                continue;
            }

            const auto producer_order = program.get_processing_order().get_processing_number(candidate->producer);
            const auto peer_order = candidate->peer == nullptr ? producer_order : program.get_processing_order().get_processing_number(candidate->peer);
            processing_order_changed |= producer_order < peer_order;

            const bool transfer_output = candidate->consumer->is_output();
            const auto fused_primitive_index = candidate->producer->get_fused_primitives().size();
            program.fuse_nodes(*candidate->producer, *candidate->consumer, &fusing_history);

            if (transfer_output) {
                // Output identity transfer renames the original output primitive. Keep the fused descriptor
                // independent and keyed by the physical producer ID for cache reconstruction.
                auto& fused_primitive = candidate->producer->get_fused_primitives().at(fused_primitive_index);
                auto fused_primitive_desc = fused_primitive.desc->clone();
                fused_primitive_desc->id = candidate->producer->id();
                fused_primitive.desc = std::move(fused_primitive_desc);
                program.transfer_output_identity(*candidate->consumer, *candidate->producer);
                OPENVINO_ASSERT(program.remove_if_dangling(*candidate->consumer), "[GPU][Vulkan] Eltwise output fusion left the consumer connected");
            }
            fused_consumer = true;
        }
    } while (fused_consumer);

    if (processing_order_changed) {
        program.get_processing_order().calc_processing_order(program);
    }
}

std::optional<fuse_eltwise::fusion_candidate> fuse_eltwise::select_candidate(program& program, program_node& consumer) const {
    if (consumer.is_in_shape_of_subgraph()) {
        return std::nullopt;
    }

    if (consumer.is_type<eltwise>()) {
        return select_eltwise_candidate(program, consumer);
    }

    if (consumer.get_dependencies().size() != 1) {
        return std::nullopt;
    }
    auto& producer = consumer.get_dependency(0);
    if (!can_merge(producer, nullptr)) {
        return std::nullopt;
    }

    const bool accepted = (consumer.is_type<activation>() && _policy.allows_activation(producer, consumer)) ||
                          (consumer.is_type<quantize>() && _policy.allows_quantize(producer, consumer)) ||
                          (consumer.is_type<reorder>() && _policy.allows_reorder(producer, consumer));
    if (!accepted) {
        return std::nullopt;
    }
    return fusion_candidate{&producer, &consumer, nullptr};
}

std::optional<fuse_eltwise::fusion_candidate> fuse_eltwise::select_eltwise_candidate(program& program, program_node& consumer) const {
    if (consumer.get_dependencies().size() != 2) {
        return std::nullopt;
    }

    const auto dependencies = consumer.get_dependencies();
    std::array<bool, 2> accepted{};
    for (size_t index = 0; index < dependencies.size(); ++index) {
        auto* producer = dependencies[index].first;
        auto* peer = dependencies[1 - index].first;
        accepted[index] = !producer->is_in_shape_of_subgraph() && !peer->is_in_shape_of_subgraph() && can_merge(*producer, peer) &&
                          _policy.allows_eltwise(*producer, *peer, consumer);
    }

    if (!accepted[0] && !accepted[1]) {
        return std::nullopt;
    }

    size_t producer_index = accepted[0] ? 0 : 1;
    if (accepted[0] && accepted[1] &&
        program.get_processing_order().get_processing_number(dependencies[0].first) <
            program.get_processing_order().get_processing_number(dependencies[1].first)) {
        producer_index = 1;
    }
    return fusion_candidate{dependencies[producer_index].first, &consumer, dependencies[1 - producer_index].first};
}

}  // namespace cldnn::vulkan
