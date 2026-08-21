// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fuse_terminal_eltwise.hpp"

#include <algorithm>
#include <array>
#include <map>
#include <utility>
#include <vector>

#include "activation_inst.h"
#include "backend_fusion_policy.hpp"
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
    return std::none_of(producer.get_dependencies().begin(),
                        producer.get_dependencies().end(),
                        [peer](const auto& dependency) {
                            return dependency.first == peer;
                        });
}

}  // namespace

void fuse_terminal_eltwise::run(program& program) const {
    std::map<primitive_id, std::vector<std::pair<primitive_id, size_t>>> fusing_history;
    bool processing_order_changed = false;
    bool fused_output = false;

    do {
        fused_output = false;
        const auto outputs = program.get_outputs();
        for (auto* output : outputs) {
            const auto candidate = select_candidate(program, *output);
            if (!candidate.has_value()) {
                continue;
            }

            const auto producer_order = program.get_processing_order().get_processing_number(candidate->producer);
            const auto peer_order = candidate->peer == nullptr
                                        ? producer_order
                                        : program.get_processing_order().get_processing_number(candidate->peer);
            if (producer_order < peer_order) {
                processing_order_changed = true;
            }

            const auto fused_primitive_index = candidate->producer->get_fused_primitives().size();
            program.fuse_nodes(*candidate->producer, *candidate->output, &fusing_history);

            // Output identity transfer renames the original output primitive. Keep the fused descriptor
            // independent and keyed by the physical producer ID for cache reconstruction.
            auto& fused_primitive = candidate->producer->get_fused_primitives().at(fused_primitive_index);
            auto fused_primitive_desc = fused_primitive.desc->clone();
            fused_primitive_desc->id = candidate->producer->id();
            fused_primitive.desc = std::move(fused_primitive_desc);

            program.transfer_output_identity(*candidate->output, *candidate->producer);
            OPENVINO_ASSERT(program.remove_if_dangling(*candidate->output),
                            "[GPU][Vulkan] Terminal Eltwise fusion left the output consumer connected");
            fused_output = true;
        }
    } while (fused_output);

    if (processing_order_changed) {
        program.get_processing_order().calc_processing_order(program);
    }
}

std::optional<fuse_terminal_eltwise::fusion_candidate> fuse_terminal_eltwise::select_candidate(
    program& program,
    program_node& output) const {
    if (!output.is_output() || output.is_in_shape_of_subgraph()) {
        return std::nullopt;
    }

    if (output.is_type<eltwise>()) {
        return select_eltwise_candidate(program, output);
    }

    fusion_kind kind;
    if (output.is_type<activation>()) {
        kind = fusion_kind::activation;
    } else if (output.is_type<quantize>()) {
        kind = fusion_kind::quantize;
    } else if (output.is_type<reorder>()) {
        kind = fusion_kind::terminal_reorder;
    } else {
        return std::nullopt;
    }

    if (output.get_dependencies().size() != 1) {
        return std::nullopt;
    }
    auto& producer = output.get_dependency(0);
    if (!can_merge(producer, nullptr) || _fusion_policy.evaluate({kind, producer, output}) != fusion_decision::accept) {
        return std::nullopt;
    }
    return fusion_candidate{&producer, &output, nullptr};
}

std::optional<fuse_terminal_eltwise::fusion_candidate> fuse_terminal_eltwise::select_eltwise_candidate(
    program& program,
    program_node& output) const {
    if (output.get_dependencies().size() != 2) {
        return std::nullopt;
    }

    const auto dependencies = output.get_dependencies();
    std::array<bool, 2> accepted{};
    for (size_t index = 0; index < dependencies.size(); ++index) {
        auto* producer = dependencies[index].first;
        auto* peer = dependencies[1 - index].first;
        accepted[index] =
            !producer->is_in_shape_of_subgraph() && !peer->is_in_shape_of_subgraph() && can_merge(*producer, peer) &&
            _fusion_policy.evaluate({fusion_kind::eltwise, *producer, output, peer}) == fusion_decision::accept;
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
    return fusion_candidate{dependencies[producer_index].first, &output, dependencies[1 - producer_index].first};
}

}  // namespace cldnn::vulkan
