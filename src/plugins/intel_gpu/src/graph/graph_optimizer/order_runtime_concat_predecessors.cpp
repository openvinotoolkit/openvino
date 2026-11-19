// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "pass_manager.h"
#include "program_node.h"
#include "runtime_concat_order.h"

using namespace cldnn;

// F9: make the processing order agree with network::_exec_order for runtime in-place concats.
//  * cancel (all queue types): a runtime-optimizable concat whose predecessors cannot be safely re-issued right before it loses its
//    can_be_optimized / runtime_skippable flags. Predecessor can_share_buffer(false) and padding masks are intentionally left as is.
//  * move (in_order only): the non-data predecessors of each remaining concat are placed, in dependency order, right before it.
//    Under out_of_order the PO stays the BFS result and network::build_exec_order performs the (idempotent) move on its own.
void order_runtime_concat_predecessors::run(program& p) {
    auto& po = p.get_processing_order();

    std::vector<program_node*> concats;
    for (auto* node : po) {
        if (runtime_concat_order::is_runtime_optimized_concat(node)) {
            concats.push_back(node);
        }
    }

    // Decide first, mutate afterwards, so that the verdicts do not depend on the visiting order.
    std::vector<program_node*> to_cancel;
    std::vector<program_node*> to_move;
    for (auto* c : concats) {
        (runtime_concat_order::must_cancel_runtime_concat(*c) ? to_cancel : to_move).push_back(c);
    }

    for (auto* c : to_cancel) {
        GPU_DEBUG_LOG << "order_runtime_concat_predecessors: cancel runtime in-place concat " << c->id() << std::endl;
        c->can_be_optimized(false);
        c->set_runtime_skippable(false);
    }

    if (p.get_config().get_queue_type() != QueueTypes::in_order) {
        return;
    }

    for (auto* c : to_move) {
        for (const auto& dep : c->get_dependencies()) {
            auto* d = dep.first;
            if (d->is_type<data>()) {
                continue;
            }
            po.erase(d);
            po.insert(c, d);
        }
    }

#ifndef NDEBUG
    if (!to_move.empty()) {
        OPENVINO_ASSERT(po.is_correct(), "[GPU] order_runtime_concat_predecessors broke the processing order");
    }
#endif
}
