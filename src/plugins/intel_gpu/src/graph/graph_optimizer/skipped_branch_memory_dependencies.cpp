// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass_manager.h"
#include "program_node.h"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include <list>

using namespace cldnn;

void skipped_branch_memory_dependencies::run(program& p) {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "pass::SkippedBranchMemoryDependencies");
    // Primitive A can't use primitive B buffer if processing_num(B) < processing_num(A) and for any usr - the user of B
    // processing_num(usr) > processing_num(A) Otherwise it could override data that has to be used in the future.
    auto& processing_order = p.get_processing_order();
    auto itrB = processing_order.begin();
    while (itrB != processing_order.end()) {
        auto& nodeB = *itrB;
        auto itrA = ++itrB;
        if (nodeB->is_constant())
            continue;
        if (nodeB->get_users().size() == 0)
            continue;

        // find the last user of B in processing order
        auto itrUsr = nodeB->get_users().begin();
        auto lastUsr = itrUsr++;
        while (itrUsr != nodeB->get_users().end()) {
            if (processing_order.get_processing_number(*lastUsr) < processing_order.get_processing_number(*itrUsr))
                lastUsr = itrUsr;
            itrUsr++;
        }

        // mark all nodes in between B and lastUsr of B as forbidden to share buffer with B
        while (itrA != processing_order.get_processing_iterator(**lastUsr) && itrA != processing_order.end()) {
            auto& nodeA = *itrA;
            itrA++;
            add_memory_dependency(nodeA, nodeB);
            add_memory_dependency(nodeB, nodeA);
        }
    }
}
