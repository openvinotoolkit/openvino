// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "pass_manager.h"
#include "data_inst.h"
#if 0 // TODO(taylor)
#include "mutable_data_inst.h"
#endif
#include "program_node.h"
#include "intel_gpu/runtime/engine.hpp"
#include "runtime/cldnn_itt.hpp"
#include <iostream>
#include <cmath>
#include <iomanip>

#include <threading/ie_cpu_streams_executor.hpp>

using namespace cldnn;

void compile_graph::run(program& p) {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNN, "CLDNN::pass::CompileGraph");
    for (auto& node : p.get_processing_order()) {
        node->set_unique_id();
        if (!node->is_type<data>()) {
            for (int i = 0; i < node->get_outputs_count() ; ++i) {
                node->get_output_layout(true, i);
            }
        }
    }

    auto task_executor = p.get_engine().get_task_executor();
    auto& proc_order = p.get_processing_order();
    std::vector<InferenceEngine::Task> tasks;
    std::exception_ptr exception;
    for (int idx = 0; idx < proc_order.size(); idx++) {
        auto& node = *(std::next(proc_order.begin(), idx));
#if 0 // TODO(taylor)
        if (!node->is_type<data>() && !(node->is_type<mutable_data>() && node->get_dependencies().empty())) {
#else
        if (!node->is_type<data>()) {
#endif
            tasks.push_back([node, &exception] {
                try {
                    node->selected_impl = node->type()->choose_impl(*node);
                } catch(...) {
                    exception = std::current_exception();
                }
            });
        }
    }

    task_executor->runAndWait(tasks);
    tasks.clear();

    if (exception) {
        std::rethrow_exception(exception);
    }
}
