// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass_manager.h"
#include "data_inst.h"
#include "mutable_data_inst.h"
#include "reshape_inst.h"
#include "program_node.h"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include <iostream>
#include <cmath>
#include <iomanip>

#include <threading/ie_cpu_streams_executor.hpp>

using namespace cldnn;

void compile_graph::run(program& p) {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "pass::CompileGraph");
    for (auto& node : p.get_processing_order()) {
        node->set_unique_id();
        if (!node->is_type<data>()) {
            node->get_output_layout();
        }
    }

    auto task_executor = p.get_task_executor();
    auto& proc_order = p.get_processing_order();
    std::vector<InferenceEngine::Task> tasks;
    std::exception_ptr exception;
    for (size_t idx = 0; idx < proc_order.size(); idx++) {
        auto& node = *(std::next(proc_order.begin(), idx));
        bool can_select_impl = !node->is_type<data>() &&
                               !(node->is_type<mutable_data>() && node->get_dependencies().empty()) &&
                               (!node->is_dynamic() || node->type()->does_dynamic_implementation_exist(*node));

        // TODO: Remove this WA once we have shape agnostic reshape kernel
        if (node->is_type<reshape>() && node->is_dynamic() && !node->can_be_optimized())
            can_select_impl = false;

        // TODO: need to come up with better handling of unsupported shape agnostic cases
        // e.g. process exceptions from choose_impl() and ignore those for dynamic parameters
        if (node->is_type<fully_connected>() && node->is_dynamic() && node->get_output_layout().get_partial_shape().size() > 3)
            can_select_impl = false;

        bool is_planar = node->get_output_layout().format == format::bfyx ||
                         node->get_output_layout().format == format::bfzyx ||
                         node->get_output_layout().format == format::bfwzyx;

        if (node->is_dynamic() && !is_planar)
            can_select_impl = false;

        if (can_select_impl) {
            tasks.push_back([node, &p, &exception] {
                try {
                    node->selected_impl = node->type()->choose_impl(*node);
                    if (node->selected_impl) {
                        auto kernel_ids = p.get_kernels_cache().add_kernels_source(node->selected_impl->get_kernels_source());
                        node->selected_impl->set_kernel_ids(kernel_ids);
                    }
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
