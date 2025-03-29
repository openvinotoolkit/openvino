// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <exception>

#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/mutable_data.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include "pass_manager.h"
#include "program_node.h"
#include "registry/implementation_manager.hpp"
#include "registry/registry.hpp"

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
    std::vector<ov::threading::Task> tasks;
    std::exception_ptr exception;

    for (size_t idx = 0; idx < proc_order.size(); idx++) {
        const auto& node = *(std::next(proc_order.begin(), idx));

        bool can_select_impl = !node->is_type<data>() && !(node->is_type<mutable_data>() && node->get_dependencies().empty());

        if (can_select_impl) {
            tasks.emplace_back([node, &exception] {
                try {
                    const auto& params = node->get_kernel_impl_params();
                    auto shape_type = ImplementationManager::get_shape_type(*params);
                    auto selected_impl_manager = node->type()->choose_impl(*node, shape_type);
                    std::string fail_reason;
                    try {
                        if (selected_impl_manager) {
                            node->selected_impl = selected_impl_manager->create(*node, *params);
                        }
                    } catch (std::exception& e) {
                        fail_reason = e.what();
                    }

                    OPENVINO_ASSERT(shape_type == shape_types::dynamic_shape || node->selected_impl != nullptr,
                                    "[GPU] Failed to select implementation for"
                                    "\nname:",
                                    node->id(),
                                    "\ntype: ",
                                    node->get_primitive()->type_string(),
                                    "\noriginal_type: ",
                                    node->get_primitive()->origin_op_type_name,
                                    fail_reason);
                } catch (std::exception&) {
                    exception = std::current_exception();
                }
            });
        }
    }

    task_executor->run_and_wait(tasks);
    tasks.clear();

    if (exception) {
        std::rethrow_exception(exception);
    }
}
