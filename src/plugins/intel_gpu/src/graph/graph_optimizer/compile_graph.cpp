// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/itt.hpp"

#include "pass_manager.h"
#include "data_inst.h"
#include "mutable_data_inst.h"
#include "reshape_inst.h"
#include "proposal_inst.h"
#include "permute_inst.h"
#include "quantize_inst.h"
#include "arg_max_min_inst.h"
#include "fully_connected_inst.h"
#include "gemm_inst.h"
#include "condition_inst.h"
#include "paged_attention_inst.h"
#include "loop_inst.h"
#include "group_normalization_inst.h"
#include "program_node.h"

#include <iostream>
#include <cmath>
#include <iomanip>

#include "openvino/runtime/threading/cpu_streams_executor.hpp"

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
    bool disable_permute_fuse_onednn_gemm = false;
    GPU_DEBUG_GET_INSTANCE(debug_config);
    GPU_DEBUG_IF(debug_config->disable_onednn_permute_fusion == 1)
        disable_permute_fuse_onednn_gemm = true;

    for (size_t idx = 0; idx < proc_order.size(); idx++) {
        auto& node = *(std::next(proc_order.begin(), idx));
        const bool use_shape_agnostic_impl = !p.get_config().get_property(ov::intel_gpu::use_only_static_kernels_for_dynamic_shape);
        const impl_types original_impl_type = node->get_preferred_impl_type();
        bool change_initial_impl = node->is_dynamic() && original_impl_type == impl_types::onednn;

        if (change_initial_impl) {
            if (node->is_type<fully_connected>()) {
                // Do not change impl (i.e. do not use ocl shape-agnostic kernels)
                // since oneDNN primitives/kernels caching mechanism will be used instead.
                change_initial_impl = false;
            } else if (node->is_type<gemm>() && !disable_permute_fuse_onednn_gemm) {
                // permute is fused to onednn gemm. The updated memory formats are not supported by ocl this keep onednn impl
                for (const auto& dep : node->get_dependencies()) {
                    if (dep.first->is_type<permute>() && dep.first->can_be_optimized() && !dep.first->is_runtime_skippable() &&
                        node->get_preferred_input_fmt() != format::any)
                        change_initial_impl = false;
                }
                for (const auto& user : node->get_users()) {
                    if (user->is_type<permute>() && user->can_be_optimized() && !user->is_runtime_skippable() &&
                        node->get_preferred_output_fmt() != format::any)
                        change_initial_impl = false;
                }
            }
            if (node->is_type<convolution>()) {
                auto w_layout = node->as<convolution>().weights().get_output_layout();
                // Convolution_fsv16_1x1 is only available shape agnostic kernel for onednn convolution which uses the block format.(fsv16)
                // Onednn convolution doesn't support input padding but most of cldnn optimized convolution require input padding except fsv16_1x1.
                if (w_layout.spatial(0) != 1 || w_layout.spatial(1) != 1) {
                    change_initial_impl = false;
                }
            }
        }

        if (change_initial_impl)
            node->set_preferred_impl_type(impl_types::ocl);

        bool can_select_impl = !node->is_type<data>() &&
                               !(node->is_type<mutable_data>() && node->get_dependencies().empty()) &&
                               (!node->is_dynamic() || (use_shape_agnostic_impl && node->type()->does_dynamic_implementation_exist(*node)));

        // TODO: Remove this WA once we have shape agnostic reshape kernel
        if (node->is_type<reshape>() && node->is_dynamic() && !node->can_be_optimized())
            can_select_impl = false;

        // TODO: Remove this WA once we have shape agnostic conv kernl with specified auto_pad attributes
        if (node->is_type<convolution>() && node->is_dynamic() && !node->as<convolution>().use_explicit_padding()) {
            can_select_impl = false;
        }

        // TODO: need to come up with better handling of unsupported shape agnostic cases
        // e.g. process exceptions from choose_impl() and ignore those for dynamic parameters
        if (node->is_type<fully_connected>() && node->is_dynamic() && node->get_output_pshape().size() > 3)
            can_select_impl = false;

        // onednn impls do not support shape agnostic kernel currently.
        if (node->get_preferred_impl_type() == impl_types::onednn && node->is_dynamic())
            can_select_impl = false;

        // TODO: Remove this WA once we have shape agnostic arg_max_min_axis kernel with non-const k input
        if (node->is_type<arg_max_min>() && node->is_dynamic() && node->as<arg_max_min>().get_primitive()->top_k == 0) {
            can_select_impl = false;
        }

        bool is_planar = format::is_default_format(node->get_output_layout().format);

        if (node->is_dynamic() && !is_planar) {
            if (!(node->is_type<convolution>() && node->get_output_layout().format == cldnn::format::b_fs_yx_fsv16) &&
                !(node->is_type<group_normalization>() && node->get_output_layout().format == cldnn::format::b_fs_yx_fsv16)) {
                can_select_impl = false;
            }
        }

        if (node->is_type<condition>() || node->is_type<loop>() || node->is_type<proposal>())
            can_select_impl = true;

        // if (node->is_type<paged_attention>())
        //     can_select_impl = false;

        if (can_select_impl) {
            tasks.push_back([node, &exception, change_initial_impl, original_impl_type] {
                try {
                    node->selected_impl = node->type()->choose_impl(*node);
                    if (change_initial_impl) {
                        GPU_DEBUG_TRACE_DETAIL << node->id() << ": use " << node->get_preferred_impl_type()
                                               << " as initial impl instead of " << original_impl_type << std::endl;
                        node->set_preferred_impl_type(original_impl_type);
                    }
                } catch(...) {
                    exception = std::current_exception();
                }
            });
        } else {
            if (change_initial_impl) {
                node->set_preferred_impl_type(original_impl_type);
            }
        }
    }

    task_executor->run_and_wait(tasks);
    tasks.clear();

    if (exception) {
        std::rethrow_exception(exception);
    }
}
