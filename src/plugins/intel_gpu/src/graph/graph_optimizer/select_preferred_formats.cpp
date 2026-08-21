// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "registry/implementation_manager.hpp"
#include "intel_gpu/primitives/implementation_desc.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "pass_manager.h"
#include "program_node.h"
#include "permute_inst.h"
#include "openvino/core/except.hpp"
#include "intel_gpu/primitives/deconvolution.hpp"
#include "intel_gpu/primitives/convolution.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "to_string_utils.h"
#include <iostream>
#include <sstream>


using namespace cldnn;

namespace {

void print_selected_formats(const program_node& n) {
    std::stringstream ss;
    ov::write_all_to_stream(ss, "select_preferred_formats:", n.id(), ":\n");

    const auto& in_fmts = n.get_preferred_input_fmts();
    const auto& out_fmts = n.get_preferred_output_fmts();

    for (size_t i = 0; i < in_fmts.size(); i++) {
        ss << "\tIn " << i << ": " << fmt_to_str(in_fmts[i]) << std::endl;
    }

    for (size_t i = 0; i < out_fmts.size(); i++) {
        ss << "\tOut " << i << ": " << fmt_to_str(out_fmts[i]) << std::endl;
    }
    GPU_DEBUG_LOG << ss.str() << std::endl;
}

static void optimize_conv_permute(program_node& node) {
    // In conv-permute pattern, sets the output format of conv to byxf so that permute can be optimized.
    // ex) oneDNN convolution -> (byxf) -> permute -> (bfyx) -> output
    //     output layout of convolution: byxf [b:1, f:128, y:2, x:2]
    //     output layout of permute:     bfyx [b:1, f:2, y:2, x:128]
    // In this case, it can be handled by changing only the shape of permute without the kernel execution.
    if (node.get_output_layout().get_rank() == 4
        && node.get_users().size() == 1 && node.get_users().front()->is_type<permute>()) {
        auto& pnode = node.get_users().front()->as<permute>();
        auto can_optimize_permute = pnode.get_output_layout().data_type == node.get_output_layout().data_type
            && !pnode.has_fused_primitives()
            && !pnode.is_output() && pnode.get_input_layout(0).is_static()
            && pnode.is_rotating_except_batch();
        if (can_optimize_permute) {
            node.set_preferred_output_fmt(0, format::byxf);
            pnode.init_preferred_fmt(1, 1);
            pnode.set_preferred_input_fmt(0, cldnn::format::byxf);
            pnode.set_preferred_output_fmt(0, cldnn::format::bfyx);
            pnode.can_be_optimized(true);
        }
    }
}

static void optimize_permute_conv(program_node& node) {
    // Goal: Eliminate the Reorder by aligning connection to byxf
    if (node.get_dependencies().empty())
        return;

    auto& dep = node.get_dependency(0);

    // Dependency must be a Permute node (not network output)
    if (!dep.is_type<permute>() || dep.is_output())
        return;

    if ((node.get_users().size() != 1) || (node.get_output_layout().get_rank() != 4))
        return;

    auto& pnode = dep.as<permute>();

    if (pnode.get_output_layout().data_type != node.get_output_layout().data_type)
        return;

    // NHWC <-> NCHW (ensures reverse rotation pattern)
    if (!pnode.is_reverse_rotating_except_batch())
        return;

    auto pnode_upstream_fmt = pnode.get_dependency(0).get_preferred_output_fmt();
    auto node_fmt = node.get_preferred_output_fmt();

    bool is_compatible_format = ((pnode_upstream_fmt == format::bfyx || pnode_upstream_fmt == format::any)
                                && (node_fmt == format::byxf));

    if (!is_compatible_format)
        return;

    // Set the layouts so that the memory buffer is re-interpreted rather than physically shuffled.
    node.set_preferred_input_fmt(0, format::byxf);

    // Set Permute Input to match upstream format
    pnode.set_preferred_input_fmt(0, format::bfyx);

    // This aligns with the Pre-Transpose memory, allowing it to be a zero copy optimization.
    // alternative approach is to force the Permute node to set output fmt to match convolution input fmt
    // which will eliminate reorder before convolution: pnode.set_preferred_output_fmt(0, node.get_preferred_input_fmt())
    // however, for non planar blocked format (b_fs_yx_fsv16) The kernel has to calculate complex offsets to pack 16 channels together into a block.
    // and may degrade performance which may possibly be not visible for small input like 56x56
    // as small working set the working set fits largely in the GPU L2 cache. And penalty for complex addressing calculation
    // may get masked by high L2 bandwidth.
    pnode.set_preferred_output_fmt(0, format::byxf);

    if (!pnode.has_fused_primitives()) {
        pnode.can_be_optimized(true);
    }
}

} // namespace

void select_preferred_formats::run(program& p) {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "pass::select_preferred_formats");

#ifdef ENABLE_ONEDNN_FOR_GPU
    auto& engine = p.get_engine();
    if (!p.get_layout_optimizer().is_empty_onednn_impls_optimization_attribute()) {
        engine.create_onednn_engine(p.get_config());
    }
#endif  // ENABLE_ONEDNN_FOR_GPU

    auto forcing_map = p.get_config().get_force_implementations();

    for (auto* n : p.get_processing_order()) {
        n->recalc_output_layout();
        if (n->is_input() || !n->is_in_data_flow()) {
            continue;
        }

        auto forced_fmt = format::any;
        auto forced_impl = impl_types::any;

        if (std::find_if(forcing_map.begin(), forcing_map.end(),
                [&n](const std::pair<std::string, ov::intel_gpu::ImplementationDesc>& it) {
                    return (it.first == n->id() && it.second.output_format != format::any);
                }) != forcing_map.end()) {
            forced_fmt = forcing_map.at(n->id()).output_format;
            forced_impl = forcing_map.at(n->id()).impl_type;
        }

        const auto& params = n->get_kernel_impl_params();
        auto shape_type = ImplementationManager::get_shape_type(*params);

        // For convolutions with ≤4 input channels (e.g. RGB/RGBD first conv), the clDNN
        // ConvolutionKernel_bfyx_to_bfyx_f16 kernel can directly produce b_fs_yx_fsv16 output
        // from bfyx input, eliminating a costly bfyx→b_fs_yx_fsv16 reorder.  However, when we
        // query with format::any, oneDNN is chosen (it outputs bfyx for shallow convolutions) and
        // its bfyx preference is recorded, which later prevents the layout optimizer from
        // assigning b_fs_yx_fsv16 to this node.
        // Fix: for shallow convolutions, first try to find a clDNN impl that supports
        // b_fs_yx_fsv16 output.  If found, use that factory so query_formats returns
        // b_fs_yx_fsv16, bypassing oneDNN's bfyx recommendation.
        bool is_shallow_conv_fsv16_candidate = false;
        if (n->is_type<convolution>()) {
            auto& conv_node = n->as<convolution>();
            auto conv_in_layout = conv_node.get_input_layout(0);
            auto conv_out_layout = conv_node.calc_output_layout();
            is_shallow_conv_fsv16_candidate = (conv_in_layout.feature() <= 4 &&
                                               conv_out_layout.feature() >= 16 &&
                                               conv_in_layout.format == format::bfyx);
        }

        // temporary set format to any as we need to query that from impl and don't want impl to be rejected
        // also drop padding as it may be handled later
        auto factory = test_format<std::shared_ptr<ImplementationManager>>(*n, format::any,
            [&shape_type](program_node& n) {
                return test_no_input_pad<std::shared_ptr<ImplementationManager>>(n, [&shape_type](program_node& n) {
                    return n.type()->choose_impl(n, shape_type);
            });
        });

        // For shallow convolutions, override with a b_fs_yx_fsv16-capable clDNN factory
        // if one is available.  This makes query_formats return b_fs_yx_fsv16 as the
        // preferred output format, which allows the bfyx_to_bfyx_f16 kernel to be selected
        // and avoids the post-conv layout reorder.
        if (is_shallow_conv_fsv16_candidate) {
            auto factory_fsv16 = test_format<std::shared_ptr<ImplementationManager>>(*n, format::b_fs_yx_fsv16,
                [&shape_type](program_node& n) {
                    return test_no_input_pad<std::shared_ptr<ImplementationManager>>(n, [&shape_type](program_node& n) {
                        return n.type()->choose_impl(n, shape_type);
                    });
                });
            if (factory_fsv16 && factory_fsv16->get_impl_type() == impl_types::ocl) {
                GPU_DEBUG_LOG << "[select_preferred_formats] Shallow conv " << n->id()
                              << ": overriding factory with clDNN ocl impl for b_fs_yx_fsv16 output\n";
                factory = factory_fsv16;
            }
        }

        if (factory) {
            try {
                auto fmts = factory->query_formats(*n);
                for (size_t i = 0; i < fmts.first.size(); i++) {
                    n->set_preferred_input_fmt(i, fmts.first[i]);
                }
                for (size_t i = 0; i < fmts.second.size(); i++) {
                    n->set_preferred_output_fmt(i, fmts.second[i]);
                }

                if ((forced_impl & factory->get_impl_type()) == factory->get_impl_type() && forced_fmt != format::any) {
                    n->set_preferred_output_fmt(0, forced_fmt);
                }
                if (factory->get_impl_type() == impl_types::onednn && (n->is_type<convolution>() || n->is_type<deconvolution>())) {
                    optimize_conv_permute(*n);
                    optimize_permute_conv(*n);
                }
            } catch (std::exception& exception) {
                GPU_DEBUG_LOG << "WARNING(select_preferred_formats): " << exception.what() << std::endl;
            }
            print_selected_formats(*n);
        }
    }
}
