// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "init_repacked_constant_inputs.hpp"

#include <oneapi/dnnl/dnnl.h>

#include <algorithm>
#include <memory>

#include "cpu_shape.h"
#include "emitters/snippets/x64/kernel_executors/brgemm_copy_b.hpp"
#include "external_repacking_adjuster.hpp"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "openvino/core/except.hpp"
#include "openvino/core/type.hpp"
#include "openvino/itt.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/port_descriptor.hpp"
#include "snippets/op/memory_access.hpp"
#include "snippets/op/reorder.hpp"
#include "snippets/shape_types.hpp"
#include "snippets/utils/utils.hpp"

namespace ov::intel_cpu::pass {

bool InitRepackedConstantInputs::run(const snippets::lowered::LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::InitRepackedConstantInputs")

    bool modified = false;

    const auto& params = linear_ir.get_parameters();
    for (const auto& [idx, _] : m_repacked_const_inputs_config) {
        OPENVINO_ASSERT(idx < params.size(), "Incorrect index of repacked input");

        const auto& param = params[idx];
        const auto executor = BrgemmExternalRepackingAdjuster::create_executor(params[idx], m_cache);

        const auto& shape_infer_seq = ov::snippets::utils::get_first_child_shape_infer_expr_seq(param);
        ov::snippets::lowered::ExpressionPtr desc_expr = param;
        ov::snippets::lowered::PortDescriptorPtr port_desc = nullptr;
        if (!shape_infer_seq.empty()) {
            // [160048] Reorder, as any another ShapeInferOp, should just propagate input shape to output using target
            // order
            //          without data movement. However, currently we have to save desc of input of the Reorder
            //          to support correct input data offsets calculations.
            //          Please, remove this code part when the mentioned ticket is completed.
            const auto& reorder_it = std::find_if(shape_infer_seq.cbegin(),
                                                  shape_infer_seq.cend(),
                                                  [](const ov::snippets::lowered::ExpressionPtr& expr) {
                                                      return ov::is_type<ov::snippets::op::Reorder>(expr->get_node());
                                                  });
            if (reorder_it != shape_infer_seq.cend()) {
                desc_expr = (*reorder_it);
                port_desc = desc_expr->get_input_port_descriptor(0);
            } else {
                desc_expr = shape_infer_seq.back();
            }
        }

        if (!port_desc) {
            for (const auto& child_input : desc_expr->get_output_port_connector(0)->get_consumers()) {
                const auto ma =
                    std::dynamic_pointer_cast<snippets::modifier::MemoryAccess>(child_input.get_expr()->get_node());
                if (ma && ma->is_memory_access_input_port(child_input.get_index())) {
                    port_desc = child_input.get_descriptor_ptr();
                    break;
                }
            }
        }

        OPENVINO_ASSERT(port_desc, "The target Port Descriptor has not been initialized!");
        const auto& layout = port_desc->get_layout();
        const auto& shape = port_desc->get_shape();

        const auto& planar_shape = ov::snippets::utils::get_planar_vdims(shape, layout);
        OPENVINO_ASSERT(planar_shape.size() > 1, "Incorrect shape of repacked input of Brgemm");
        const auto& K = *++planar_shape.rbegin();
        const auto& N = *planar_shape.rbegin();
        const auto& dst_prc = param->get_node()->get_output_element_type(0);

        BrgemmExternalRepackingAdjuster::update_kernel(executor, shape, layout, N, K);

        const auto& config = static_cast<const BrgemmCopyBKernelConfig&>(executor->get_config());
        const auto& blk_shape = BrgemmExternalRepackingAdjuster::get_blk_shape(planar_shape,
                                                                               config.get_wei_N_blk(),
                                                                               config.get_wei_K_blk());
        const auto& order = BrgemmExternalRepackingAdjuster::get_blk_order(planar_shape.size());
        const auto& desc = std::make_shared<CpuBlockedMemoryDesc>(dst_prc, Shape(planar_shape), blk_shape, order);

        const auto src_prc_size = dnnl_data_type_size(config.get_original_wei_dt());
        const auto dst_prc_size = dst_prc.size();

        ov::snippets::VectorDims src_offsets;
        ov::snippets::VectorDims dst_offsets;
        ov::snippets::utils::init_strides(shape, shape.size(), src_prc_size, 0, src_offsets);
        ov::snippets::utils::init_strides(blk_shape, blk_shape.size(), dst_prc_size, 0, dst_offsets);
        // Last three dimensions of blocked shapes are processed in the kernel. To align with src, we removed last
        // stride
        dst_offsets.pop_back();

        m_repacked_const_inputs_config.at(idx) = InputRepacker(executor->get_kernel(), desc, src_offsets, dst_offsets);
    }

    return modified;
}

}  // namespace ov::intel_cpu::pass
