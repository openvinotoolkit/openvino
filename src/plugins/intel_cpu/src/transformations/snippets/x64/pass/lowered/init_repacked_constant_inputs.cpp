// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "init_repacked_constant_inputs.hpp"

#include "external_repacking_adjuster.hpp"
#include "snippets/itt.hpp"
#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"

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

        const auto& layout = port_desc->get_layout();
        const auto& shape = port_desc->get_shape();

        const auto& planar_shape = ov::snippets::utils::get_planar_vdims(shape, layout);
        OPENVINO_ASSERT(planar_shape.size() > 1, "Incorrect shape of repacked input of Brgemm");
        const auto& K = *++planar_shape.rbegin();
        const auto& N = *planar_shape.rbegin();
        const auto& prc = param->get_node()->get_output_element_type(0);

        BrgemmExternalRepackingAdjuster::update_kernel(executor, shape, layout, N, K, prc);

        const auto& blk_shape =
            BrgemmExternalRepackingAdjuster::get_blk_shape(planar_shape, prc, BrgemmCopyB::is_transposed(layout));
        const auto& order = BrgemmExternalRepackingAdjuster::get_blk_order(planar_shape.size());
        const auto& desc = std::make_shared<CpuBlockedMemoryDesc>(prc, Shape(planar_shape), blk_shape, order);

        ov::snippets::VectorDims src_offsets, dst_offsets;
        ov::snippets::utils::init_strides(shape, shape.size(), prc.size(), 0, src_offsets);
        ov::snippets::utils::init_strides(blk_shape, blk_shape.size(), prc.size(), 0, dst_offsets);
        // Last three dimensions of blocked shapes are processed in th kernel. To align with src, we removed last stride
        dst_offsets.pop_back();

        m_repacked_const_inputs_config.at(idx) = RepackedInput(executor->get_kernel(), desc, src_offsets, dst_offsets);
    }

    return modified;
}

}  // namespace ov::intel_cpu::pass
