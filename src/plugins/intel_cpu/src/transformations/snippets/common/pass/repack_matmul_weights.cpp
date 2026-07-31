// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "repack_matmul_weights.hpp"

#include <cstddef>
#include <memory>
#include <numeric>
#include <unordered_set>
#include <utility>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/itt.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/port_descriptor.hpp"
#include "snippets/op/reorder.hpp"
#include "snippets/utils/utils.hpp"

namespace ov::intel_cpu::pass {

RepackMatMulWeights::MatMulWeightsSource RepackMatMulWeights::get_weights_source(
    const std::shared_ptr<ov::Node>& matmul_node,
    [[maybe_unused]] const MemoryPtr& orig_src_mem_ptr) {
    if (const auto& reorder =
            ov::as_type_ptr<ov::snippets::op::Reorder>(matmul_node->input_value(1).get_node_shared_ptr())) {
        const auto& port_desc = ov::snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(reorder->input(0));
        return {ov::snippets::utils::pshape_to_vdims(reorder->get_input_partial_shape(0)), port_desc->get_layout()};
    }

    auto shape = ov::snippets::utils::pshape_to_vdims(matmul_node->get_input_partial_shape(1));
    VectorDims layout(shape.size());
    std::iota(layout.begin(), layout.end(), 0);
    return {std::move(shape), std::move(layout)};
}

CpuBlockedMemoryDescPtr RepackMatMulWeights::get_src_cpu_desc(const RepackMatMulWeights::MatMulWeightsSource& source,
                                                              ov::element::Type precision) {
    const auto planar_shape = Shape{ov::snippets::utils::get_preordered_vdims(source.shape, source.layout)};
    return std::make_shared<CpuBlockedMemoryDesc>(precision, planar_shape, source.shape, source.layout);
}

bool RepackMatMulWeights::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(RepackMatMulWeights);
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "ov::intel_cpu::pass::RepackMatMulWeights")

    const auto& params = model->get_parameters();
    std::unordered_set<size_t> weights_idxs;
    for (const auto& repacker_entry : m_input_repackers) {
        const auto i = repacker_entry.first;
        OPENVINO_ASSERT(i < params.size(), "Incorrect index of externally repacked weights");
        OPENVINO_ASSERT(i < m_src_mem_ptrs.size(), "Incorrect memory index of externally repacked weights");
        const auto& parameter = params[i];

        const auto shape_infer_leaf = ov::snippets::utils::get_leaf_node_of_first_child_shape_infer_seq(parameter);
        const auto& first_child = shape_infer_leaf ? shape_infer_leaf : parameter;
        const auto consumers = first_child->output(0).get_target_inputs();
        OPENVINO_ASSERT(consumers.size() == 1, "Expected one consumer for externally repacked weights");
        const auto consumer = consumers.cbegin()->get_node()->shared_from_this();

        const auto& orig_src_mem_ptr = m_src_mem_ptrs[i];
        const auto repacked = repack(consumer, get_weights_source(consumer, orig_src_mem_ptr), orig_src_mem_ptr);
        if (!repacked) {
            OPENVINO_ASSERT(supports_runtime_repacking(),
                            "Failed to repack weights input ",
                            i,
                            " for ",
                            consumer->get_friendly_name(),
                            ": runtime repacking is not supported");
            continue;
        }

        m_src_mem_ptrs[i] = repacked->memory;
        m_input_repackers[i] = InputRepacker(nullptr, repacked->desc, {}, {});
        weights_idxs.insert(i);
    }

    return !weights_idxs.empty();
}
}  // namespace ov::intel_cpu::pass
