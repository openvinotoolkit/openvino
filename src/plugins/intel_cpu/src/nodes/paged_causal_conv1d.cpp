// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_causal_conv1d.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <vector>

#include "cpu_memory.h"
#include "cpu_shape.h"
#include "graph_context.h"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "nodes/kernels/paged_causal_conv1d.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/paged_causal_conv1d.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu::node {

PagedCausalConv1D::PagedCausalConv1D(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
}

bool PagedCausalConv1D::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                             std::string& errorMessage) noexcept {
    if (op == nullptr || !ov::is_type<ov::op::internal::PagedCausalConv1D>(op)) {
        errorMessage = "Node is not an instance of ov::op::internal::PagedCausalConv1D.";
        return false;
    }

    return true;
}

void PagedCausalConv1D::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    // Natively accept f32/f16/bf16 on the float-typed ports. The kernel treats input_embeds,
    // conv_weight, conv_bias and output uniformly as DataT; we declare all of them with the
    // same precision as input_embeds (port 0).
    // conv_state_table (port 1) is declared independently; the kernel converts its elements
    // to/from f32 during accumulation and write-back.
    const auto data_precision = getOriginalInputPrecisionAtPort(0);
    OPENVINO_ASSERT(any_of(data_precision, ov::element::f32, ov::element::f16, ov::element::bf16),
                    "PagedCausalConv1D supports only f32/f16/bf16 input_embeds precision, got ",
                    data_precision);
    const auto state_precision = getOriginalInputPrecisionAtPort(1);
    OPENVINO_ASSERT(any_of(state_precision, ov::element::f32, ov::element::f16, ov::element::bf16),
                    "PagedCausalConv1D supports only f32/f16/bf16 conv_state_table precision, got ",
                    state_precision);

    std::vector<PortConfigurator> input_configs;
    input_configs.reserve(getParentEdges().size());
    input_configs.emplace_back(LayoutType::ncsp, data_precision, getInputShapeAtPort(0), false, -1);
    input_configs.emplace_back(LayoutType::ncsp, state_precision, getInputShapeAtPort(1), false, -1);
    input_configs.emplace_back(LayoutType::ncsp, data_precision, getInputShapeAtPort(2), false, -1);
    input_configs.emplace_back(LayoutType::ncsp, data_precision, getInputShapeAtPort(3), false, -1);
    input_configs.emplace_back(LayoutType::ncsp, ov::element::i32, getInputShapeAtPort(4), false, -1);
    input_configs.emplace_back(LayoutType::ncsp, ov::element::i32, getInputShapeAtPort(5), false, -1);
    input_configs.emplace_back(LayoutType::ncsp, ov::element::i32, getInputShapeAtPort(6), false, -1);
    input_configs.emplace_back(LayoutType::ncsp, ov::element::i32, getInputShapeAtPort(7), false, -1);
    input_configs.emplace_back(LayoutType::ncsp, ov::element::i32, getInputShapeAtPort(8), false, -1);

    std::vector<PortConfigurator> output_configs = {
        PortConfigurator{LayoutType::ncsp, data_precision, getOutputShapeAtPort(0), false, -1}};

    addSupportedPrimDesc(input_configs, output_configs, impl_desc_type::ref_any);
}

void PagedCausalConv1D::createPrimitive() {
    // Allocate a per-worker-thread f32 scratch buffer holding one promoted conv_state block.
    // Shape: [num_worker_threads, hidden_size * kernel_size]. Each parallel (sequence, channel-block)
    // task picks its buffer row via parallel_get_thread_num() in the kernel.
    // hidden_size (port 0 dim 1) and kernel_size (port 1 dim 2) are static by model convention.
    const auto& input_embeds_dims = getInputShapeAtPort(0).getDims();
    const auto& state_table_dims = getInputShapeAtPort(1).getDims();
    const size_t hidden_size = input_embeds_dims[1];
    const size_t kernel_size = state_table_dims[2];
    const auto num_threads = static_cast<size_t>(context->getCpuParallel()->get_num_worker_threads());
    auto mem_desc =
        std::make_shared<CpuBlockedMemoryDesc>(ov::element::f32,
                                               ov::intel_cpu::Shape{num_threads, hidden_size * kernel_size});
    m_tmpLocalState = context->getScratchPad()->createScratchPadMem(mem_desc);
}

void PagedCausalConv1D::execute([[maybe_unused]] const dnnl::stream& strm) {
    const auto& input_embeds_shape = getSrcMemoryAtPort(0)->getStaticDims();
    const auto& state_table_shape = getSrcMemoryAtPort(1)->getStaticDims();
    const auto& weight_shape = getSrcMemoryAtPort(2)->getStaticDims();
    const auto& bias_shape = getSrcMemoryAtPort(3)->getStaticDims();

    const size_t batch_size_in_tokens = input_embeds_shape[0];
    const size_t hidden_size = input_embeds_shape[1];
    const size_t kernel_size = state_table_shape[2];

    OPENVINO_ASSERT(state_table_shape[1] == hidden_size,
                    "PagedCausalConv1D: conv_state_table hidden_size (",
                    state_table_shape[1],
                    ") != input_embeds hidden_size (",
                    hidden_size,
                    ").");

    // Linear attention models use depthwise convolution where group_size == hidden_size,
    // i.e. conv_weight[1] (in_channels per group) must be 1.
    OPENVINO_ASSERT(weight_shape.size() == 3 && weight_shape[1] == 1,
                    "PagedCausalConv1D only supports depthwise convolution (conv_weight[1] must be 1). "
                    "Got conv_weight shape [",
                    weight_shape[0],
                    ", ",
                    weight_shape[1],
                    ", ",
                    weight_shape[2],
                    "].");

    const bool has_bias = bias_shape[0] != 0;

    const auto data_precision = getSrcMemoryAtPort(0)->getDescPtr()->getPrecision();
    const auto state_precision = getSrcMemoryAtPort(1)->getDescPtr()->getPrecision();
    const auto* input_embeds_raw = getSrcMemoryAtPort(0)->getData();
    auto* conv_state_raw = getSrcMemoryAtPort(1)->getData();
    const auto* conv_weight_raw = getSrcMemoryAtPort(2)->getData();
    const auto* conv_bias_raw = has_bias ? getSrcMemoryAtPort(3)->getData() : nullptr;
    const auto* subsequence_begins = getSrcDataAtPortAs<const int32_t>(4);
    const auto* block_indices = getSrcDataAtPortAs<const int32_t>(5);
    const auto* block_indices_begins = getSrcDataAtPortAs<const int32_t>(6);
    const auto* past_lens = getSrcDataAtPortAs<const int32_t>(7);
    const auto* cache_interval = getSrcDataAtPortAs<const int32_t>(8);

    auto* output_embeds_raw = getDstMemoryAtPort(0)->getData();

    const auto& subseq_shape = getSrcMemoryAtPort(4)->getStaticDims();
    OPENVINO_ASSERT(!subseq_shape.empty(), "PagedCausalConv1D expects non-empty subsequence_begins input.");
    OPENVINO_ASSERT(subseq_shape[0] >= 1, "PagedCausalConv1D expects subsequence_begins shape[0] >= 1.");

    const size_t seq_count = subseq_shape[0] - 1;

    ov::Extensions::Cpu::XARCH::paged_causal_conv1d_exec(input_embeds_raw,
                                                         conv_state_raw,
                                                         conv_weight_raw,
                                                         conv_bias_raw,
                                                         has_bias,
                                                         subsequence_begins,
                                                         block_indices,
                                                         block_indices_begins,
                                                         past_lens,
                                                         cache_interval,
                                                         output_embeds_raw,
                                                         batch_size_in_tokens,
                                                         hidden_size,
                                                         kernel_size,
                                                         seq_count,
                                                         data_precision,
                                                         state_precision,
                                                         m_tmpLocalState->getDataAs<float>(),
                                                         context->getCpuParallel());
}

}  // namespace ov::intel_cpu::node
