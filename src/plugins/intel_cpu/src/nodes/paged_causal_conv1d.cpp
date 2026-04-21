// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_causal_conv1d.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <vector>

#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "nodes/kernels/paged_causal_conv1d.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/paged_causal_conv1d.hpp"
#include "shape_inference/shape_inference_cpu.hpp"

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

    const auto data_precision = ov::element::f32;
    const auto state_precision = getOriginalInputPrecisionAtPort(1);
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

void PagedCausalConv1D::execute([[maybe_unused]] const dnnl::stream& strm) {
    const auto input_embeds_shape = getSrcMemoryAtPort(0)->getStaticDims();
    const auto state_table_shape = getSrcMemoryAtPort(1)->getStaticDims();
    const auto weight_shape = getSrcMemoryAtPort(2)->getStaticDims();
    const auto bias_shape = getSrcMemoryAtPort(3)->getStaticDims();

    const size_t batch_size_in_tokens = input_embeds_shape[0];
    const size_t hidden_size = input_embeds_shape[1];
    const size_t num_blocks = state_table_shape[0];
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

    const auto* input_embeds = getSrcDataAtPortAs<const float>(0);
    const auto state_precision = getSrcMemoryAtPort(1)->getDescPtr()->getPrecision();
    auto* conv_state_raw = getSrcMemoryAtPort(1)->getData();
    const auto* conv_weight = getSrcDataAtPortAs<const float>(2);
    const float* conv_bias = has_bias ? getSrcDataAtPortAs<const float>(3) : nullptr;
    const auto* subsequence_begins = getSrcDataAtPortAs<const int32_t>(4);
    const auto* block_indices = getSrcDataAtPortAs<const int32_t>(5);
    const auto* block_indices_begins = getSrcDataAtPortAs<const int32_t>(6);
    const auto* past_lens = getSrcDataAtPortAs<const int32_t>(7);
    const auto* cache_interval = getSrcDataAtPortAs<const int32_t>(8);

    auto* output_embeds = getDstDataAtPortAs<float>(0);

    const auto subseq_shape = getSrcMemoryAtPort(4)->getStaticDims();
    OPENVINO_ASSERT(!subseq_shape.empty(), "PagedCausalConv1D expects non-empty subsequence_begins input.");
    OPENVINO_ASSERT(subseq_shape[0] >= 1, "PagedCausalConv1D expects subsequence_begins shape[0] >= 1.");

    const size_t seq_count = subseq_shape[0] - 1;
    const size_t state_stride = hidden_size * kernel_size;
    std::vector<float> local_state(state_stride);

    auto dispatch_kernel = [&](auto* conv_state_table) {
        kernels::paged_causal_conv1d_optimized(input_embeds,
                                               conv_state_table,
                                               conv_weight,
                                               conv_bias,
                                               has_bias,
                                               subsequence_begins,
                                               block_indices,
                                               block_indices_begins,
                                               past_lens,
                                               cache_interval,
                                               output_embeds,
                                               batch_size_in_tokens,
                                               hidden_size,
                                               kernel_size,
                                               num_blocks,
                                               seq_count,
                                               local_state.data());
    };

    if (state_precision == ov::element::f32) {
        dispatch_kernel(static_cast<float*>(conv_state_raw));
    } else if (state_precision == ov::element::f16) {
        dispatch_kernel(static_cast<ov::float16*>(conv_state_raw));
    } else if (state_precision == ov::element::bf16) {
        dispatch_kernel(static_cast<ov::bfloat16*>(conv_state_raw));
    } else {
        OPENVINO_ASSERT(false,
                        "PagedCausalConv1D: unsupported conv_state_table precision ",
                        state_precision,
                        ". Expected f32, f16, or bf16.");
    }
}

}  // namespace ov::intel_cpu::node
