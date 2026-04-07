// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_causal_conv1d.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "cpu_types.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "nodes/kernels/paged_causal_conv1d.hpp"
#include "nodes/node_config.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"
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
    std::vector<PortConfigurator> input_configs;
    input_configs.reserve(getParentEdges().size());
    input_configs.emplace_back(LayoutType::ncsp, data_precision, getInputShapeAtPort(0), false, -1);
    input_configs.emplace_back(LayoutType::ncsp, data_precision, getInputShapeAtPort(1), false, -1);
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
    OPENVINO_ASSERT(getOriginalInputPrecisionAtPort(0) == ov::element::f32,
                    "PagedCausalConv1D supports only f32 input_embeds precision in CPU plugin.");
    OPENVINO_ASSERT(getOriginalInputPrecisionAtPort(1) == ov::element::f32,
                    "PagedCausalConv1D supports only f32 conv_state_table precision in CPU plugin.");
    OPENVINO_ASSERT(getOriginalInputPrecisionAtPort(2) == ov::element::f32,
                    "PagedCausalConv1D supports only f32 conv_weight precision in CPU plugin.");
    OPENVINO_ASSERT(getOriginalInputPrecisionAtPort(3) == ov::element::f32,
                    "PagedCausalConv1D supports only f32 conv_bias precision in CPU plugin.");

    const auto input_embeds_shape = getInputShapeAtPort(0).getDims();
    const auto state_table_shape = getInputShapeAtPort(1).getDims();
    const auto weight_shape = getInputShapeAtPort(2).getDims();
    const auto bias_shape = getInputShapeAtPort(3).getDims();

    OPENVINO_ASSERT(input_embeds_shape.size() == 2,
                    "PagedCausalConv1D expects input_embeds rank 2, got ",
                    input_embeds_shape.size());
    OPENVINO_ASSERT(state_table_shape.size() == 3,
                    "PagedCausalConv1D expects conv_state_table rank 3, got ",
                    state_table_shape.size());
    OPENVINO_ASSERT(weight_shape.size() == 3,
                    "PagedCausalConv1D expects conv_weight rank 3, got ",
                    weight_shape.size());

    const size_t batch_size_in_tokens = input_embeds_shape[0];
    const size_t hidden_size = input_embeds_shape[1];
    const size_t num_blocks = state_table_shape[0];
    const size_t state_hidden_size = state_table_shape[1];
    const size_t kernel_size = state_table_shape[2];

    OPENVINO_ASSERT(hidden_size == state_hidden_size,
                    "PagedCausalConv1D expects hidden size match between input_embeds and conv_state_table. Got ",
                    hidden_size,
                    " and ",
                    state_hidden_size);
    OPENVINO_ASSERT(weight_shape[0] == hidden_size,
                    "PagedCausalConv1D expects conv_weight out_channels equal hidden_size. Got ",
                    weight_shape[0],
                    " and ",
                    hidden_size);
    OPENVINO_ASSERT(weight_shape[2] == kernel_size,
                    "PagedCausalConv1D expects conv_weight kernel size equal conv_state_table kernel size. Got ",
                    weight_shape[2],
                    " and ",
                    kernel_size);
    OPENVINO_ASSERT(bias_shape.size() == 1,
                    "PagedCausalConv1D expects conv_bias rank 1, got ",
                    bias_shape.size());

    const bool has_bias = bias_shape[0] != 0;
    OPENVINO_ASSERT(!has_bias || bias_shape[0] == hidden_size,
                    "PagedCausalConv1D expects conv_bias shape[0] to be 0 (optional) or hidden_size. Got ",
                    bias_shape[0],
                    " and hidden_size=",
                    hidden_size);

    const auto* input_embeds = getSrcDataAtPortAs<const float>(0);
    auto* conv_state_table = getSrcMemoryAtPort(1)->getDataAs<float>();
    const auto* conv_weight = getSrcDataAtPortAs<const float>(2);
    const float* conv_bias = has_bias ? getSrcDataAtPortAs<const float>(3) : nullptr;
    const auto* subsequence_begins = getSrcDataAtPortAs<const int32_t>(4);
    const auto* block_indices = getSrcDataAtPortAs<const int32_t>(5);
    const auto* block_indices_begins = getSrcDataAtPortAs<const int32_t>(6);
    const auto* past_lens = getSrcDataAtPortAs<const int32_t>(7);
    const auto* cache_interval = getSrcDataAtPortAs<const int32_t>(8);

    auto* output_embeds = getDstDataAtPortAs<float>(0);

    const auto& subseq_shape = getInputShapeAtPort(4).getDims();
    OPENVINO_ASSERT(!subseq_shape.empty(), "PagedCausalConv1D expects non-empty subsequence_begins input.");
    OPENVINO_ASSERT(subseq_shape[0] >= 1, "PagedCausalConv1D expects subsequence_begins shape[0] >= 1.");

    const size_t seq_count = subseq_shape[0] - 1;
    const size_t state_stride = hidden_size * kernel_size;
    std::vector<float> local_state(state_stride);

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
}

}  // namespace ov::intel_cpu::node
