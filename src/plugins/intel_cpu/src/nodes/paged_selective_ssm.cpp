// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_selective_ssm.h"

#include <memory>
#include <string>
#include <vector>

#include "cpu_memory.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "nodes/executors/ref/selective_ssm_ref_kernel.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/paged_selective_ssm.hpp"
#include "shape_inference/shape_inference_cpu.hpp"

namespace ov::intel_cpu::node {

PagedSelectiveSSM::PagedSelectiveSSM(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
}

void PagedSelectiveSSM::initSupportedPrimitiveDescriptors() {
    std::vector<PortConfigurator> inPortConfigs;
    for (size_t i = 0; i < getParentEdges().size(); ++i) {
        inPortConfigs.emplace_back(LayoutType::ncsp,
                                   getOriginalInputPrecisionAtPort(i),
                                   getInputShapeAtPort(i),
                                   false,
                                   -1);
    }

    std::vector<PortConfigurator> outPortConfigs = {
        PortConfigurator{LayoutType::ncsp, getOriginalOutputPrecisionAtPort(0), getOutputShapeAtPort(0), false, -1}};
    addSupportedPrimDesc(inPortConfigs, outPortConfigs, impl_desc_type::ref_any);
}

void PagedSelectiveSSM::execute([[maybe_unused]] const dnnl::stream& strm) {
    PlainTensor A(getSrcMemoryAtPort(0));
    PlainTensor dt(getSrcMemoryAtPort(1));
    PlainTensor B(getSrcMemoryAtPort(2));
    PlainTensor x(getSrcMemoryAtPort(3));
    PlainTensor C(getSrcMemoryAtPort(4));
    PlainTensor recurrent_state_table(getSrcMemoryAtPort(5));
    PlainTensor subsequence_begins(getSrcMemoryAtPort(6));
    PlainTensor block_indices(getSrcMemoryAtPort(7));
    PlainTensor block_indices_begins(getSrcMemoryAtPort(8));
    PlainTensor num_processed_tokens(getSrcMemoryAtPort(9));
    PlainTensor cache_interval(getSrcMemoryAtPort(10));
    PlainTensor output(getDstMemoryAtPort(0));

    detail::paged_selective_ssm_reference(A,
                                          dt,
                                          B,
                                          x,
                                          C,
                                          recurrent_state_table,
                                          subsequence_begins,
                                          block_indices,
                                          block_indices_begins,
                                          num_processed_tokens,
                                          cache_interval,
                                          output);
}

bool PagedSelectiveSSM::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                             std::string& errorMessage) noexcept {
    if (op == nullptr || !ov::is_type<ov::op::internal::PagedSelectiveSSM>(op)) {
        errorMessage = "Node is not an instance of ov::op::internal::PagedSelectiveSSM.";
        return false;
    }
    return true;
}

}  // namespace ov::intel_cpu::node
