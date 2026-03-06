// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gated_delta_net.h"

#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <vector>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "graph_context.h"
#include "kernels/linear_attn/recurrent_linear_attn.hpp"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/gated_delta_net.hpp"
#include "shape_inference/shape_inference_internal_dyn.hpp"
#include "utils/plain_tensor.hpp"

using namespace ov::Extensions::Cpu;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace ov::Extensions::Cpu::XARCH;

namespace ov::intel_cpu::node {

GatedDeltaNet::GatedDeltaNet(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, InternalDynShapeInferFactory()) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    const auto& gdn = ov::as_type_ptr<ov::op::GatedDeltaNet>(op);
    fuse_q_scale = gdn->get_config().fuse_q_scale;
    fuse_qk_l2norm = gdn->get_config().fuse_qk_l2norm;
}

void GatedDeltaNet::initSupportedPrimitiveDescriptors() {
    auto dataPrecision = ov::element::f32;
    std::vector<PortConfigurator> inPortConfigs;
    for (size_t i = 0; i < getParentEdges().size(); ++i) {
        inPortConfigs.emplace_back(LayoutType::ncsp, dataPrecision, getInputShapeAtPort(i), false, -1);
    }
    std::vector<PortConfigurator> outPortConfigs = {
        PortConfigurator{LayoutType::ncsp, dataPrecision, getOutputShapeAtPort(0), false, -1},
        PortConfigurator{LayoutType::ncsp, dataPrecision, getOutputShapeAtPort(1), false, -1}};
    addSupportedPrimDesc(inPortConfigs, outPortConfigs, impl_desc_type::ref_any);
}

void GatedDeltaNet::execute([[maybe_unused]] const dnnl::stream& strm) {
    auto originalInputNumber = getOriginalInputsNumber();
    std::vector<MemoryPtr> inputs(originalInputNumber);
    std::vector<MemoryPtr> outputs(2);

    for (size_t i = 0; i < originalInputNumber; i++) {
        inputs[i] = getSrcMemoryAtPort(i);
    }
    std::vector<VectorDims> output_dims = {inputs[2]->getStaticDims(), inputs[3]->getStaticDims()};
    redefineOutputMemory(output_dims);

    outputs[0] = getDstMemoryAtPort(0);
    outputs[1] = getDstMemoryAtPort(1);

    PlainTensor query(inputs[0]);
    PlainTensor key(inputs[1]);
    PlainTensor value(inputs[2]);
    PlainTensor recurrent_state(inputs[3]);
    PlainTensor gate(inputs[4]);
    PlainTensor beta(inputs[5]);
    PlainTensor output_attn(outputs[0]);
    PlainTensor output_recurrent_state(outputs[1]);
    // q, k, h per (B, H, V)
    const auto& q_dims = inputs[0]->getStaticDims();
    const auto& v_dims = inputs[2]->getStaticDims();
    const size_t B = q_dims[0];
    const size_t H = q_dims[2];
    const size_t K = q_dims[3];
    const size_t V = v_dims[3];
    temp_buffer.resize<float>({B * H * V * 3 * K});
    recurrent_linear_attn(query,
                          key,
                          value,
                          recurrent_state,
                          gate,
                          beta,
                          fuse_qk_l2norm,
                          fuse_q_scale,
                          output_attn,
                          output_recurrent_state,
                          temp_buffer);
}

bool GatedDeltaNet::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                         std::string& errorMessage) noexcept {
    if (!ov::is_type<ov::op::GatedDeltaNet>(op)) {
        errorMessage = "Node is not an instance of ov::op::GatedDeltaNet.";
        return false;
    }
    return true;
}

}  // namespace ov::intel_cpu::node
