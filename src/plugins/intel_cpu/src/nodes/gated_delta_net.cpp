// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gated_delta_net.h"

#include <common/utils.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <vector>

#include "config.h"
#include "cpu_memory.h"
#include "cpu_types.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "nodes/common/blocked_desc_creator.h"
#include "nodes/kernels/scaled_attn/executor_pa_common.hpp"
#include "nodes/node_config.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "shape_inference/shape_inference_internal_dyn.hpp"
#include "transformations/utils/utils.hpp"
#include "utils/general_utils.h"
#include "kernels/linear_attn/recurrent_linear_attn.hpp"

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
}

void GatedDeltaNet::initSupportedPrimitiveDescriptors() {
    auto dataPrecision = ov::element::f32;
    std::vector<PortConfigurator> inPortConfigs;
    for (size_t i = 0; i < getParentEdges().size(); ++i) {
        inPortConfigs.emplace_back(LayoutType::ncsp, dataPrecision, getInputShapeAtPort(i), false, -1);
    }
    std::vector<PortConfigurator> outPortConfigs = {
        PortConfigurator{LayoutType::ncsp, dataPrecision, getOutputShapeAtPort(0), false, -1},
        PortConfigurator{LayoutType::ncsp, dataPrecision, getOutputShapeAtPort(1), false, -1}
    };
    addSupportedPrimDesc(inPortConfigs, outPortConfigs, impl_desc_type::ref_any);
}

void GatedDeltaNet::createPrimitive() {
    return;
}

void GatedDeltaNet::execute([[maybe_unused]] const dnnl::stream& strm) {
    auto orginInputNumber = getOriginalInputsNumber();
    std::vector<MemoryPtr> inputs(orginInputNumber);
    std::vector<MemoryPtr> outputs(2);

    for (size_t i = 0; i < orginInputNumber; i++) {
        inputs[i] = getSrcMemoryAtPort(i);
    }
    std::vector<VectorDims> output_dims = {inputs[0]->getStaticDims(), inputs[3]->getStaticDims()};
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
    recurrent_linear_attn(query, key, value, recurrent_state, gate, beta, output_attn, output_recurrent_state);
}

bool GatedDeltaNet::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                          std::string& errorMessage) noexcept {
    return true;
}

}  // namespace ov::intel_cpu::node
