// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "linear_attn.h"

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

LinearAttention::LinearAttention(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, InternalDynShapeInferFactory()) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
}

void LinearAttention::initSupportedPrimitiveDescriptors() {
    auto dataPrecision = getOriginalInputPrecisionAtPort(0);
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

void LinearAttention::createPrimitive() {
    return;
}

void LinearAttention::execute([[maybe_unused]] const dnnl::stream& strm) {
    auto orginInputNumber = getOriginalInputsNumber();
    std::vector<MemoryPtr> inputs(orginInputNumber);
    std::vector<MemoryPtr> outputs(2);

    for (size_t i = 0; i < orginInputNumber; i++) {
        inputs[i] = getSrcMemoryAtPort(i);
    }
    std::vector<VectorDims> output_dims = {inputs[0]->getStaticDims(), inputs[5]->getStaticDims()};
    redefineOutputMemory(output_dims);

    outputs[0] = getDstMemoryAtPort(0);
    outputs[1] = getDstMemoryAtPort(1);

    PlainTensor query(inputs[0]);
    PlainTensor key(inputs[1]);
    PlainTensor value(inputs[2]);
    PlainTensor beta(inputs[3]);
    PlainTensor g(inputs[4]);
    PlainTensor initial_states(outputs[0]);
    PlainTensor output(outputs[0]);
    PlainTensor output_hidden_states(inputs[1]);
    recurrent_linear_attn(query, key, value, beta, g, initial_states, output, output_hidden_states);
}

bool LinearAttention::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                          std::string& errorMessage) noexcept {
    return true;
}

}  // namespace ov::intel_cpu::node
