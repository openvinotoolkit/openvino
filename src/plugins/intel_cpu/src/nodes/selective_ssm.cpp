// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "selective_ssm.h"

#include <array>
#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <utility>
#include <vector>

#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "nodes/common/blocked_desc_creator.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_factory.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/selective_ssm_config.hpp"
#include "nodes/node_config.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/selective_ssm.hpp"
#include "shape_inference/shape_inference_cpu.hpp"

namespace ov::intel_cpu::node {

namespace {

using InputPortBinding = std::pair<int, size_t>;

constexpr std::array input_port_bindings{
    InputPortBinding{ARG_SSM_A, 0},
    InputPortBinding{ARG_SSM_DT, 1},
    InputPortBinding{ARG_SSM_B, 2},
    InputPortBinding{ARG_SSM_X, 3},
    InputPortBinding{ARG_SSM_C, 4},
    InputPortBinding{ARG_SSM_STATE, 5},
};

}  // namespace

SelectiveSSM::SelectiveSSM(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string error_message;
    if (!isSupportedOperation(op, error_message)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(error_message);
    }
}

void SelectiveSSM::initSupportedPrimitiveDescriptors() {
    const auto& creators_map = BlockedDescCreator::getCommonCreators();

    MemoryDescArgs descs;
    for (const auto& [arg_id, port_id] : input_port_bindings) {
        descs[arg_id] = creators_map.at(LayoutType::ncsp)
                            ->createSharedDesc(getOriginalInputPrecisionAtPort(port_id), getInputShapeAtPort(port_id));
    }
    descs[ARG_SSM_OUT] = creators_map.at(LayoutType::ncsp)
                             ->createSharedDesc(getOriginalOutputPrecisionAtPort(0), getOutputShapeAtPort(0));
    descs[ARG_SSM_OUT_STATE] = creators_map.at(LayoutType::ncsp)
                                   ->createSharedDesc(getOriginalOutputPrecisionAtPort(1), getOutputShapeAtPort(1));

    auto execution_context = std::make_shared<ExecutorContext>(context, getImplPriority(), privateWeightCache);
    m_factory = std::make_shared<ExecutorFactory<SelectiveSSMAttrs>>(m_attrs, execution_context, descs);

    const auto node_descriptors_list = m_factory->getProperMemoryDescriptors(descs);
    for (const auto& node_descriptors : node_descriptors_list) {
        NodeConfig node_config;
        node_config.inConfs.resize(getParentEdges().size());

        for (const auto& [arg_id, port_id] : input_port_bindings) {
            if (node_descriptors.count(arg_id)) {
                node_config.inConfs[port_id] = PortConfig{node_descriptors.at(arg_id)};
            }
        }

        node_config.outConfs.emplace_back(node_descriptors.at(ARG_SSM_OUT));
        node_config.outConfs.emplace_back(node_descriptors.at(ARG_SSM_OUT_STATE));
        supportedPrimitiveDescriptors.emplace_back(node_config, impl_desc_type::undef);
    }
}

void SelectiveSSM::bindMemoryArguments() {
    for (const auto& [arg_id, port_id] : input_port_bindings) {
        m_memory[arg_id] = getSrcMemoryAtPort(port_id);
    }
    m_memory[ARG_SSM_OUT] = getDstMemoryAtPort(0);
    m_memory[ARG_SSM_OUT_STATE] = getDstMemoryAtPort(1);
}

void SelectiveSSM::createPrimitive() {
    bindMemoryArguments();

    m_executor = m_factory->make(m_memory);
    Node::createPrimitive();
    getSelectedPrimitiveDescriptor()->setImplementationType(m_executor->implType());
}

void SelectiveSSM::prepareParams() {
    bindMemoryArguments();

    m_executor->update(m_memory);
    getSelectedPrimitiveDescriptor()->setImplementationType(m_executor->implType());
}

void SelectiveSSM::execute([[maybe_unused]] const dnnl::stream& strm) {
    bindMemoryArguments();

    m_executor->execute(m_memory);
}

bool SelectiveSSM::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                        std::string& error_message) noexcept {
    if (op == nullptr || !ov::is_type<ov::op::internal::SelectiveSSM>(op)) {
        error_message = "Node is not an instance of ov::op::internal::SelectiveSSM.";
        return false;
    }
    return true;
}

}  // namespace ov::intel_cpu::node
