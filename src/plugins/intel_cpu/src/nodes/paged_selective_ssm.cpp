// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_selective_ssm.h"

#include <memory>
#include <string>
#include <vector>

#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "nodes/common/blocked_desc_creator.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_factory.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/paged_selective_ssm_config.hpp"
#include "nodes/node_config.h"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/paged_selective_ssm.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu::node {

PagedSelectiveSSM::PagedSelectiveSSM(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    m_atoi[ARG_PAGED_SSM_A] = 0;
    m_atoi[ARG_PAGED_SSM_DT] = 1;
    m_atoi[ARG_PAGED_SSM_B] = 2;
    m_atoi[ARG_PAGED_SSM_X] = 3;
    m_atoi[ARG_PAGED_SSM_C] = 4;
    m_atoi[ARG_PAGED_SSM_STATE] = 5;
    m_atoi[ARG_PAGED_SSM_SUBSEQUENCE_BEGINS] = 6;
    m_atoi[ARG_PAGED_SSM_BLOCK_INDICES] = 7;
    m_atoi[ARG_PAGED_SSM_BLOCK_INDICES_BEGINS] = 8;
    m_atoi[ARG_PAGED_SSM_NUM_PROCESSED_TOKENS] = 9;
    m_atoi[ARG_PAGED_SSM_CACHE_INTERVAL] = 10;
}

void PagedSelectiveSSM::initSupportedPrimitiveDescriptors() {
    const auto data_precision = getOriginalInputPrecisionAtPort(0);
    OPENVINO_ASSERT(any_of(data_precision, ov::element::f32, ov::element::f16, ov::element::bf16),
                    "PagedSelectiveSSM supports only f32/f16/bf16 data, got ",
                    data_precision,
                    ".");
    for (size_t port = 1; port <= 5; ++port) {
        OPENVINO_ASSERT(getOriginalInputPrecisionAtPort(port) == data_precision,
                        "PagedSelectiveSSM requires one data precision on ports 0..5.");
    }
    const auto index_precision = getOriginalInputPrecisionAtPort(6);
    OPENVINO_ASSERT(any_of(index_precision, ov::element::i32, ov::element::i64),
                    "PagedSelectiveSSM supports only i32/i64 metadata, got ",
                    index_precision,
                    ".");
    for (size_t port = 7; port <= 10; ++port) {
        OPENVINO_ASSERT(getOriginalInputPrecisionAtPort(port) == index_precision,
                        "PagedSelectiveSSM requires one metadata precision on ports 6..10.");
    }

    const auto& creators_map = BlockedDescCreator::getCommonCreators();
    MemoryDescArgs descs;
    for (const auto& [arg_id, port_id] : m_atoi) {
        descs[arg_id] = creators_map.at(LayoutType::ncsp)
                            ->createSharedDesc(getOriginalInputPrecisionAtPort(port_id), getInputShapeAtPort(port_id));
    }
    descs[ARG_PAGED_SSM_OUT] = creators_map.at(LayoutType::ncsp)
                                   ->createSharedDesc(getOriginalOutputPrecisionAtPort(0), getOutputShapeAtPort(0));

    auto execution_context = std::make_shared<ExecutorContext>(context, getImplPriority(), privateWeightCache);
    m_factory = std::make_shared<ExecutorFactory<PagedSelectiveSSMAttrs>>(m_attrs, execution_context, descs);

    const auto node_descriptors_list = m_factory->getProperMemoryDescriptors(descs);
    for (const auto& node_descriptors : node_descriptors_list) {
        NodeConfig node_config;
        node_config.inConfs.resize(getParentEdges().size());
        for (const auto& [arg_id, port_id] : m_atoi) {
            if (node_descriptors.count(arg_id)) {
                node_config.inConfs[port_id] = PortConfig{node_descriptors.at(arg_id)};
            }
        }
        node_config.outConfs.emplace_back(node_descriptors.at(ARG_PAGED_SSM_OUT));
        supportedPrimitiveDescriptors.emplace_back(node_config, impl_desc_type::undef);
    }
}

void PagedSelectiveSSM::createPrimitive() {
    for (const auto& [arg_id, port_id] : m_atoi) {
        m_memory[arg_id] = getSrcMemoryAtPort(port_id);
    }
    m_memory[ARG_PAGED_SSM_OUT] = getDstMemoryAtPort(0);

    m_executor = m_factory->make(m_memory);
    Node::createPrimitive();
    getSelectedPrimitiveDescriptor()->setImplementationType(m_executor->implType());
}

void PagedSelectiveSSM::prepareParams() {
    for (const auto& [arg_id, port_id] : m_atoi) {
        m_memory[arg_id] = getSrcMemoryAtPort(port_id);
    }
    m_memory[ARG_PAGED_SSM_OUT] = getDstMemoryAtPort(0);

    m_executor->update(m_memory);
    getSelectedPrimitiveDescriptor()->setImplementationType(m_executor->implType());
}

void PagedSelectiveSSM::execute([[maybe_unused]] const dnnl::stream& strm) {
    for (const auto& [arg_id, port_id] : m_atoi) {
        m_memory[arg_id] = getSrcMemoryAtPort(port_id);
    }
    m_memory[ARG_PAGED_SSM_OUT] = getDstMemoryAtPort(0);

    m_executor->execute(m_memory);
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
