// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_selective_ssm.h"

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
#include "nodes/executors/paged_selective_ssm_config.hpp"
#include "nodes/node_config.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/paged_selective_ssm.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu::node {

namespace {

using InputPortBinding = std::pair<int, size_t>;
constexpr auto output_port = output_port_index(PagedSelectiveSSMOutputPort::Output);

constexpr std::array input_port_bindings{
    InputPortBinding{ARG_PAGED_SSM_A, input_port_index(PagedSelectiveSSMInputPort::A)},
    InputPortBinding{ARG_PAGED_SSM_DT, input_port_index(PagedSelectiveSSMInputPort::TimeStep)},
    InputPortBinding{ARG_PAGED_SSM_B, input_port_index(PagedSelectiveSSMInputPort::InputProjection)},
    InputPortBinding{ARG_PAGED_SSM_X, input_port_index(PagedSelectiveSSMInputPort::Input)},
    InputPortBinding{ARG_PAGED_SSM_C, input_port_index(PagedSelectiveSSMInputPort::OutputProjection)},
    InputPortBinding{ARG_PAGED_SSM_STATE, input_port_index(PagedSelectiveSSMInputPort::State)},
    InputPortBinding{ARG_PAGED_SSM_SUBSEQUENCE_BEGINS, input_port_index(PagedSelectiveSSMInputPort::SubsequenceBegins)},
    InputPortBinding{ARG_PAGED_SSM_BLOCK_INDICES, input_port_index(PagedSelectiveSSMInputPort::BlockIndices)},
    InputPortBinding{ARG_PAGED_SSM_BLOCK_INDICES_BEGINS,
                     input_port_index(PagedSelectiveSSMInputPort::BlockIndicesBegins)},
    InputPortBinding{ARG_PAGED_SSM_NUM_PROCESSED_TOKENS,
                     input_port_index(PagedSelectiveSSMInputPort::NumProcessedTokens)},
    InputPortBinding{ARG_PAGED_SSM_CACHE_INTERVAL, input_port_index(PagedSelectiveSSMInputPort::CacheInterval)},
};

static_assert(input_port_bindings.size() == paged_ssm_input_count);
static_assert(paged_ssm_output_count == 1);

}  // namespace

PagedSelectiveSSM::PagedSelectiveSSM(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string error_message;
    if (!isSupportedOperation(op, error_message)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(error_message);
    }
}

void PagedSelectiveSSM::initSupportedPrimitiveDescriptors() {
    const auto data_precision = getOriginalInputPrecisionAtPort(input_port_index(PagedSelectiveSSMInputPort::Input));
    OPENVINO_ASSERT(any_of(data_precision, ov::element::f32, ov::element::f16, ov::element::bf16),
                    "PagedSelectiveSSM supports only f32/f16/bf16 data, got ",
                    data_precision,
                    ".");
    std::array<ov::element::Type, paged_ssm_input_count> input_precisions;
    for (size_t port = 0; port < paged_ssm_input_count; ++port) {
        input_precisions[port] = getOriginalInputPrecisionAtPort(port);
    }

    // BF16 precision enforcement preserves constant A in f32 while lowering the activation path. The executor uses
    // one DataT, selected by x, so regular CPU reorders convert the remaining computation inputs to that precision.
    for (const auto port : paged_ssm_computation_ports) {
        const auto port_index = input_port_index(port);
        OPENVINO_ASSERT(any_of(input_precisions[port_index], ov::element::f32, ov::element::f16, ov::element::bf16),
                        "PagedSelectiveSSM supports only f32/f16/bf16 computation inputs, got ",
                        input_precisions[port_index],
                        " at port ",
                        port_index,
                        ".");
        input_precisions[port_index] = data_precision;
    }
    const auto state_precision = input_precisions[input_port_index(PagedSelectiveSSMInputPort::State)];
    OPENVINO_ASSERT(any_of(state_precision, ov::element::f32, ov::element::f16, ov::element::bf16),
                    "PagedSelectiveSSM supports only f32/f16/bf16 state, got ",
                    state_precision,
                    ".");
    const auto index_precision = input_precisions[input_port_index(PagedSelectiveSSMInputPort::SubsequenceBegins)];
    OPENVINO_ASSERT(any_of(index_precision, ov::element::i32, ov::element::i64),
                    "PagedSelectiveSSM supports only i32/i64 metadata, got ",
                    index_precision,
                    ".");
    for (const auto port : paged_ssm_metadata_ports) {
        OPENVINO_ASSERT(input_precisions[input_port_index(port)] == index_precision,
                        "PagedSelectiveSSM requires all metadata inputs to have one precision.");
    }

    const auto& creators_map = BlockedDescCreator::getCommonCreators();
    MemoryDescArgs descs;
    for (const auto& [arg_id, port_id] : input_port_bindings) {
        descs[arg_id] = creators_map.at(LayoutType::ncsp)
                            ->createSharedDesc(input_precisions[port_id], getInputShapeAtPort(port_id));
    }
    descs[ARG_PAGED_SSM_OUT] =
        creators_map.at(LayoutType::ncsp)->createSharedDesc(data_precision, getOutputShapeAtPort(output_port));

    auto execution_context = std::make_shared<ExecutorContext>(context, getImplPriority(), privateWeightCache);
    m_factory = std::make_shared<ExecutorFactory<PagedSelectiveSSMAttrs>>(m_attrs, execution_context, descs);

    const auto node_descriptors_list = m_factory->getProperMemoryDescriptors(descs);
    for (const auto& node_descriptors : node_descriptors_list) {
        NodeConfig node_config;
        node_config.inConfs.resize(getParentEdges().size());
        for (const auto& [arg_id, port_id] : input_port_bindings) {
            if (node_descriptors.count(arg_id)) {
                node_config.inConfs[port_id] = PortConfig{node_descriptors.at(arg_id)};
            }
        }
        node_config.outConfs.emplace_back(node_descriptors.at(ARG_PAGED_SSM_OUT));
        supportedPrimitiveDescriptors.emplace_back(node_config, impl_desc_type::undef);
    }
}

void PagedSelectiveSSM::bindMemoryArguments() {
    for (const auto& [arg_id, port_id] : input_port_bindings) {
        m_memory[arg_id] = getSrcMemoryAtPort(port_id);
    }
    m_memory[ARG_PAGED_SSM_OUT] = getDstMemoryAtPort(output_port);
}

void PagedSelectiveSSM::createPrimitive() {
    bindMemoryArguments();

    m_executor = m_factory->make(m_memory);
    Node::createPrimitive();
    getSelectedPrimitiveDescriptor()->setImplementationType(m_executor->implType());
}

void PagedSelectiveSSM::prepareParams() {
    bindMemoryArguments();

    m_executor->update(m_memory);
    getSelectedPrimitiveDescriptor()->setImplementationType(m_executor->implType());
}

void PagedSelectiveSSM::execute([[maybe_unused]] const dnnl::stream& strm) {
    bindMemoryArguments();

    m_executor->execute(m_memory);
}

bool PagedSelectiveSSM::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                             std::string& error_message) noexcept {
    if (op == nullptr || !ov::is_type<ov::op::internal::PagedSelectiveSSM>(op)) {
        error_message = "Node is not an instance of ov::op::internal::PagedSelectiveSSM.";
        return false;
    }
    return true;
}

}  // namespace ov::intel_cpu::node
