// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transpose.h"

#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <vector>

#include "cpu_types.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "nodes/common/blocked_desc_creator.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_factory.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/transpose_config.hpp"
#include "nodes/node_config.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/transpose.hpp"
#include "shape_inference/custom/transpose.hpp"
#include "utils/general_utils.h"
using namespace dnnl;

namespace ov::intel_cpu::node {

bool Transpose::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (none_of(op->get_type_info(), ov::op::v1::Transpose::get_type_info_static())) {
            errorMessage = "Node is not an instance of the Transpose operation from opset1.";
            return false;
        }

        if (op->get_input_node_ptr(INPUT_ORDER_IDX)->get_type_info() != ov::op::v0::Constant::get_type_info_static()) {
            // TODO: Support parameterized Order input for dynamic shapes.
            errorMessage = "Constant expected as the second input for static shapes.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

Transpose::Transpose(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, TransposeShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    if (op->get_input_node_ptr(INPUT_ORDER_IDX)->get_type_info() == ov::op::v0::Constant::get_type_info_static()) {
        isInputOrderConst = true;
        order = ov::as_type<ov::op::v0::Constant>(op->get_input_node_ptr(INPUT_ORDER_IDX))->cast_vector<size_t>();

        if (order.empty()) {
            size_t rank = getInputShapeAtPort(INPUT_DATA_IDX).getRank();
            for (size_t i = 1LU; i <= rank; ++i) {
                order.emplace_back(rank - i);
            }
        }
    }

    attrs.permuteParams.order = order;
}

void Transpose::getSupportedDescriptors() {}

void Transpose::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    prec = getOriginalInputPrecisionAtPort(0);

    const auto& creatorsMap = BlockedDescCreator::getCommonCreators();

    NodeConfig config;
    config.inConfs.resize(2);
    config.outConfs.resize(1);
    config.inConfs[INPUT_DATA_IDX].inPlace(-1);
    config.inConfs[INPUT_DATA_IDX].constant(false);
    config.inConfs[INPUT_ORDER_IDX].constant(isInputOrderConst);
    config.inConfs[INPUT_ORDER_IDX].setMemDesc(
        creatorsMap.at(LayoutType::ncsp)->createSharedDesc(ov::element::i32, getInputShapeAtPort(INPUT_ORDER_IDX)));
    config.outConfs[0].inPlace(isOptimized ? 0 : -1);
    config.outConfs[0].constant(false);
    transpose_context = std::make_shared<ExecutorContext>(context, getImplPriority());

    auto supportedPrimitiveDescriptorsBuilder = [this](const NodeConfig& config) {
        supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown);
    };

    const auto& inputDataShape = getInputShapeAtPort(INPUT_DATA_IDX);
    const auto& outputDataShape = getOutputShapeAtPort(0);
    if (any_of(inputDataShape.getRank(), 4U, 5U)) {
        config.inConfs[0].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(prec, inputDataShape));
        config.outConfs[0].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(prec, outputDataShape));
        supportedPrimitiveDescriptorsBuilder(config);
#if defined(OPENVINO_ARCH_X86_64)
        const auto& srcDims = inputDataShape.getDims();
        if (srcDims[1] != Shape::UNDEFINED_DIM && srcDims[1] % 8 == 0) {
            config.inConfs[0].setMemDesc(creatorsMap.at(LayoutType::nCsp8c)->createSharedDesc(prec, inputDataShape));
            supportedPrimitiveDescriptorsBuilder(config);
        }

        if (srcDims[1] != Shape::UNDEFINED_DIM && srcDims[1] % 16 == 0) {
            config.inConfs[0].setMemDesc(creatorsMap.at(LayoutType::nCsp16c)->createSharedDesc(prec, inputDataShape));
            supportedPrimitiveDescriptorsBuilder(config);
        }
#endif  // OPENVINO_ARCH_X86_64
        if (any_of(prec, ov::element::f32, ov::element::f16, ov::element::i8, ov::element::u8, ov::element::bf16)) {
            config.inConfs[0].setMemDesc(creatorsMap.at(LayoutType::nspc)->createSharedDesc(prec, inputDataShape));
            config.outConfs[0].setMemDesc(creatorsMap.at(LayoutType::nspc)->createSharedDesc(prec, outputDataShape));
            supportedPrimitiveDescriptorsBuilder(config);
        }
    } else {
        // general plain case
        config.inConfs[0].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(prec, inputDataShape));
        config.outConfs[0].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(prec, outputDataShape));
        supportedPrimitiveDescriptorsBuilder(config);
    }
}

bool Transpose::neverExecute() const {
    return isOptimized || getSelectedPrimitiveDescriptor()->hasZeroInputDimsAtPort(0);
}

bool Transpose::isExecutable() const {
    return !isOptimized && !isInputTensorAtPortEmpty(0);
}

bool Transpose::needPrepareParams() const {
    if (isDynamicNode()) {
        return true;
    }
    return inputShapesModified();
}

void Transpose::prepareParams() {
    if (isOptimized) {
        return;
    }

    m_memory[ARG_SRC] = getSrcMemoryAtPort(INPUT_DATA_IDX);
    m_memory[ARG_DST] = getDstMemoryAtPort(0);
    CPU_NODE_ASSERT(execPtr, "Transpose executor was not created.");
    CPU_NODE_ASSERT(execPtr->update(m_memory), "Failed to update Transpose executor.");
}

void Transpose::createPrimitive() {
    if (isOptimized) {
        return;
    }

    auto dstMemPtr = getDstMemoryAtPort(0);
    auto srcMemPtr = getSrcMemoryAtPort(INPUT_DATA_IDX);
    CPU_NODE_ASSERT(dstMemPtr, "Destination memory is null.");
    CPU_NODE_ASSERT(srcMemPtr, "Input memory is null.");
    CPU_NODE_ASSERT(getSelectedPrimitiveDescriptor(), "Preferable primitive descriptor was not set.");

    attrs.permuteParams.data_size =
        getSelectedPrimitiveDescriptor()->getConfig().inConfs[0].getMemDesc()->getPrecision().size();
    m_memory[ARG_SRC] = srcMemPtr;
    m_memory[ARG_DST] = dstMemPtr;

    MemoryDescArgs descs{{ARG_SRC, srcMemPtr->getDescPtr()}, {ARG_DST, dstMemPtr->getDescPtr()}};
    auto factory = std::make_shared<ExecutorFactory<TransposeAttrs>>(attrs, transpose_context, descs);
    execPtr = factory->make(m_memory, false);

    Node::createPrimitive();
}

void Transpose::execute([[maybe_unused]] const dnnl::stream& strm) {
    if (isOptimized) {
        return;
    }

    CPU_NODE_ASSERT(execPtr, "Primitive was not created.");
    m_memory[ARG_SRC] = getSrcMemoryAtPort(INPUT_DATA_IDX);
    m_memory[ARG_DST] = getDstMemoryAtPort(0);
    execPtr->execute(m_memory);
}

void Transpose::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

bool Transpose::created() const {
    return getType() == Type::Transpose;
}

}  // namespace ov::intel_cpu::node
