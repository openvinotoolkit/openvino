// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transpose.h"

#include "nodes/common/reorder_prim.h"
#include <openvino/op/constant.hpp>
#include <openvino/op/transpose.hpp>

using namespace dnnl;
using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {

bool Transpose::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (op->get_type_info() != op::v1::Transpose::get_type_info_static()) {
            errorMessage = "Node is not an instance of the Transpose operation from opset1.";
            return false;
        }

        if (op->get_input_node_ptr(INPUT_ORDER_IDX)->get_type_info() != op::v0::Constant::get_type_info_static()) {
            // TODO: Support parameterized Order input for dynamic shapes.
            errorMessage = "Constant expected as the second input for static shapes.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

namespace {
class TransposeDynShapeInfer : public ShapeInferEmptyPads {
public:
    TransposeDynShapeInfer() = default;
    Result infer(
            const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
            const std::unordered_map<size_t, MemoryPtr>& data_dependency) override {
        IE_THROW(NotImplemented) << "TODO: Support parameterized Order input for dynamic shapes.";
    }
    port_mask_t get_port_mask() const override {
        return EMPTY_PORT_MASK;
    }
private:
};

class TransposeShapeInfer : public ShapeInferEmptyPads {
public:
    TransposeShapeInfer(const size_t& out_rank, const std::vector<size_t>& axes_vec)
            : m_out_rank(out_rank), m_axes_vec(axes_vec), m_outputShape(out_rank, 1), m_needReverse(axes_vec.empty()) {}

    Result infer(
            const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
            const std::unordered_map<size_t, MemoryPtr>& data_dependency) override {
        const VectorDims& shapeIn = input_shapes[0].get();
        if (m_needReverse) {
            for (size_t i = 0; i < m_out_rank; ++i) {
                m_outputShape[i] = shapeIn[m_out_rank - 1 - i];
            }
        } else {
            for (size_t i = 0; i < m_out_rank; ++i) {
                m_outputShape[i] = shapeIn[m_axes_vec[i]];
            }
        }
        return {{m_outputShape}, ShapeInferStatus::success};
    }

    port_mask_t get_port_mask() const override {
        return EMPTY_PORT_MASK;
    }

private:
    const size_t m_out_rank;
    const std::vector<size_t> m_axes_vec;
    VectorDims m_outputShape;
    const bool m_needReverse;
};

class TransposeShapeInferFactory : public ShapeInferFactory {
public:
    TransposeShapeInferFactory(const std::shared_ptr<ov::Node>& op) : m_op(op) {}
    ShapeInferPtr makeShapeInfer() const override {
        if (const auto order = ov::as_type_ptr<const op::v0::Constant>(m_op->get_input_node_shared_ptr(op::v1::Transpose::ORDER))) {
            const auto axes_vec = order->cast_vector<size_t>();
            return std::make_shared<TransposeShapeInfer>(m_op->get_output_partial_shape(0).rank().get_length(), axes_vec);
        } else {
            return std::make_shared<TransposeDynShapeInfer>();
        }
    }

private:
    const std::shared_ptr<ov::Node> m_op;
};
} // namespace

Transpose::Transpose(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
        : Node(op, context, TransposeShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    if (auto inputOrder = ov::as_type<op::v0::Constant>(op->get_input_node_ptr(INPUT_ORDER_IDX))) {
        isInputOrderConst = true;
        order = inputOrder->cast_vector<size_t>();

        if (order.empty()) {
            size_t rank = getInputShapeAtPort(INPUT_DATA_IDX).getRank();
            for (size_t i = 1lu; i <= rank; ++i) {
                order.emplace_back(rank - i);
            }
        }
    }
}

void Transpose::getSupportedDescriptors() {
}

void Transpose::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    const auto& dataPrc = getOriginalInputPrecisionAtPort(0);
    auto orderPrc = getOriginalInputPrecisionAtPort(1);
    if (!one_of(orderPrc, Precision::I32, Precision::I64)) {
        orderPrc = Precision::I32;
    }

    auto& creatorsMap = BlockedDescCreator::getCommonCreators();

    NodeConfig config;
    config.inConfs.resize(2);
    config.outConfs.resize(1);
    config.inConfs[INPUT_DATA_IDX].inPlace(-1);
    config.inConfs[INPUT_DATA_IDX].constant(false);
    config.inConfs[INPUT_ORDER_IDX].constant(isInputOrderConst);
    config.inConfs[INPUT_ORDER_IDX].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
            orderPrc, getInputShapeAtPort(INPUT_ORDER_IDX)));
    config.outConfs[0].inPlace(-1);
    config.outConfs[0].constant(false);
    transpose_context = std::make_shared<ExecutorContext>(context, getImplPriority());

    auto supportedPrimitiveDescriptorsBuilder = [this](NodeConfig config, TransposeParams transposeParams) {
        std::vector<MemoryDescPtr> srcMemoryDescs;
        for (size_t i = 0; i < config.inConfs.size(); i++) {
            srcMemoryDescs.push_back(config.inConfs[i].getMemDesc());
        }
        std::vector<MemoryDescPtr> dstMemoryDescs;
        for (size_t i = 0; i < config.outConfs.size(); i++) {
            dstMemoryDescs.push_back(config.outConfs[i].getMemDesc());
        }
        auto factory = std::make_shared<TransposeExecutorFactory>(transposeParams, srcMemoryDescs, dstMemoryDescs, transpose_context);
        supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown, factory});
    };

    const auto& inputDataShape = getInputShapeAtPort(INPUT_DATA_IDX);
    const auto& outputDataShape = getOutputShapeAtPort(0);
    if (inputDataShape.getRank() == 4 || inputDataShape.getRank() == 5) {
        config.inConfs[0].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(dataPrc, inputDataShape));
        config.outConfs[0].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(dataPrc, outputDataShape));
        supportedPrimitiveDescriptorsBuilder(config, transposeParams);
#if defined(OPENVINO_ARCH_X86_64)
        const auto& srcDims = inputDataShape.getDims();
        if (srcDims[1] != Shape::UNDEFINED_DIM && srcDims[1] % 8 == 0) {
            config.inConfs[0].setMemDesc(creatorsMap.at(LayoutType::nCsp8c)->createSharedDesc(dataPrc, inputDataShape));
            supportedPrimitiveDescriptorsBuilder(config, transposeParams);
        }

        if (srcDims[1] != Shape::UNDEFINED_DIM && srcDims[1] % 16 == 0) {
            config.inConfs[0].setMemDesc(creatorsMap.at(LayoutType::nCsp16c)->createSharedDesc(dataPrc, inputDataShape));
            supportedPrimitiveDescriptorsBuilder(config, transposeParams);
        }
#endif // OPENVINO_ARCH_X86_64
        if (one_of(dataPrc, Precision::FP32, Precision::I8, Precision::U8)) {
            config.inConfs[0].setMemDesc(creatorsMap.at(LayoutType::nspc)->createSharedDesc(dataPrc, inputDataShape));
            config.outConfs[0].setMemDesc(creatorsMap.at(LayoutType::nspc)->createSharedDesc(dataPrc, outputDataShape));
            supportedPrimitiveDescriptorsBuilder(config, transposeParams);
        }
    } else {
        // general plain case
        config.inConfs[0].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(dataPrc, inputDataShape));
        config.outConfs[0].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(dataPrc, outputDataShape));
        supportedPrimitiveDescriptorsBuilder(config, transposeParams);
    }
}

bool Transpose::isExecutable() const {
    return !isInputTensorAtPortEmpty(0);
}

bool Transpose::needPrepareParams() const {
    return inputShapesModified();
}

void Transpose::prepareParams() {
    if (performAsReorder) {
        //  Transpose(order={0,3,1,2}) can be performed as Reorder(acdb=>abcd)
        auto srcMemPtr = getParentEdgeAt(INPUT_DATA_IDX)->getMemoryPtr();
        auto dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
        auto dstDesc = dstMemPtr->getDescWithType<DnnlMemoryDesc>()->getDnnlDesc();
        auto srcDesc = dnnl::memory::desc(dstDesc.get_dims(), dstDesc.get_data_type(), memory::format_tag::acdb);
        auto result = getReorderPrim(context->getParamsCache(), getEngine(), srcDesc, dstDesc);
        if (!result) {
            THROW_CPU_NODE_ERR << ". Reorder primitive descriptor was not found.";
        }
        prim = result;

        getSelectedPrimitiveDescriptor()->setImplementationType(
            parse_impl_name(DnnlExtensionUtils::query_impl_info_str(prim.get_primitive_desc())));

        primArgs = {{DNNL_ARG_SRC, srcMemPtr->getPrimitive()}, {DNNL_ARG_DST, dstMemPtr->getPrimitive()}};
#ifdef CPU_DEBUG_CAPS
        if (prim) {
            auto pd = prim.get_primitive_desc();
            DEBUG_LOG("verbose##", getName(), "##", DnnlExtensionUtils::query_pd_info(pd), "\n");
        }
#endif
        return;
    }

    auto srcDesc = getParentEdgeAt(INPUT_DATA_IDX)->getMemory().getDescWithType<BlockedMemoryDesc>();
    transposeParams.permuteParams.src_block_dims = srcDesc->getBlockDims();
    auto dstDesc = getChildEdgeAt(0)->getMemory().getDescWithType<BlockedMemoryDesc>();
    transposeParams.permuteParams.dst_block_dims = dstDesc->getBlockDims();

    if (!isInputOrderConst) {
        auto mem = getParentEdgeAt(0)->getMemoryPtr();
        auto orderLen = mem->getSize();
        if (mem->getDesc().getPrecision() == Precision::I64) {
            auto orderPtr = reinterpret_cast<const int64_t*>(mem->getData());
            transposeParams.permuteParams.order.assign(orderPtr, orderPtr + orderLen);
        } else {
            auto orderPtr = reinterpret_cast<const int32_t*>(mem->getData());
            transposeParams.permuteParams.order.assign(orderPtr, orderPtr + orderLen);
        }
    }

    auto engine = getEngine();
    auto builder = [&srcDesc, &dstDesc, this](const PermuteParams& key) -> std::shared_ptr<TransposeExecutor> {
        dnnl::primitive_attr attr;
        auto selectedPD = getSelectedPrimitiveDescriptor();
        auto jitExec = selectedPD->getExecutorFactoryAs<TransposeExecutorFactory>()->makeExecutor(transposeParams,
                                                                                                  {srcDesc},
                                                                                                  {dstDesc},
                                                                                                  attr);
        return jitExec;
    };

    auto cache = context->getParamsCache();
    auto result = cache->getOrCreate(transposeParams.permuteParams, builder);

    if (!result.first) {
        THROW_CPU_NODE_ERR << ". Primitive descriptor was not found.";
    }

    execPtr = result.first;
}

void Transpose::createPrimitive() {
    auto dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto srcMemPtr = getParentEdgeAt(INPUT_DATA_IDX)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->isAllocated())
        THROW_CPU_NODE_ERR << ". Destination memory was not allocated.";
    if (!srcMemPtr || !srcMemPtr->isAllocated())
        THROW_CPU_NODE_ERR << ". Input memory was not allocated.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_CPU_NODE_ERR << ". Preferable primitive descriptor was not set.";

    if (getParentEdgeAt(INPUT_DATA_IDX)->getMemory().getDesc().hasLayoutType(LayoutType::ncsp) &&
        getChildEdgeAt(0)->getMemory().getDesc().hasLayoutType(LayoutType::ncsp) &&
        order == std::vector<size_t>{0, 3, 1, 2}) {
        performAsReorder = true;
    }

    if (!performAsReorder) {
        transposeParams.permuteParams.data_size = getSelectedPrimitiveDescriptor()->getConfig().inConfs[0].getMemDesc()->getPrecision().size();
        if (isInputOrderConst)
            transposeParams.permuteParams.order = order;
        auto srcDesc = getParentEdgeAt(INPUT_DATA_IDX)->getMemory().getDescWithType<BlockedMemoryDesc>();
        transposeParams.permuteParams.src_block_order = srcDesc->getOrder();
        auto dstDesc = getChildEdgeAt(0)->getMemory().getDescWithType<BlockedMemoryDesc>();
        transposeParams.permuteParams.dst_block_order = dstDesc->getOrder();
    }

    if (inputShapesDefined() && isExecutable()) {
        prepareParams();
        updateLastInputDims();
    }
}

void Transpose::execute(dnnl::stream strm) {
    if (prim) {
        prim.execute(strm, primArgs);
    } else if (execPtr) {
        auto dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
        auto srcMemPtr = getParentEdgeAt(INPUT_DATA_IDX)->getMemoryPtr();

        int MB = srcMemPtr->getStaticDims()[0];

        execPtr->exec({srcMemPtr}, {dstMemPtr}, MB);
    } else {
        THROW_CPU_NODE_ERR << "could not be executed. Primitive was not created.";
    }
}

void Transpose::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

bool Transpose::created() const {
    return getType() == Type::Transpose;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
