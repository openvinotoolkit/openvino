// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <optional>

#include "acl_fullyconnected.hpp"
#include "acl_utils.hpp"
#include "common/primitive_desc_iface.hpp"
#include "cpu/acl/acl_utils.hpp"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "nodes/common/cpu_convert.h"
#include "nodes/common/cpu_memcpy.h"
#include "nodes/common/reorder_prim.h"
#include "nodes/convert.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "utils/cpu_utils.hpp"
#include "utils/debug_capabilities.h"

namespace ov::intel_cpu {

VectorDims acl_fc_executor::makeDummyInputDims(const Shape& inShape, const Shape& wShape) {
    const auto& weightDims = wShape.getStaticDims();

    auto inMinDims = inShape.getMinDims();
    auto inMaxDims = inShape.getMaxDims();
    inMinDims.back() = weightDims.back();
    inMaxDims.back() = weightDims.back();

    return MemoryDescUtils::makeDummyShape(Shape(inMinDims, inMaxDims)).getStaticDims();
}

VectorDims acl_fc_executor::makeDummyOutputDims(const VectorDims& inShape,
                                                const VectorDims& wShape,
                                                const size_t out_rank) {
    size_t activationRank = inShape.size();
    size_t channelRank = wShape.size() - 1;
    // activation   weight    output_shape
    // NCHW         CoCHW     NCo
    // TNC          CoC       TNCo
    // NC           CoC       NCo
    VectorDims outputShape(out_rank, 1);
    // set Co
    outputShape.back() = wShape[0];
    // set batch dims
    size_t batchRank = activationRank - channelRank;
    size_t startIdx = out_rank - batchRank - 1;
    for (size_t i = 0; i < batchRank; i++) {
        outputShape[i + startIdx] = inShape[i];
    }

    return outputShape;
}

DnnlMemoryDescPtr acl_fc_executor::makeTransposedWeightDescriptor(const DnnlMemoryDescPtr& srcDesc,
                                                                  const DnnlMemoryDescPtr& dstDesc) {
    const auto& weiDesc = srcDesc->getDnnlDesc();
    dnnl::memory::dims wgtDims2D = reshapeDownToRank<2>(weiDesc.get_dims());
    const auto reorderedWeiDesc = dnnl::memory::desc{wgtDims2D, weiDesc.get_data_type(), dnnl::memory::format_tag::ba};
    const auto transposedWeiDesc = reorderedWeiDesc.reshape(dstDesc->getDnnlDesc().get_dims());

    return DnnlExtensionUtils::makeDescriptor(transposedWeiDesc);
}

std::optional<MemoryPtr> acl_fc_executor::convertWeightPrecision(const MemoryPtr& input,
                                                                 const MemoryPtr& output,
                                                                 ov::element::Type weightPrecision) {
    MemoryArgs memoryArgs;
    memoryArgs[ARG_SRC] = input;
    memoryArgs[ARG_DST] = output;

    auto aclWeightsConverter = std::make_shared<acl_fc_executor::ACLWeightsConverter>();
    if (aclWeightsConverter->update(memoryArgs)) {
        aclWeightsConverter->execute(memoryArgs);
        return std::optional<MemoryPtr>(memoryArgs.at(ARG_DST));
    }

    if (!node::Convert::isSupportedDesc(input->getDesc()) || !node::Convert::isSupportedDesc(output->getDesc())) {
        return {};
    }

    auto data = static_cast<const uint8_t*>(input->getData());
    std::vector<uint8_t> tmpBuff;
    tmpBuff.resize(output->getSize());
    cpu_convert(data,
                tmpBuff.data(),
                DnnlExtensionUtils::DataTypeToElementType(input->getDataType()),
                weightPrecision,
                input->getSize() / input->getDesc().getPrecision().size());

    return std::optional<MemoryPtr>(std::make_shared<Memory>(output->getPrimitive().get_engine(),
                                                             output->getDesc().cloneWithNewPrecision(weightPrecision),
                                                             tmpBuff.data()));
}

std::optional<MemoryPtr> acl_fc_executor::reorderDataFallback(const MemoryPtr& input,
                                                              const MemoryPtr& output,
                                                              const ExecutorContext::CPtr& context) {
    if (output->getDataType() == input->getDataType()) {
        return {};
    }
    const auto inPrc = DnnlExtensionUtils::DataTypeToElementType(input->getDataType());
    auto convertedDstMemoryDesc = output->getDesc().cloneWithNewPrecision(inPrc);
    dnnl::reorder reorderWithoutConvert =
        getReorderPrim(context->getRuntimeCache(),
                       output->getPrimitive().get_engine(),
                       input->getPrimitive().get_desc(),
                       MemoryDescUtils::convertToDnnlMemoryDesc(convertedDstMemoryDesc)->getDnnlDesc());

    if (reorderWithoutConvert &&
        parse_impl_name(reorderWithoutConvert.get_primitive_desc()->impl()->name()) != ref_any) {
        auto convertOutputOpt = convertWeightPrecision(input, output, inPrc);
        if (!convertOutputOpt) {
            return {};
        }
        auto convertOutput = *convertOutputOpt;

        if (reorderWithoutConvert) {
            dnnl::stream loc_stream(output->getPrimitive().get_engine(), dnnl::stream::flags::in_order);
            reorderWithoutConvert.execute(
                loc_stream,
                {{DNNL_ARG_FROM, convertOutput->getPrimitive()}, {DNNL_ARG_TO, output->getPrimitive()}});
            return std::optional<MemoryPtr>(output);
        }
    }
    return {};
}

MemoryPtr acl_fc_executor::reorderData(const DnnlMemoryDescPtr& srcWeightDesc,
                                       const DnnlMemoryDescPtr& dstWeightDesc,
                                       const MemoryCPtr& weightsMem,
                                       const ExecutorContext::CPtr& context) {
    MemoryPtr input = std::make_shared<Memory>(context->getEngine(), srcWeightDesc, weightsMem->getData());
    MemoryPtr output = std::make_shared<Memory>(context->getEngine(), dstWeightDesc);
    if (!input->getDesc().isDefined() || !output->getDesc().isDefined()) {
        OPENVINO_THROW("Can't reorder data with dynamic shapes");
    }

    if (input->getShape().hasZeroDims() || output->getShape().hasZeroDims()) {
        return output;
    }

    if (input->getDesc().isCompatible(output->getDesc())) {
        auto srcPtr = static_cast<uint8_t*>(input->getData());
        auto dstPtr = static_cast<uint8_t*>(output->getData());
        auto copySize = output->getSize();
        cpu_memcpy(dstPtr, srcPtr, copySize);
        return output;
    }

    // try directly reorder
    auto engine = output->getPrimitive().get_engine();
    dnnl::reorder directReorder = getReorderPrim(context->getRuntimeCache(),
                                                 engine,
                                                 input->getPrimitive().get_desc(),
                                                 output->getPrimitive().get_desc());

    if (!directReorder || parse_impl_name(directReorder.get_primitive_desc()->impl()->name()) == ref_any) {
        // try precision conversion then do the reorder
        auto fallbackOutput = reorderDataFallback(input, output, context);
        if (fallbackOutput) {
            return *fallbackOutput;
        }
    }
    // if precision conversion does not work then do direct reference reorder
    if (directReorder) {
        dnnl::stream loc_stream(engine, dnnl::stream::flags::in_order);
        directReorder.execute(loc_stream,
                              {{DNNL_ARG_FROM, input->getPrimitive()}, {DNNL_ARG_TO, output->getPrimitive()}});
    } else {
        OPENVINO_THROW("Could not make onednn reorder.");
    }
    return output;
}

MemoryPtr acl_fc_executor::reorderWeights(const MemoryArgs& memory,
                                          const ExecutorContext::CPtr context,
                                          ACLFCAttrs& aclfcAttrs,
                                          DnnlMemoryDescPtr dnnlSrcDesc,
                                          DnnlMemoryDescPtr dnnlDstDesc) {
    auto create = [&]() {
        MemoryPtr weightsMemory = memory.at(ARG_WEI);
        if (aclfcAttrs.isWeightsRepacked || aclfcAttrs.isConvertedWeights) {
            weightsMemory = reorderData(dnnlSrcDesc, dnnlDstDesc, memory.at(ARG_WEI), context);
            DEBUG_LOG("ACLFullyConnectedExecutor: cache miss, perform packing");
        }
        return weightsMemory;
    };

    auto weightCache = context->getWeightsCache();
    if (weightCache != nullptr) {
        const auto& wgtDims = memory.at(ARG_WEI)->getStaticDims();
        const auto N = wgtDims[0];
        const auto K = wgtDims[1];
        std::string format = "fc_acl_" + std::to_string(N) + "_" + std::to_string(K);
        const std::string string_hash = format + "_" + std::to_string(memory.at(ARG_WEI)->getSize()) + "_" +
                                        std::to_string(reinterpret_cast<uint64_t>(memory.at(ARG_WEI)->getData()));
        DEBUG_LOG("ACLFullyConnectedExecutor: findOrCreate, string_hash: ", string_hash);
        return *weightCache->findOrCreate(string_hash, create);
    }

    DEBUG_LOG("ACLFullyConnectedExecutor: Weights cache is not available");
    return create();
}

MemoryPtr acl_fc_executor::prepareWeightMemory(const MemoryArgs& memory,
                                               const ExecutorContext::CPtr& context,
                                               const FCAttrs& attrs,
                                               ACLFCAttrs& aclfcAttrs,
                                               const PostOps& postOps,
                                               arm_compute::WeightFormat& expectedWeightFormat,
                                               arm_compute::TensorInfo& weiTensorInfo) {
    MemoryArgs memoryArgs;
    memoryArgs[ARG_BIAS] = memory.at(ARG_BIAS);
    memoryArgs[ARG_WEI] = memory.at(ARG_WEI);

    auto originalWeightsDesc = memory.at(ARG_WEI)->getDescPtr();
    // normalize weights to 2D
    const auto& wgtDims = originalWeightsDesc->getShape().getStaticDims();
    const VectorDims wgtDims2D = reshapeDownToRank<2>(wgtDims);
    originalWeightsDesc = std::make_shared<CpuBlockedMemoryDesc>(originalWeightsDesc->getPrecision(), Shape{wgtDims2D});
    auto dnnlSrcDesc = MemoryDescUtils::convertToDnnlMemoryDesc(originalWeightsDesc);
    auto dstDesc = originalWeightsDesc->cloneWithNewPrecision(aclfcAttrs.inputPrecision);
    auto dnnlDstDesc = MemoryDescUtils::convertToDnnlMemoryDesc(dstDesc);

    if (memory.at(ARG_SRC_0)->getShape().isDynamic()) {
        const auto& inShape = memory.at(ARG_SRC_0)->getShape();
        const auto& wShape = originalWeightsDesc->getShape();
        const auto& inDymmyDims = makeDummyInputDims(inShape, wShape);
        const auto& outDymmyDims =
            makeDummyOutputDims(inDymmyDims, wShape.getStaticDims(), memory.at(ARG_DST)->getShape().getRank());
        memoryArgs[ARG_SRC_0] =
            std::make_shared<Memory>(context->getEngine(),
                                     memory.at(ARG_SRC_0)->getDescPtr()->cloneWithNewDims(inDymmyDims));
        memoryArgs[ARG_DST] =
            std::make_shared<Memory>(context->getEngine(),
                                     memory.at(ARG_DST)->getDescPtr()->cloneWithNewDims(outDymmyDims));
    } else {
        memoryArgs[ARG_SRC_0] = memory.at(ARG_SRC_0);
        memoryArgs[ARG_DST] = memory.at(ARG_DST);
    }
    // TODO: ACLWeightFormatGenerator should be replaced with Reorder executor
    // that calls ACL NEReorder + NETranspose or dnnl::reorder depending on backend availability
    auto aclWeightsRepack = std::make_shared<acl_fc_executor::ACLWeightFormatGenerator>(attrs, postOps, memoryArgs);
    bool isNeededReorder = aclWeightsRepack->update(memoryArgs);
    expectedWeightFormat =
        isNeededReorder ? aclWeightsRepack->getOptImplWeightFormat() : arm_compute::WeightFormat::UNSPECIFIED;
    weiTensorInfo = aclWeightsRepack->getTensorInfo(ACLArgs::ACL_WEI);

    if (isNeededReorder) {
        dnnl::impl::dim_t o_dim = 0;
        dnnl::impl::dim_t inner_dim = 1;
        std::vector<dnnl::impl::dim_t> remaining_dims = {};
        auto weights_md_ = dnnlDstDesc->getDnnlDesc().get();
        dnnl::impl::cpu::acl::acl_utils::reorder_to_weight_format(weiTensorInfo,
                                                                  *weights_md_,
                                                                  expectedWeightFormat,
                                                                  inner_dim,
                                                                  o_dim,
                                                                  remaining_dims,
                                                                  {});
        if (aclfcAttrs.weightsNonTransposed) {
            dnnlSrcDesc = makeTransposedWeightDescriptor(dnnlSrcDesc, dnnlDstDesc);
        }
        aclfcAttrs.isWeightsRepacked = true;
        return reorderWeights(memory, context, aclfcAttrs, dnnlSrcDesc, dnnlDstDesc);
    }
    if (!aclfcAttrs.weightsNonTransposed) {
        dnnlDstDesc = makeTransposedWeightDescriptor(dnnlDstDesc, dnnlSrcDesc);
        aclfcAttrs.isWeightsRepacked = true;
    }
    return reorderWeights(memory, context, aclfcAttrs, dnnlSrcDesc, dnnlDstDesc);
}

static bool checkPostOps(const PostOps& postOps) {
    // Add postops
    if (!postOps.empty() && postOps.size() == 1) {
        if (const auto activation = std::dynamic_pointer_cast<ActivationPostOp>(postOps[0])) {
            if (checkActivationLayerInfo(convertToEltwiseAlgorithm(activation->type()))) {
                return true;
            }
        }
    }
    return false;
}

static void initFCAttrs(const FCAttrs& attrs,
                        ACLTensorAttrs& aclTensorAttrs,
                        ACLFCAttrs& aclfcAttrs,
                        const MemoryArgs& memory,
                        arm_compute::FullyConnectedLayerInfo& fullyConnectedLayerInfo,
                        const PostOps& postOps) {
    aclTensorAttrs.hasLayoutTypeNHWC = memory.at(ARG_SRC)->getDescPtr()->hasLayoutType(LayoutType::nspc);
    fullyConnectedLayerInfo.weights_trained_layout = getAclDataLayoutByMemoryDesc(memory.at(ARG_WEI)->getDescPtr());
    aclfcAttrs.inputPrecision = memory.at(ARG_SRC)->getDescPtr()->getPrecision();
    fullyConnectedLayerInfo.transpose_weights = false;
    aclfcAttrs.weightsNonTransposed = attrs.weightsNonTransposed;

    if (checkPostOps(postOps)) {
        auto activation = std::dynamic_pointer_cast<ActivationPostOp>(postOps[0]);
        fullyConnectedLayerInfo.activation_info = getActivationLayerInfo(convertToEltwiseAlgorithm(activation->type()),
                                                                         activation->alpha(),
                                                                         activation->beta(),
                                                                         activation->gamma());
    }

    if (memory.at(ARG_SRC)->getPrecision() != memory.at(ARG_WEI)->getPrecision()) {
        aclfcAttrs.isConvertedWeights = true;
    }
}

arm_compute::TensorShape acl_fc_executor::normalizeDimsTo2D(const arm_compute::TensorShape shape) {
    size_t norm_dim = std::accumulate(shape.begin() + 1, shape.end(), 1, std::multiplies<>());
    return arm_compute::TensorShape(shape[0], norm_dim);
}

void acl_fc_executor::updateFCTensorsShapes(ACLShapes& aclMemoryShapes) {
    aclMemoryShapes[ACLArgs::ACL_WEI] = normalizeDimsTo2D(aclMemoryShapes[ACLArgs::ACL_WEI]);
    aclMemoryShapes[ACLArgs::ACL_SRC_0] = normalizeDimsTo2D(aclMemoryShapes[ACLArgs::ACL_SRC_0]);
    aclMemoryShapes[ACLArgs::ACL_DST] = normalizeDimsTo2D(aclMemoryShapes[ACLArgs::ACL_DST]);
    std::swap(aclMemoryShapes[ACLArgs::ACL_WEI][0], aclMemoryShapes[ACLArgs::ACL_WEI][1]);
}

arm_compute::Status acl_fc_executor::ACLWeightsConverter::validateTensorsInfo(const ACLInfos& aclMemoryInfos) {
    return arm_compute::NECast::validate(aclMemoryInfos[ACLArgs::ACL_SRC_0].get(),
                                         aclMemoryInfos[ACLArgs::ACL_DST].get(),
                                         arm_compute::ConvertPolicy::SATURATE);
}

ACLFunction acl_fc_executor::ACLWeightsConverter::configureFunction(const ACLTensors& aclMemoryTensors) {
    auto neCast = std::make_unique<arm_compute::NECast>();
    neCast->configure(aclMemoryTensors[ACLArgs::ACL_SRC_0].get(),
                      aclMemoryTensors[ACLArgs::ACL_DST].get(),
                      arm_compute::ConvertPolicy::SATURATE);
    return neCast;
}

acl_fc_executor::ACLWeightFormatGenerator::ACLWeightFormatGenerator(const FCAttrs& attrs,
                                                                    const PostOps& postOps,
                                                                    const MemoryArgs& memory) {
    initFCAttrs(attrs, aclTensorAttrs, aclfcAttrs, memory, fullyConnectedLayerInfo, postOps);
}

void acl_fc_executor::ACLWeightFormatGenerator::updateTensorsShapes(ACLShapes& aclMemoryShapes) {
    updateFCTensorsShapes(aclMemoryShapes);
}

arm_compute::Status acl_fc_executor::ACLWeightFormatGenerator::validateTensorsInfo(const ACLInfos& aclMemoryInfos) {
    if (aclfcAttrs.isConvertedWeights) {
        aclMemoryInfos[ACLArgs::ACL_WEI]->set_data_type(aclMemoryInfos[ACLArgs::ACL_SRC_0]->data_type());
    }
    int icTotal = aclMemoryInfos[ACLArgs::ACL_SRC_0]->dimension(0);
    return arm_compute::NEFullyConnectedLayer::has_opt_impl(
        expectedWeightFormat,
        aclMemoryInfos[ACLArgs::ACL_SRC_0].get(),
        aclMemoryInfos[ACLArgs::ACL_WEI].get(),
        aclMemoryInfos[ACLArgs::ACL_BIAS].get(),
        aclMemoryInfos[ACLArgs::ACL_DST].get(),
        fullyConnectedLayerInfo,
        arm_compute::WeightsInfo(false, 1, 1, icTotal, false, arm_compute::WeightFormat::ANY));
}

ACLFunction acl_fc_executor::ACLWeightFormatGenerator::configureFunction(const ACLTensors& aclMemoryTensors) {
    return std::make_unique<arm_compute::NEFullyConnectedLayer>();
}

}  // namespace ov::intel_cpu
