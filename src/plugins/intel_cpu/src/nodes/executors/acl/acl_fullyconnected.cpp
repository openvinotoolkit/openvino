// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <oneapi/dnnl/dnnl_types.h>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <common/primitive_desc_iface.hpp>
#include "dnnl_postops_composer.h"

#include "acl_fullyconnected.hpp"
#include "acl_utils.hpp"
#include "nodes/executors/dnnl/dnnl_post_op_data.hpp"
#include "nodes/convert.h"
#include "nodes/reorder.h"
#include "nodes/executors/dnnl/dnnl_utils.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "utils/debug_capabilities.h"
#include "nodes/executors/debug_messages.hpp"
#include "nodes/executors/implementation_utils.hpp"
#include "nodes/common/cpu_convert.h"
#include "nodes/common/cpu_memcpy.h"
#include "nodes/common/reorder_prim.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_memory_desc.h"

namespace ov {
namespace intel_cpu {

using namespace dnnl;
using namespace ov::element;

static VectorDims makeDummyInputDims(const Shape& inShape, const Shape& wShape) {
    const auto& weightDims = wShape.getStaticDims();

    auto inMinDims = inShape.getMinDims();
    auto inMaxDims = inShape.getMaxDims();
    inMinDims.back() = weightDims.back();
    inMaxDims.back() = weightDims.back();

    return MemoryDescUtils::makeDummyShape(Shape(inMinDims, inMaxDims)).getStaticDims();
}

static VectorDims makeDummyOutputDims(const VectorDims& inShape, const VectorDims& wShape, const size_t out_rank) {
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

DnnlMemoryDescPtr makeTransposedWeightDescriptor(const DnnlMemoryDescPtr srcDesc,
                                                                  const DnnlMemoryDescPtr dstDesc,
                                                                  bool weightsNonTransposed) {
    if (!weightsNonTransposed)
        return srcDesc;

    const auto& weiDesc = srcDesc->getDnnlDesc();
    const auto reorderedWeiDesc =
        dnnl::memory::desc{weiDesc.get_dims(), weiDesc.get_data_type(), dnnl::memory::format_tag::ba};
    const auto transposedWeiDesc = reorderedWeiDesc.reshape(dstDesc->getDnnlDesc().get_dims());

    return DnnlExtensionUtils::makeDescriptor(transposedWeiDesc);
}

void reorderData(const IMemory &input, const IMemory &output, MultiCachePtr cache) {
    if (!input.getDesc().isDefined() || !output.getDesc().isDefined())
        OPENVINO_THROW("Can't reorder data with dynamic shapes");

    if (input.getShape().hasZeroDims() || output.getShape().hasZeroDims()) {
        return;
    }

    if (input.getDesc().isCompatible(output.getDesc())) {
        if (input.getDesc().getPrecision() == element::string) {
            auto srcPtr = input.getDataAs<StringMemory::OvString>();
            auto dstPtr = output.getDataAs<StringMemory::OvString>();
            std::copy(srcPtr, srcPtr + output.getShape().getElementsCount(), dstPtr);
        } else {
            auto srcPtr = static_cast<uint8_t*>(input.getData());
            auto dstPtr = static_cast<uint8_t*>(output.getData());

            auto copySize = output.getSize();
            cpu_memcpy(dstPtr, srcPtr, copySize);
        }
    } else {
        dnnl::reorder reorder;
        std::vector<uint8_t> tmpBuff;

        auto srcMemory = input.getPrimitive();
        auto dstMemory = output.getPrimitive();

        auto srcMemoryDesc = srcMemory.get_desc();
        auto dstMemoryDesc = dstMemory.get_desc();

        auto engine = dstMemory.get_engine();

        if (srcMemoryDesc.get_ndims() != dstMemoryDesc.get_ndims()) {
            //rank mismatch, try to reshape source mem descriptor
            constexpr bool allowEmpty = true;
            auto reshapedSrcMemDesc = srcMemoryDesc.reshape(dstMemoryDesc.get_dims(), allowEmpty);
            if (reshapedSrcMemDesc) {
                srcMemoryDesc = reshapedSrcMemDesc;
                srcMemory = dnnl::memory(srcMemoryDesc, engine, srcMemory.get_data_handle());
            }
        }

        // try directly reorder
        reorder = getReorderPrim(cache, engine, srcMemoryDesc, dstMemoryDesc);
        if (!reorder) {
            // try precision conversion then do the reorder
            if (output.getDataType() != input.getDataType() && node::Convert::isSupportedDesc(input.getDesc()) &&
                node::Convert::isSupportedDesc(output.getDesc())) {
                //we probably could not make the reorder because there is no one supporting this precision conversion
                //lets try to convert data first using cpu_convert
                auto data = static_cast<const uint8_t *>(input.getData());
                tmpBuff.resize(output.getSize());

                const auto outPrc = DnnlExtensionUtils::DataTypeToElementType(output.getDataType());
                cpu_convert(data, tmpBuff.data(), DnnlExtensionUtils::DataTypeToElementType(input.getDataType()),
                            outPrc, input.getSize() / input.getDesc().getPrecision().size());

                auto tmpDesc = input.getDesc().cloneWithNewPrecision(outPrc);
                Memory tmpMem(engine, std::move(tmpDesc), tmpBuff.data());

                srcMemory = tmpMem.getPrimitive();
                reorder = getReorderPrim(cache, dstMemory.get_engine(), srcMemory.get_desc(), dstMemory.get_desc());
            }
            if (!reorder) {
                OPENVINO_THROW("No reorder available for the following tensor descriptors: ",
                               input.getDesc().serializeFormat(),
                               " and ",
                               output.getDesc().serializeFormat());
            }
        }
        if (reorder) {
            dnnl::stream loc_stream(engine, dnnl::stream::flags::in_order);
            reorder.execute(loc_stream, {{DNNL_ARG_FROM, srcMemory}, {DNNL_ARG_TO, dstMemory}});
        } else {
            OPENVINO_THROW("Could not make onednn reorder.");
        }
    }
}

static MemoryPtr prepareWeightMemory(const MemoryArgs &memory,
                                     const ExecutorContext::CPtr context,
                                     const FCAttrs &attrs,
                                     const ACLFCAttrs& aclfcAttrs,
                                     const PostOps &postOps) {
    DEBUG_LOG("ACLFullyConnectedExecutor: prepack weights");
    const auto& wgtDims = memory.at(ARG_WEI)->getStaticDims();
    const auto N = wgtDims[0];
    const auto K = wgtDims[1];

    auto create = [&]() {
        MemoryPtr final_ptr = memory.at(ARG_WEI);
        // Convert weights precision
        if (aclfcAttrs.isConvertedWeights) {
            std::chrono::high_resolution_clock::time_point __start = std::chrono::high_resolution_clock::now();
            MemoryArgs memoryArgs;
            memoryArgs[ARG_SRC_0] = memory.at(ARG_WEI);
            memoryArgs[ARG_DST] = std::make_shared<Memory>(context->getEngine(),
                                                           memoryArgs[ARG_SRC_0]->getDescPtr()->cloneWithNewPrecision(
                                                                   aclfcAttrs.inputPrecision));
            auto aclWeightsConverter = std::make_shared<acl_fc_executor::ACLWeightsConverter>();
            if (aclWeightsConverter->update(memoryArgs)) {
                aclWeightsConverter->execute(memoryArgs);
            } else {
                auto count_wei_elem = std::accumulate(memoryArgs[ARG_SRC_0]->getStaticDims().begin(),
                                                      memoryArgs[ARG_SRC_0]->getStaticDims().end(),
                                                      1,
                                                      std::multiplies<>());
                cpu_convert(memoryArgs[ARG_SRC_0]->getData(),
                            memoryArgs[ARG_DST]->getData(),
                            memoryArgs[ARG_SRC_0]->getPrecision(),
                            memoryArgs[ARG_DST]->getPrecision(),
                            count_wei_elem);
            }
            final_ptr = memoryArgs[ARG_DST];
            std::chrono::high_resolution_clock::time_point __finish = std::chrono::high_resolution_clock::now();
            //std::stringstream msg;
            //msg << "convert;" << std::chrono::duration_cast<std::chrono::microseconds>(__finish - __start).count() << std::endl;
            //std::cout << msg.str();
        }
        // Packed weights
        {
            std::chrono::high_resolution_clock::time_point __start = std::chrono::high_resolution_clock::now();
            arm_compute::WeightFormat expectedWeightFormat;
            /*bool isNeededReorder;
            {
                MemoryArgs memoryArgs;
                memoryArgs[ARG_BIAS]  = memory.at(ARG_BIAS);
                memoryArgs[ARG_WEI]   = final_ptr;
                if (memory.at(ARG_SRC_0)->getShape().isDynamic()) {
                    const auto& inShape = memory.at(ARG_SRC_0)->getShape();
                    const auto& wShape = final_ptr->getShape();
                    const auto& inDymmyDims = makeDummyInputDims(inShape, wShape);
                    const auto& outDymmyDims = makeDummyOutputDims(inDymmyDims, wShape.getStaticDims(), memory.at(ARG_DST)->getShape().getRank());
                    memoryArgs[ARG_SRC_0] = std::make_shared<Memory>(context->getEngine(),
                                                                     memory.at(ARG_SRC_0)->getDescPtr()->cloneWithNewDims(inDymmyDims));
                    memoryArgs[ARG_DST] = std::make_shared<Memory>(context->getEngine(),
                                                                   memory.at(ARG_DST)->getDescPtr()->cloneWithNewDims(outDymmyDims));
                } else {
                    memoryArgs[ARG_SRC_0] = memory.at(ARG_SRC_0);
                    memoryArgs[ARG_DST]   = memory.at(ARG_DST);
                }
                auto aclWeightsRepack = std::make_shared<acl_fc_executor::ACLWeightFormatGenerator>(attrs, postOps, memoryArgs);
                isNeededReorder = aclWeightsRepack->update(memoryArgs);
                expectedWeightFormat = aclWeightsRepack->getOptImplWeightFormat();
            }
            if (isNeededReorder && expectedWeightFormat == arm_compute::WeightFormat::OHWI) {
                MemoryArgs memoryArgs;
                memoryArgs[ARG_SRC_0] = final_ptr;
                memoryArgs[ARG_DST] = std::make_shared<Memory>(context->getEngine(),
                                                               memoryArgs[ARG_SRC_0]->getDescPtr()->clone());
                auto aclWeightsReorder = std::make_shared<acl_fc_executor::ACLWeightsReorder>(
                        arm_compute::WeightFormat::OHWI, expectedWeightFormat);
                if (aclWeightsReorder->update(memoryArgs)) {
                    aclWeightsReorder->execute(memoryArgs);
                    final_ptr = memoryArgs[ARG_DST];
                }
            }*/

            //std::chrono::high_resolution_clock::time_point __finish = std::chrono::high_resolution_clock::now();
            //std::stringstream msg;
            //msg << "reorder" << isNeededReorder  << "/" << static_cast<int>(expectedWeightFormat) << ";" <<
            //std::chrono::duration_cast<std::chrono::microseconds>(__finish - __start).count() << std::endl;
            //std::cout << msg.str();
        }
        // Transpose weights
        if (!aclfcAttrs.weightsNonTransposed) {
            //std::chrono::high_resolution_clock::time_point __start = std::chrono::high_resolution_clock::now();

            auto reverse_weights_dims = memory.at(ARG_WEI)->getStaticDims();
            if (reverse_weights_dims.size() == 3) {
                reverse_weights_dims = VectorDims(
                        {reverse_weights_dims[0] * reverse_weights_dims[1], reverse_weights_dims[2]});
            }
            std::reverse(reverse_weights_dims.begin(), reverse_weights_dims.end());
            MemoryArgs memoryArgs;
            memoryArgs[ARG_SRC_0] = final_ptr;
            memoryArgs[ARG_DST] = std::make_shared<Memory>(context->getEngine(),
                                                           CpuBlockedMemoryDesc(final_ptr->getPrecision(),
                                                                                intel_cpu::Shape(reverse_weights_dims)));

            /*auto aclWeightsTranspose = std::make_shared<acl_fc_executor::ACLWeightsTranspose>();
            if (aclWeightsTranspose->update(memoryArgs)) {
                aclWeightsTranspose->execute(memoryArgs);
                final_ptr = memoryArgs[ARG_DST];
            }*/
//ONEDNN SECTION START
            //const auto& eng = context->getEngine();
            //auto srcDesc = memory.at(ARG_SRC)->getDescPtr();
            const auto& weiDesc = memoryArgs[ARG_DST]->getDescPtr();//memory.at(ARG_WEI)->getDescPtr();
            //const auto& biasDesc = memory.at(ARG_BIAS)->getDescPtr();
            auto dstDesc = memoryArgs[ARG_SRC_0]->getDescPtr();//memory.at(ARG_DST)->getDescPtr();

            /*if (srcDesc->getShape().isDynamic()) {
                const auto& inShape = srcDesc->getShape();
                const auto& wShape = weiDesc->getShape();
                const auto& inDymmyDims = makeDummyInputDims(inShape, wShape);
                srcDesc = srcDesc->cloneWithNewDims(inDymmyDims);
                const auto& outDymmyDims =
                    makeDummyOutputDims(inDymmyDims, wShape.getStaticDims(), dstDesc->getShape().getRank());
                dstDesc = dstDesc->cloneWithNewDims(outDymmyDims);
            }*/

            //const dnnl::memory::desc srcDnnlDesc = MemoryDescUtils::convertToDnnlMemoryDesc(srcDesc)->getDnnlDesc();
            //const dnnl::memory::desc weiDnnlDesc = MemoryDescUtils::convertToDnnlMemoryDesc(weiDesc)->getDnnlDesc();
            //const dnnl::memory::desc dstDnnlDesc = MemoryDescUtils::convertToDnnlMemoryDesc(dstDesc)->getDnnlDesc();
            //const dnnl::memory::desc biaDnnlDesc = MemoryDescUtils::convertToDnnlMemoryDesc(biasDesc)->getDnnlDesc();
            //const auto postOpData = createPrimitiveAttrs(attrs, postOps, memory, context, false);
            //const auto useWeightsDecompression = false;

            /*const auto useSparseWeights = attrs.sparseWeights;
            const auto primDesc = createPrimitiveDesc(srcDnnlDesc,
                                                    weiDnnlDesc,
                                                    biaDnnlDesc,
                                                    dstDnnlDesc,
                                                    postOpData.attr,
                                                    context->getEngine(),
                                                    context->getImplPriorities(),
                                                    useSparseWeights,
                                                    useWeightsDecompression);*/

            //const auto weightsDesc = MemoryDescUtils::convertToDnnlMemoryDesc(dstDesc);
            //DnnlExtensionUtils::makeDescriptor(/*primDesc.weights_desc()*/dstDnnlDesc);
            auto originalWeightsDesc = MemoryDescUtils::convertToDnnlMemoryDesc(weiDesc);//MemoryDescUtils::convertToDnnlMemoryDesc(weiDesc);

            /*auto weiDescDims = weiDesc->getShape().getDims();
            std::swap(weiDescDims[0], weiDescDims[1]);
            auto weiDescRevertedDims = weiDesc->cloneWithNewDims(weiDescDims);
            auto originalWeightsDesc = MemoryDescUtils::convertToDnnlMemoryDesc(weiDescRevertedDims);*/

            auto dstDescDims = dstDesc->getShape().getDims();
            std::swap(dstDescDims[0], dstDescDims[1]);
            auto dstDescRevertedDims = weiDesc->cloneWithNewDims(dstDescDims);
            const auto weightsDesc = MemoryDescUtils::convertToDnnlMemoryDesc(dstDescRevertedDims);

            originalWeightsDesc = makeTransposedWeightDescriptor(originalWeightsDesc, weightsDesc, true/*aclfcAttrs.weightsNonTransposed*/);

            /*final_ptr = utils::prepareWeightsMemory(originalWeightsDesc,
                                      weightsDesc,
                                      memoryArgs[ARG_SRC_0],//memory.at(ARG_WEI),
                                      context,
                                      false);*/

            DnnlMemoryDescPtr srcWeightDesc = originalWeightsDesc;
            DnnlMemoryDescPtr dstWeightDesc = weightsDesc;
            MemoryCPtr weightsMem = memoryArgs[ARG_SRC_0];//final_ptr;//memory.at(ARG_WEI);

            Memory srcMemory{context->getEngine(), srcWeightDesc, weightsMem->getData()};
            MemoryPtr _ptr = std::make_shared<Memory>(context->getEngine(), dstWeightDesc);
            auto rtCache = context->getRuntimeCache();
            /*node::Reorder::*/reorderData(srcMemory, *_ptr, rtCache);
            final_ptr = _ptr;
            //ONEDNN SECTION END
            //std::chrono::high_resolution_clock::time_point __finish = std::chrono::high_resolution_clock::now();
            //std::stringstream msg;
            //msg << "transpose;" << std::chrono::duration_cast<std::chrono::microseconds>(__finish - __start).count() << std::endl;
            //std::cout << msg.str();
        }
        DEBUG_LOG("ACLFullyConnectedExecutor: cache miss, perform packing");
        return final_ptr;
    };

    auto weightCache = context->getWeightsCache();
    if (weightCache != nullptr) {
        std::string format = "fc_acl_" + std::to_string(N) + "_" + std::to_string(K);
        const std::string string_hash = format + "_" + std::to_string(memory.at(ARG_WEI)->getSize()) + "_" +
                                        std::to_string(reinterpret_cast<uint64_t>(memory.at(ARG_WEI)->getData()));
        DEBUG_LOG("ACLFullyConnectedExecutor: findOrCreate, string_hash: ", string_hash);
        return *weightCache->findOrCreate(string_hash, create);
    }

    DEBUG_LOG("ACLFullyConnectedExecutor: Weights cache is not available");
    return create();
}

static bool checkPostOps(const PostOps &postOps) {
    if (postOps.empty()) {
        return true;
    }
    if (postOps.size() > 1) {
        return false;
    }
    if (const auto activation = std::dynamic_pointer_cast<ActivationPostOp>(postOps[0])) {
        if (checkActivationLayerInfo(convertToEltwiseAlgorithm(activation->type()))) {
            return true;
        }
    }
    return false;
}

static void initFCAttrs(const FCAttrs &attrs,
                        ACLTensorAttrs& aclTensorAttrs,
                        ACLFCAttrs& aclfcAttrs,
                        const MemoryArgs &memory,
                        arm_compute::FullyConnectedLayerInfo& fullyConnectedLayerInfo,
                        const PostOps &postOps) {
    aclTensorAttrs.hasLayoutTypeNHWC = memory.at(ARG_SRC)->getDescPtr()->hasLayoutType(LayoutType::nspc);
    fullyConnectedLayerInfo.weights_trained_layout = getAclDataLayoutByMemoryDesc(memory.at(ARG_WEI)->getDescPtr());
    aclfcAttrs.inputPrecision = memory.at(ARG_SRC)->getDescPtr()->getPrecision();
    fullyConnectedLayerInfo.transpose_weights = false;
    aclfcAttrs.weightsNonTransposed = attrs.weightsNonTransposed;

    if (!postOps.empty() && checkPostOps(postOps)) {
        auto activation = std::dynamic_pointer_cast<ActivationPostOp>(postOps[0]);
        fullyConnectedLayerInfo.activation_info = getActivationLayerInfo(
                convertToEltwiseAlgorithm(activation->type()),
                activation->alpha(), activation->beta(), activation->gamma());
    }

    if (memory.at(ARG_SRC)->getPrecision() != memory.at(ARG_WEI)->getPrecision()) {
        aclfcAttrs.isConvertedWeights = true;
    }
}

ACLFullyConnectedExecutor::ACLFullyConnectedExecutor(const FCAttrs &attrs,
                                                     const PostOps &postOps,
                                                     const MemoryArgs &memory,
                                                     const ExecutorContext::CPtr context) {
    initFCAttrs(attrs, aclTensorAttrs, aclfcAttrs, memory, fullyConnectedLayerInfo, postOps);
    packedWeights = prepareWeightMemory(memory, context, attrs, aclfcAttrs, postOps);
}

bool ACLFullyConnectedExecutor::supports(const FCConfig &config) {
    VERIFY(one_of(srcType(config), ov::element::f16, ov::element::f32), UNSUPPORTED_SRC_PRECISIONS);
    VERIFY(one_of(weiType(config), ov::element::f16, ov::element::f32), UNSUPPORTED_WEI_PRECISIONS);
    VERIFY(postOpsNumbers(config) < 2,                                  UNSUPPORTED_NUMBER_OF_POSTOPS);
    VERIFY(checkPostOps(config.postOps),                                UNSUPPORTED_TYPE_OF_POSTOPS);
    VERIFY(one_of(srcRank(config), 2U, 3U, 4U),                         UNSUPPORTED_SRC_RANK);
    VERIFY(one_of(weiRank(config), 2U, 3U),                             UNSUPPORTED_WEI_RANK);
    return true;
}

static void updateFCTensorsShapes(ACLShapes& aclMemoryShapes) {
    if (aclMemoryShapes[ACLArgs::ACL_WEI].num_dimensions() == 3U) {
        aclMemoryShapes[ACLArgs::ACL_WEI] = arm_compute::TensorShape(
                {aclMemoryShapes[ACLArgs::ACL_WEI][0] * aclMemoryShapes[ACLArgs::ACL_WEI][1],
                 aclMemoryShapes[ACLArgs::ACL_WEI][2]});
    }

    if (one_of(aclMemoryShapes[ACLArgs::ACL_SRC_0].num_dimensions(), 3U, 4U)) {
        aclMemoryShapes[ACLArgs::ACL_SRC_0] = arm_compute::TensorShape({
            aclMemoryShapes[ACLArgs::ACL_WEI][0],
            aclMemoryShapes[ACLArgs::ACL_SRC_0].total_size() / aclMemoryShapes[ACLArgs::ACL_WEI][0]});
    }

    if (one_of(aclMemoryShapes[ACLArgs::ACL_DST].num_dimensions(), 3U, 4U)) {
        aclMemoryShapes[ACLArgs::ACL_DST] = arm_compute::TensorShape({
            aclMemoryShapes[ACLArgs::ACL_WEI][1],
            aclMemoryShapes[ACLArgs::ACL_SRC_0][1]});
    }

    std::swap(aclMemoryShapes[ACLArgs::ACL_WEI][0], aclMemoryShapes[ACLArgs::ACL_WEI][1]);
}

void ACLFullyConnectedExecutor::updateTensorsShapes(ACLShapes& aclMemoryShapes) {
    updateFCTensorsShapes(aclMemoryShapes);
}

arm_compute::Status ACLFullyConnectedExecutor::validateTensorsInfo(const ACLInfos & aclMemoryInfos) {
    if (aclfcAttrs.isConvertedWeights) {
        aclMemoryInfos[ACLArgs::ACL_WEI]->set_data_type(aclMemoryInfos[ACLArgs::ACL_SRC_0]->data_type());
    }
    return arm_compute::NEFullyConnectedLayer::validate(
            aclMemoryInfos[ACLArgs::ACL_SRC_0].get(),
            aclMemoryInfos[ACLArgs::ACL_WEI].get(),
            aclMemoryInfos[ACLArgs::ACL_BIAS].get(),
            aclMemoryInfos[ACLArgs::ACL_DST].get(),
            fullyConnectedLayerInfo,
            weightsInfo);
}

ACLFunction ACLFullyConnectedExecutor::configureFunction(const ACLTensors & aclMemoryTensors) {
    auto neFC = std::make_unique<arm_compute::NEFullyConnectedLayer>();
    neFC->configure(
            aclMemoryTensors[ACLArgs::ACL_SRC_0].get(),
            aclMemoryTensors[ACLArgs::ACL_WEI].get(),
            aclMemoryTensors[ACLArgs::ACL_BIAS].get(),
            aclMemoryTensors[ACLArgs::ACL_DST].get(),
            fullyConnectedLayerInfo,
            weightsInfo);

    if (aclfcAttrs.isConvertedWeights || !aclfcAttrs.weightsNonTransposed) {
        aclTensorAttrs.memoryUsageIndicator[ACLArgs::ACL_WEI] = false;
        aclMemoryTensors[ACLArgs::ACL_WEI]->allocator()->import_memory(packedWeights->getData());
    }
    return neFC;
}

arm_compute::Status acl_fc_executor::ACLWeightsConverter::validateTensorsInfo(const ACLInfos &aclMemoryInfos) {
    return arm_compute::NECast::validate(aclMemoryInfos[ACLArgs::ACL_SRC_0].get(),
                                         aclMemoryInfos[ACLArgs::ACL_DST].get(),
                                         arm_compute::ConvertPolicy::SATURATE);
}

ACLFunction acl_fc_executor::ACLWeightsConverter::configureFunction(const ACLTensors &aclMemoryTensors) {
    auto neCast = std::make_unique<arm_compute::NECast>();
    neCast->configure(aclMemoryTensors[ACLArgs::ACL_SRC_0].get(),
                      aclMemoryTensors[ACLArgs::ACL_DST].get(),
                      arm_compute::ConvertPolicy::SATURATE);
    return neCast;
}


arm_compute::Status acl_fc_executor::ACLWeightsTranspose::validateTensorsInfo(const ACLInfos &aclMemoryInfos) {
    return arm_compute::NETranspose::validate(aclMemoryInfos[ACLArgs::ACL_SRC_0].get(),
                                              aclMemoryInfos[ACLArgs::ACL_DST].get());
}

ACLFunction acl_fc_executor::ACLWeightsTranspose::configureFunction(const ACLTensors &aclMemoryTensors) {
    auto neTranspose = std::make_unique<arm_compute::NETranspose>();
    neTranspose->configure(aclMemoryTensors[ACLArgs::ACL_SRC_0].get(),
                           aclMemoryTensors[ACLArgs::ACL_DST].get());
    return neTranspose;
}

acl_fc_executor::ACLWeightFormatGenerator::ACLWeightFormatGenerator(const FCAttrs &attrs,
                                                                    const PostOps &postOps,
                                                                    const MemoryArgs &memory) {
    initFCAttrs(attrs, aclTensorAttrs, aclfcAttrs, memory, fullyConnectedLayerInfo, postOps);
}

void acl_fc_executor::ACLWeightFormatGenerator::updateTensorsShapes(ACLShapes &aclMemoryShapes) {
    updateFCTensorsShapes(aclMemoryShapes);
}

arm_compute::Status acl_fc_executor::ACLWeightFormatGenerator::validateTensorsInfo(const ACLInfos &aclMemoryInfos) {
    if (aclfcAttrs.isConvertedWeights) {
        aclMemoryInfos[ACLArgs::ACL_WEI]->set_data_type(aclMemoryInfos[ACLArgs::ACL_SRC_0]->data_type());
    }
    return arm_compute::NEFullyConnectedLayer::has_opt_impl(
            expectedWeightFormat,
            aclMemoryInfos[ACLArgs::ACL_SRC_0].get(),
            aclMemoryInfos[ACLArgs::ACL_WEI].get(),
            aclMemoryInfos[ACLArgs::ACL_BIAS].get(),
            aclMemoryInfos[ACLArgs::ACL_DST].get(),
            fullyConnectedLayerInfo,
            weightsInfo);
}

ACLFunction acl_fc_executor::ACLWeightFormatGenerator::configureFunction(const ACLTensors &aclMemoryTensors) {
    return std::make_unique<arm_compute::NEFullyConnectedLayer>();
}

arm_compute::Status acl_fc_executor::ACLWeightsReorder::validateTensorsInfo(const ACLInfos &aclMemoryInfos) {
#if defined(OPENVINO_ARCH_ARM64)
    return arm_compute::NEReorderLayer::validate(aclMemoryInfos[ACLArgs::ACL_SRC_0].get(),
                                                 aclMemoryInfos[ACLArgs::ACL_DST].get(),
                                                 inWeightFormat,
                                                 outWeightFormat);
#else
    return arm_compute::NECopy::validate(aclMemoryInfos[ACLArgs::ACL_SRC_0].get(),
                                         aclMemoryInfos[ACLArgs::ACL_DST].get());
#endif
}

ACLFunction acl_fc_executor::ACLWeightsReorder::configureFunction(const ACLTensors &aclMemoryTensors) {
#if defined(OPENVINO_ARCH_ARM64)
    auto neReorderLayer = std::make_unique<arm_compute::NEReorderLayer>();
    neReorderLayer->configure(aclMemoryTensors[ACLArgs::ACL_SRC_0].get(),
                              aclMemoryTensors[ACLArgs::ACL_DST].get(),
                              inWeightFormat,
                              outWeightFormat);
    return neReorderLayer;
#else
    auto neCopy = std::make_unique<arm_compute::NECopy>();
    neCopy->configure(aclMemoryTensors[ACLArgs::ACL_SRC_0].get(),
                              aclMemoryTensors[ACLArgs::ACL_DST].get());
    return neCopy;
#endif
}

}   // namespace intel_cpu
}   // namespace ov
