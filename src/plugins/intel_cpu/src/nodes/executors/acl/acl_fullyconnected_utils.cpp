// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "nodes/executors/acl/acl_fullyconnected_utils.hpp"

#include <arm_compute/core/CoreTypes.h>
#include <arm_compute/core/Error.h>
#include <arm_compute/core/TensorInfo.h>
#include <arm_compute/core/TensorShape.h>
#include <arm_compute/core/Types.h>
#include <arm_compute/function_info/FullyConnectedLayerInfo.h>
#include <arm_compute/runtime/NEON/functions/NECast.h>
#include <arm_compute/runtime/NEON/functions/NEFullyConnectedLayer.h>
#include <oneapi/dnnl/dnnl_types.h>

#include <any>
#include <common/c_types_map.hpp>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "acl_utils.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/primitive_desc_iface.hpp"
#include "common/utils.hpp"
#include "cpu_memory.h"
#include "cpu_shape.h"
#include "cpu_types.h"
#include "dnnl_extension_utils.h"
#include "memory_desc/cpu_memory_desc.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_memory_desc.h"
#include "nodes/common/cpu_convert.h"
#include "nodes/common/cpu_memcpy.h"
#include "nodes/common/reorder_prim.h"
#include "nodes/convert.h"
#include "nodes/executors/acl/acl_common_executor.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/fullyconnected_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "post_ops.hpp"
#include "thread_pool_imp.hpp"
#include "utils/cpu_utils.hpp"
#include "utils/debug_capabilities.h"

namespace ov::intel_cpu {

namespace {

// Set an ACL stride value, guarding against uint32 overflow (ACL strides are uint32).
dnnl::impl::status_t safe_set_strides(arm_compute::Strides& strides,
                                      size_t dim,
                                      size_t val,
                                      bool inc_dim = true) {
    if (val > std::numeric_limits<uint32_t>::max()) {
        return dnnl::impl::status::unimplemented;
    }
    strides.set(dim, val, inc_dim);
    return dnnl::impl::status::success;
}

// Fill a oneDNN blocked memory descriptor (and the matching ACL TensorInfo) that
// describes weights repacked into an ACL fixed-format weight layout.
//
// This used to live in oneDNN (dnnl::impl::cpu::acl::acl_utils::reorder_to_weight_format)
// but the oneDNN fork no longer builds ACL (it has been replaced by KleidiAI). The CPU
// plugin still uses ARM Compute Library through its own executors, so the helper is kept
// here, operating on the same oneDNN memory descriptor types.
dnnl::impl::status_t reorder_to_weight_format(arm_compute::TensorInfo& info,
                                              dnnl::impl::memory_desc_t& md,
                                              arm_compute::WeightFormat wf,
                                              dnnl::impl::dim_t I_dim,
                                              dnnl::impl::dim_t O_dim,
                                              const std::vector<dnnl::impl::dim_t>& spatial_dims,
                                              const std::vector<dnnl::impl::dim_t>& batch_dims) {
    using namespace dnnl::impl;

    md.format_kind = format_kind::blocked;
    md.format_desc.blocking = blocking_desc_t {};
    const int interleaved_by = arm_compute::interleave_by(wf);
    const int block_by = arm_compute::block_by(wf);

    // I dimension becomes densest (apart from blocking)
    md.format_desc.blocking.strides[I_dim] = interleaved_by * block_by;
    md.padded_dims[I_dim] = utils::rnd_up(md.dims[I_dim], block_by);

    // Then any spatial dimensions (e.g. HW)
    dim_t ldb = interleaved_by * md.padded_dims[I_dim];
    for (dim_t sd : spatial_dims) {
        md.format_desc.blocking.strides[sd] = ldb;
        ldb *= md.padded_dims[sd];
    }

    // O dim (which was the innermost) becomes the outermost (apart from batching)
    md.format_desc.blocking.strides[O_dim] = ldb;
    md.padded_dims[O_dim] = utils::rnd_up(md.dims[O_dim], interleaved_by);

    // Update the batch dimensions, starting with stride of the innermost batch
    const dim_t innermost_batch_stride = md.padded_dims[I_dim] * md.padded_dims[O_dim];
    dim_t batch_stride = innermost_batch_stride;
    for (dim_t bd : batch_dims) {
        md.format_desc.blocking.strides[bd] = batch_stride;
        batch_stride *= md.padded_dims[bd];
    }

    // Weights can only be blocked if they are also interleaved
    if (interleaved_by > 1) {
        md.format_desc.blocking.inner_nblks = 1 + static_cast<int>(block_by > 1);

        md.format_desc.blocking.inner_idxs[0] = O_dim;
        md.format_desc.blocking.inner_blks[0] = interleaved_by;
        if (block_by > 1) {
            md.format_desc.blocking.inner_idxs[1] = I_dim;
            md.format_desc.blocking.inner_blks[1] = block_by;
        }
    }

    if (arm_compute::is_fixed_format_fast_math(wf)) {
        md.data_type = dnnl_bf16;
        info.set_data_type(arm_compute::DataType::BFLOAT16);
    }

    // The data layout is now determined by the manually set strides
    info.set_data_layout(arm_compute::DataLayout::UNKNOWN);

    // y is the leading dimension of b (ldb) in the GEMM d = a*b + c
    // z is the (collapsed) batch dimension stride
    arm_compute::Strides new_strides_in_bytes = info.strides_in_bytes();
    CHECK(safe_set_strides(new_strides_in_bytes, 1, ldb * info.element_size()));
    CHECK(safe_set_strides(new_strides_in_bytes, 2, innermost_batch_stride * info.element_size()));

    info.init(info.tensor_shape(),
              info.num_channels(),
              info.data_type(),
              new_strides_in_bytes,
              info.offset_first_element_in_bytes(),
              memory_desc_wrapper(md).size());
    return status::success;
}

}  // namespace

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
        return memoryArgs.at(ARG_DST);
    }

    if (!node::Convert::isSupportedDesc(input->getDesc()) || !node::Convert::isSupportedDesc(output->getDesc())) {
        return {};
    }

    const auto* data = static_cast<const uint8_t*>(input->getData());
    std::vector<uint8_t> tmpBuff;
    tmpBuff.resize(output->getSize());
    cpu_parallel_convert(data,
                         tmpBuff.data(),
                         DnnlExtensionUtils::DataTypeToElementType(input->getDataType()),
                         weightPrecision,
                         input->getSize() / input->getDesc().getPrecision().size());

    return std::make_shared<Memory>(output->getPrimitive().get_engine(),
                                    output->getDesc().cloneWithNewPrecision(weightPrecision),
                                    tmpBuff.data());
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
            dnnl::stream loc_stream = make_stream(output->getPrimitive().get_engine(), context->getThreadPool());
            reorderWithoutConvert.execute(
                loc_stream,
                {{DNNL_ARG_FROM, convertOutput->getPrimitive()}, {DNNL_ARG_TO, output->getPrimitive()}});
            return output;
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
    OPENVINO_ASSERT(input->getDesc().isDefined() && output->getDesc().isDefined(),
                    "Can't reorder data with dynamic shapes");

    if (input->getShape().hasZeroDims() || output->getShape().hasZeroDims()) {
        return output;
    }

    if (input->getDesc().isCompatible(output->getDesc())) {
        auto* srcPtr = static_cast<uint8_t*>(input->getData());
        auto* dstPtr = static_cast<uint8_t*>(output->getData());
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
        dnnl::stream loc_stream = make_stream(engine, context->getThreadPool());
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
        return static_cast<MemoryPtr>(*weightCache->findOrCreate(string_hash, create));
    }

    DEBUG_LOG("ACLFullyConnectedExecutor: Weights cache is not available");
    return create();
}

MemoryPtr acl_fc_executor::prepareWeightMemory(const MemoryArgs& memory,
                                               const ExecutorContext::CPtr& context,
                                               const FCAttrs& attrs,
                                               ACLFCAttrs& aclfcAttrs,
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
    auto aclWeightsRepack = std::make_shared<acl_fc_executor::ACLWeightFormatGenerator>(attrs, memoryArgs);
    bool isNeededReorder = aclWeightsRepack->update(memoryArgs);
    expectedWeightFormat =
        isNeededReorder ? aclWeightsRepack->getOptImplWeightFormat() : arm_compute::WeightFormat::UNSPECIFIED;
    weiTensorInfo = aclWeightsRepack->getTensorInfo(ACLArgs::ACL_WEI);

    if (isNeededReorder) {
        dnnl::impl::dim_t o_dim = 0;
        dnnl::impl::dim_t inner_dim = 1;
        std::vector<dnnl::impl::dim_t> remaining_dims = {};
        auto* weights_md_ = dnnlDstDesc->getDnnlDesc().get();
        reorder_to_weight_format(weiTensorInfo,
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
        if (const auto* const activation = std::any_cast<ActivationPostOp>(postOps.data())) {
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
                        arm_compute::FullyConnectedLayerInfo& fullyConnectedLayerInfo) {
    aclTensorAttrs.hasLayoutTypeNHWC = memory.at(ARG_SRC)->getDescPtr()->hasLayoutType(LayoutType::nspc);
    fullyConnectedLayerInfo.weights_trained_layout = getAclDataLayoutByMemoryDesc(memory.at(ARG_WEI)->getDescPtr());
    aclfcAttrs.inputPrecision = memory.at(ARG_SRC)->getDescPtr()->getPrecision();
    fullyConnectedLayerInfo.transpose_weights = false;
    aclfcAttrs.weightsNonTransposed = attrs.weightsNonTransposed;

    if (checkPostOps(attrs.postOps)) {
        const auto& activation = std::any_cast<const ActivationPostOp&>(attrs.postOps[0]);
        fullyConnectedLayerInfo.activation_info = getActivationLayerInfo(convertToEltwiseAlgorithm(activation.type()),
                                                                         activation.alpha(),
                                                                         activation.beta(),
                                                                         activation.gamma());
    }

    if (memory.at(ARG_SRC)->getPrecision() != memory.at(ARG_WEI)->getPrecision()) {
        aclfcAttrs.isConvertedWeights = true;
    }
}

arm_compute::TensorShape acl_fc_executor::normalizeDimsTo2D(const arm_compute::TensorShape shape) {
    size_t norm_dim = std::accumulate(shape.begin() + 1, shape.end(), 1, std::multiplies<>());
    return {shape[0], norm_dim};
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

acl_fc_executor::ACLWeightFormatGenerator::ACLWeightFormatGenerator(const FCAttrs& attrs, const MemoryArgs& memory) {
    initFCAttrs(attrs, aclTensorAttrs, aclfcAttrs, memory, fullyConnectedLayerInfo);
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

ACLFunction acl_fc_executor::ACLWeightFormatGenerator::configureFunction(
    [[maybe_unused]] const ACLTensors& aclMemoryTensors) {
    return std::make_unique<arm_compute::NEFullyConnectedLayer>();
}

}  // namespace ov::intel_cpu
