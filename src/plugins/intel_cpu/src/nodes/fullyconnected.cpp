// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fullyconnected.h"

#include "eltwise.h"
#include "input.h"
#include "fake_quantize.h"
#include "input.h"
#include "memory_desc/blocked_memory_desc.h"
#include "reorder.h"
#include "transformations/cpu_opset/common/op/fully_connected.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "dnnl_extension_utils.h"
#include "onednn/dnnl.h"
#include "utils/general_utils.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "utils/cpu_utils.hpp"
#include "math.h"

#include "onednn/dnnl.h"
#include "oneapi/dnnl/dnnl.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "common/primitive_hashing_utils.hpp"
#include "common/primitive_desc.hpp"
#include "common/primitive_desc_iface.hpp"
#include "common/cpu_convert.h"
#include "shape_inference/custom/fullyconnected.hpp"

#include "ie_parallel.hpp"
#include "common/dnnl_thread.hpp"
#include "tbb/blocked_range.h"
#include "tbb/blocked_range2d.h"
#include "tbb/blocked_range3d.h"
#include "tbb/parallel_for.h"
#include "tbb/parallel_reduce.h"
#include "tbb/parallel_sort.h"
#include "tbb/task_arena.h"
#include "tbb/task_scheduler_observer.h"
#include "tbb/task_group.h"


#include <string>
#include <vector>
#include <chrono>

#ifdef OV_CPU_WITH_MLAS
#include "mlas/sgemm.hpp"
#endif

using namespace dnnl;
using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {
namespace {

struct FCKey {
    DnnlMemoryDescCPtr inp0;
    DnnlMemoryDescCPtr inp1;
    DnnlMemoryDescCPtr bias;
    DnnlMemoryDescCPtr out;
    dnnl::primitive_attr attr;
    impl_desc_type implType;
    bool useConv1x1;

    size_t hash() const;
    bool operator==(const FCKey& rhs) const;
};

size_t FCKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;

    for (const auto& ptr : {inp0, inp1, bias, out}) {
        if (ptr) {
            seed = hash_combine(seed, get_md_hash(*ptr->getDnnlDesc().get()));
        }
    }

    seed = hash_combine(seed, get_attr_hash(*attr.get()));
    seed = hash_combine(seed, implType);
    seed = hash_combine(seed, useConv1x1);
    return seed;
}

bool FCKey::operator==(const FCKey &rhs) const {
    bool retVal = true;
    if (inp0 != rhs.inp0) {
        retVal = retVal && inp0 && rhs.inp0 && inp0->getDnnlDesc() == rhs.inp0->getDnnlDesc();
    }
    if (inp1 != rhs.inp1) {
        retVal = retVal && inp1 && rhs.inp1 && inp1->getDnnlDesc() == rhs.inp1->getDnnlDesc();
    }
    if (bias != rhs.bias) {
        retVal = retVal && bias && rhs.bias && bias->getDnnlDesc() == rhs.bias->getDnnlDesc();
    }
    if (out != rhs.out) {
        retVal = retVal && out && rhs.out && out->getDnnlDesc() == rhs.out->getDnnlDesc();
    }
    retVal = retVal && *attr.get() == *rhs.attr.get() &&
             implType == rhs.implType && useConv1x1 == rhs.useConv1x1;
    return retVal;
}

} // namespace

bool FullyConnected::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto fc = std::dynamic_pointer_cast<const FullyConnectedNode>(op);
        if (!fc) {
            errorMessage = "Only legacy FullyConnected operation is supported";
            return false;
        }
        if (fc->get_input_size() == 3 && std::dynamic_pointer_cast<const ngraph::opset1::Constant>(fc->get_input_node_shared_ptr(BIAS_ID)) == nullptr) {
            errorMessage = "Only Constant operation on 'bias' input is supported";
            return false;
        }
        const auto inRank = fc->get_input_partial_shape(DATA_ID).size();
        const auto weightRank = fc->get_input_partial_shape(WEIGHTS_ID).size();
        if (!one_of(inRank, 2u, 3u, 4u)) {
            errorMessage = "Doesn't support 'data' input with rank: " + std::to_string(inRank);
            return false;
        }
        if ((one_of(inRank, 2u, 3u) && weightRank != 2) || (inRank == 4 && weightRank != 4)) {
            errorMessage = "Doesn't support 'data' input with rank: " + std::to_string(inRank) +
                           " and 'weight' input with rank: " + std::to_string(weightRank);
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

FullyConnected::FullyConnected(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context)
        : Node(op, context, FCShapeInferFactory(op)), withBiases(false) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage))
        IE_THROW(NotImplemented) << errorMessage;

    errorPrefix = "FullyConnected node with name '" + getName() + "'";
    if (context->getConfig().fcSparseWeiDecompressionRate < 1.0f)
        minSparseRate = context->getConfig().fcSparseWeiDecompressionRate;

    expectedBiasDims = {getInputShapeAtPort(WEIGHTS_ID).getStaticDims()[0]};
}

std::vector<memory::format_tag> FullyConnected::getAvailableFormatsForDims(const Shape &dims) const {
    if (dims.getRank() == 0)
        return {memory::format_tag::x};
    else if (dims.getRank() == 1)
        return {memory::format_tag::x};
    else if (dims.getRank() == 2)
        return {memory::format_tag::nc};
    else if (dims.getRank() == 3)
        return {memory::format_tag::tnc};
    else if (dims.getRank() == 4)
        return {memory::format_tag::nChw8c, memory::format_tag::nChw16c, memory::format_tag::nhwc, memory::format_tag::nchw};
    else if (dims.getRank() == 5)
        return {memory::format_tag::nCdhw8c, memory::format_tag::nCdhw16c, memory::format_tag::ndhwc, memory::format_tag::ncdhw};
    return {memory::format_tag::any};
}

VectorDims FullyConnected::makeDummyInputDims() const {
    const auto& inShape = getInputShapeAtPort(DATA_ID);
    const auto& weightDims = getInputShapeAtPort(WEIGHTS_ID).getStaticDims();

    auto inMinDims = inShape.getMinDims();
    auto inMaxDims = inShape.getMaxDims();

    if (inMinDims.size() == 3) {
        inMinDims.back() = weightDims.back();
        inMaxDims.back() = weightDims.back();
    } else {
        for (size_t i = 1; i < inMinDims.size(); i++) {
            inMinDims[i] = weightDims[i];
            inMaxDims[i] = weightDims[i];
        }
    }
    return MemoryDescUtils::makeDummyShape(Shape(inMinDims, inMaxDims)).getStaticDims();
}

VectorDims FullyConnected::makeDummyOutputDims(const VectorDims& inDims) const {
    std::vector<Shape> inShapes = {Shape(inDims), getInputShapeAtPort(WEIGHTS_ID)};
    if (inputShapes.size() > 2) {
        inShapes.emplace_back(getInputShapeAtPort(BIAS_ID));
    }
    return shapeInferGeneric(inShapes).front();
}

bool FullyConnected::canBeExecutedInInt8() const {
    auto firstInputPrecision = getOriginalInputPrecisionAtPort(0);
    auto secondInputPrecision = getOriginalInputPrecisionAtPort(1);

    return one_of(firstInputPrecision, Precision::U8, Precision::I8) && secondInputPrecision == Precision::I8;
}

void FullyConnected::getSupportedDescriptors() {
    if (getParentEdges().size() != 2 && getParentEdges().size() != 3)
        IE_THROW() << errorPrefix << " has incorrect number of input edges";
    if (getChildEdges().empty())
        IE_THROW()<< errorPrefix << " has incorrect number of output edges";

    auto inputDataType = DnnlExtensionUtils::IEPrecisionToDataType(getOriginalInputPrecisionAtPort(DATA_ID));
    outputDataType = DnnlExtensionUtils::IEPrecisionToDataType(getOriginalOutputPrecisionAtPort(DATA_ID));

    if (!fusedWith.empty()) {
        outputDataType = DnnlExtensionUtils::IEPrecisionToDataType(fusedWith[fusedWith.size() - 1]->getOriginalOutputPrecisionAtPort(0));
    }
    auto weightsDataType = DnnlExtensionUtils::IEPrecisionToDataType(getOriginalInputPrecisionAtPort(WEIGHTS_ID));

    withBiases = getOriginalInputsNumber() == 3;

    useSparseWeights = useSparseWeightsDecompression();
    useWeightsDecompressionImpl = dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2) &&
                                  one_of(inputDataType, memory::data_type::f32, memory::data_type::bf16) &&
                                  weightsDataType == memory::data_type::u8;

    // revert back outputDataType on special cases
    if (inputDataType == memory::data_type::f32) {
        // oneDNN only support f32 output when input is f32, even if FQ is fused
        outputDataType = memory::data_type::f32;
    } else if (inputDataType == memory::data_type::bf16) {
        // bf16 input only supports bf16/f32 output, even if FQ is fused as post-ops
        if (one_of(outputDataType , memory::data_type::u8, memory::data_type::s8)) {
            outputDataType = memory::data_type::bf16;
        }
    } else if (inputDataType == memory::data_type::f16) {
#if defined(OV_CPU_WITH_ACL)
        // acl fc does not support precisions conversion
        outputDataType = weightsDataType = memory::data_type::f16;
#else
        // f16 input only supports f16/f32 output, even if FQ is fused as post-ops
        if (!one_of(outputDataType , memory::data_type::f32, memory::data_type::f16)) {
            outputDataType = memory::data_type::f16;
        }
#endif
    } else if (one_of(inputDataType, memory::data_type::u8, memory::data_type::s8)) {
        if (weightsDataType != memory::data_type::s8) {
            // weight has to be s8 for INT8 mode, otherwise fallback to
            // f32 mode
            inputDataType = outputDataType = memory::data_type::f32;
        } else if (one_of(outputDataType, memory::data_type::f16)) {
            // INT8 inner-product only supports u8/s8/s32/f32/bf16,
            // other precision needs fallback to f32
            outputDataType = memory::data_type::f32;
        }
    } else {
        // s32/u32/... unsupported input data types, fallback to f32
        inputDataType = outputDataType = memory::data_type::f32;
    }

    inDims = isDynamicNode() ? makeDummyInputDims() : getInputShapeAtPort(DATA_ID).getStaticDims();
    outDims = isDynamicNode() ? makeDummyOutputDims(inDims) : getOutputShapeAtPort(0).getStaticDims();
#if defined(OV_CPU_WITH_MLAS) && (defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64))
    // MLAS doesn't support post-ops fusing and only supports FP32. INT8 is not enabled yet
    // Disable MLAS when FC could fuse post-ops
    useMlas = !useSparseWeights && !useWeightsDecompressionImpl &&
              (inputDataType == memory::data_type::f32 && weightsDataType == memory::data_type::f32) &&
              fusedWith.empty();
    auto wgtDims = getInputShapeAtPort(WEIGHTS_ID).getStaticDims();
    // MLAS cannot support weight dims > 2, e.g. [1,64,9,9] * [10,64,9,9]
    if (useMlas && wgtDims.size() > 2) {
        bool allOnes = true;
        for (size_t i = 2; i < wgtDims.size(); i++) {
            allOnes = allOnes && wgtDims[i] == 1;
        }
        useMlas = useMlas && allOnes;
    }
    if (useMlas && withBiases) {
        const auto& biasDims = getInputShapeAtPort(BIAS_ID).getStaticDims();
        bool isByChannel = biasDims.back() == outDims.back();
        for (size_t i = 0; i < biasDims.size() - 1; i++) {
            isByChannel = isByChannel && biasDims[i] == 1;
        }
        useMlas = useMlas && isByChannel;
    }
#endif
#ifdef CPU_DEBUG_CAPS
    // Select Sgemm type by ENV MLAS/ONEDNN, MLAS is used by default
    if (getenv("OV_CPU_FC_EXEC_TYPE")) {
        if (std::string(getenv("OV_CPU_FC_EXEC_TYPE")) != "MLAS") {
            useMlas = false;
        }
    }
#endif
    if (useMlas) return;

    for (auto format : getAvailableFormatsForDims(getInputShapeAtPort(0))) {
        auto in_candidate = dnnl::memory::desc(DnnlExtensionUtils::convertToDnnlDims(inDims), inputDataType, format);
        auto out_candidate = dnnl::memory::desc(DnnlExtensionUtils::convertToDnnlDims(outDims), outputDataType, dnnl::memory::format_tag::any);

        createDescriptorInternal(in_candidate, out_candidate);
    }
}

#ifdef OV_CPU_WITH_MLAS
void FullyConnected::prepackMLASWeight() {
    auto prepareMLASWeight = [&](const int64_t N, const int64_t K) {
        if (!getParentEdgeAt(WEIGHTS_ID)->getParent()->isConstant())
            IE_THROW() << "Weight input is not const for node " << getName() << ".";
        auto weightsMem = getParentEdgeAt(WEIGHTS_ID)->getMemoryPtr();
        if (!weightsMem)
            IE_THROW() << "Cannot get const weights edgeMem for node " << getName() << ".";
        auto packedBsize = mlas_sgemm_pack_get_size(N, K);
        MemoryPtr ptr;
        auto create = [&]() {
            float* weightPtr = reinterpret_cast<float*>(weightsMem->getData());
            size_t ldb = weightsNonTransposed ? N : K;
            MemoryPtr _ptr =
                std::make_shared<Memory>(getEngine(),
                                         intel_cpu::CpuBlockedMemoryDesc(Precision::I8, intel_cpu::Shape{packedBsize}));
            float* prepackedDst = reinterpret_cast<float*>(_ptr->getData());
            mlas_sgemm_pack(weightsNonTransposed ? "F" : "T", N, K, ldb, weightPtr, prepackedDst);
            return _ptr;
        };

        auto weightCache = context->getWeightsCache();
        if (weightCache != nullptr) {
            std::string format = "gemm_mlas_" + std::to_string(N) + "_" + std::to_string(K);
            const std::string string_hash = getName() + "_" + format + "_" + std::to_string(weightsMem->getSize()) +
                                            "_" + std::to_string(reinterpret_cast<uint64_t>(weightsMem->getData()));

            ptr = *weightCache->findOrCreate(string_hash, create);
        } else {
            ptr = create();
        }
        return ptr;
    };
    const auto& wgtDims = getParentEdgeAt(WEIGHTS_ID)->getMemoryPtr()->getStaticDims();
    // Weights are transposed by MatMulConstTransposesExtraction
    // K is the IC of weight
    // the weight is reshaped to [-1, K] in ConvertMatMulToFC
    K = wgtDims[1];
    N = wgtDims[0];

    mlasPackedPtr = prepareMLASWeight(N, K);
}
#endif

void FullyConnected::createPrimitive() {
#ifdef OV_CPU_WITH_MLAS
    if (useMlas) {
        Node::createPrimitive();
        prepackMLASWeight();
        return;
    }
#endif
    setPostOps(attr, outDims);
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    Node::createPrimitive();
    appendPostOpArgs(attr, primArgs, postOpsArgs);
}

void FullyConnected::prepareParams() {
    auto srcMemPtr = getParentEdgesAtPort(0)[0]->getMemoryPtr();
    auto dstMemPtr = getChildEdgesAtPort(0)[0]->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->isAllocated())
        IE_THROW() << "Destination memory hasn't been allocated.";
    if (!srcMemPtr || !srcMemPtr->isAllocated())
        IE_THROW() << "Input memory hasn't been allocated.";
    MemoryPtr biasMemPtr = nullptr;
    if (withBiases) {
        biasMemPtr = getParentEdgesAtPort(2)[0]->getMemoryPtr();
        if (!biasMemPtr || !biasMemPtr->isAllocated())
            IE_THROW() << "Input memory hasn't been allocated.";
    }

    NodeDesc *selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        IE_THROW() << "Preferable primitive descriptor is not set for node " << getName() << ".";
#ifdef OV_CPU_WITH_MLAS
    // M should be normalized and updated
    if (useMlas) {
        outDims = dstMemPtr->getStaticDims();
        if (outDims.size() > 2) {
            M = std::accumulate(outDims.begin(), outDims.end() - 1, 1, std::multiplies<size_t>());
        } else {
            M = outDims[0];
        }
        return;
    }
#endif
    DnnlMemoryDescPtr weightDesc = MemoryDescUtils::convertToDnnlMemoryDesc(weightDescIP);
    DnnlMemoryDescCPtr biasDesc = nullptr;
    if (biasMemPtr) {
        biasDesc = biasMemPtr->getDescWithType<DnnlMemoryDesc>();
    }

    DnnlMemoryDescCPtr inDesc = srcMemPtr->getDescWithType<DnnlMemoryDesc>();
    DnnlMemoryDescCPtr outDesc = dstMemPtr->getDescWithType<DnnlMemoryDesc>();

    useConv1x1 = canBeExecutedInConv1x1();
    FCKey key = {inDesc,
                 weightDesc,
                 biasDesc,
                 outDesc,
                 attr,
                 implementationTypeIP,
                 useConv1x1};

    auto& engine = getEngine();

    auto builder = [&engine](const FCKey& key) -> executorPtr {
        // use conv1x1 primitive for computation
        if (key.useConv1x1) {
            auto prim_desc = createDescriptorInternalForConv(key.inp0, key.inp1, key.bias, key.out, key.attr, engine);
            const bool found = DnnlExtensionUtils::find_implementation(prim_desc, brgconv_avx512_1x1);

            if (found)
                return std::make_shared<DnnlExecutor>(prim_desc);
        }

        // fallback to normal inner product primitive
        auto inDesc = key.inp0->getDnnlDesc();
        const auto& inDims = inDesc.get_dims(); // @TODO query + copy might be slow
        if (inDims.size() == 3) {
            auto normalizedInDims = {inDims[0] * inDims[1], inDims[2]};
            inDesc = inDesc.reshape(normalizedInDims);
        }
        auto outDesc = key.out->getDnnlDesc();
        const auto& outDims = outDesc.get_dims(); // @TODO query + copy might be slow

        if (outDims.size() == 3) {
            auto normalizedOutDims = { outDims[0] * outDims[1], outDims[2] };
            outDesc = outDesc.reshape(normalizedOutDims);
        }
        auto wghDescAny = dnnl::memory::desc(DnnlExtensionUtils::convertToDnnlDims(key.inp1->getShape().getStaticDims()),
                        key.inp1->getDataType(), memory::format_tag::any);
        dnnl::inner_product_forward::primitive_desc prim_desc;
        if (key.bias) {
            prim_desc = dnnl::inner_product_forward::primitive_desc(
                engine,
                dnnl::prop_kind::forward_inference,
                inDesc,
                wghDescAny,
                key.bias->getDnnlDesc(),
                outDesc,
                key.attr);
        } else {
            prim_desc = dnnl::inner_product_forward::primitive_desc(
                engine,
                dnnl::prop_kind::forward_inference,
                inDesc,
                wghDescAny,
                outDesc,
                key.attr);
        }
        auto first_desc = dnnl::inner_product_forward::primitive_desc(prim_desc.get());
        const bool found = DnnlExtensionUtils::find_implementation(prim_desc, key.implType);

        if (found)
            return std::make_shared<DnnlExecutor>(prim_desc);

        // For dynamic shape, the expected implement type kernel can support with dummy shape but
        // not the run time inference shape. In this case, the implementation type will be
        // ignored and the first available primitive descriptor will be chosen
        return std::make_shared<DnnlExecutor>(first_desc);
    };

    auto cache = context->getParamsCache();
    auto result = cache->getOrCreate(key, builder);

    if (!result.first) {
        IE_THROW() << "Primitive descriptor was not found for node " << getName() << ".";
    }

    auto prevExecPtr = execPtr;
    execPtr = result.first;

    if (execPtr) {
        if (execPtr->getSrcDesc()->isCompatible(*inDesc)) {
            primArgs[DNNL_ARG_SRC] = srcMemPtr->getPrimitive();
        } else {
            primArgs[DNNL_ARG_SRC] = dnnl::memory(execPtr->getDnnlSrcDesc(), engine, srcMemPtr->getData());
        }

        if (execPtr->getDstDesc()->isCompatible(*outDesc)) {
            primArgs[DNNL_ARG_DST] = dstMemPtr->getPrimitive();
        } else {
            primArgs[DNNL_ARG_DST] = dnnl::memory(execPtr->getDnnlDstDesc(), engine, dstMemPtr->getData());
        }

        if (!prevExecPtr || !execPtr->getWeightDesc()->isCompatible(*(prevExecPtr->getWeightDesc()))) {
            if (weightsNonTransposed) {
                primArgs[DNNL_ARG_WEIGHTS] = prepareWeightMemory(execPtr->getWeightDesc(), makeTransposedWeightDescriptor())->getPrimitive();
            } else {
                primArgs[DNNL_ARG_WEIGHTS] = prepareWeightMemory(execPtr->getWeightDesc())->getPrimitive();
            }
        }
        // changed shapes may also cause the kernel type changed
        selected_pd->setImplementationType(execPtr->getImplementationType());
        // WA: We update implType to know whether weights decompression was used inside the kernel
        if (selected_pd->getImplementationType() == ov::intel_cpu::brgemm_avx512_amx && useSparseWeights) {
            selected_pd->setImplementationType(ov::intel_cpu::brgemm_sparse_avx512_amx);
        }
        // maybe expected 1x1 conv is not created, update the flag depends on the real type
        useConv1x1 = execPtr->getImplementationType() == brgconv_avx512_1x1;

        if (withBiases) {
            primArgs[DNNL_ARG_BIAS] = biasMemPtr->getPrimitive();
        }

        auto schratchpadMem = getScratchPadMem(execPtr->getScratchPadDesc());
        primArgs[DNNL_ARG_SCRATCHPAD] = schratchpadMem->getPrimitive();
#ifdef CPU_DEBUG_CAPS
        if (result.second == CacheEntryBase::LookUpStatus::Miss) {
            auto pd = execPtr->getPrimitiveDesc();
            DEBUG_LOG("verbose##", getName(), "##", DnnlExtensionUtils::query_pd_info(pd), "\n");
        }
#endif
    } else {
        IE_THROW() << "Executor is not created for node " << getName() << ".";
    }
}

#ifdef OV_CPU_WITH_MLAS
void FullyConnected::executeMLAS() {
    const auto dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    const auto src0MemPtr = getParentEdgeAt(0)->getMemoryPtr();
    const auto biasMemPtr = withBiases ? getParentEdgeAt(BIAS_ID)->getMemoryPtr() : nullptr;
    int64_t lda = K;
    int64_t ldb = K;
    int64_t ldc = N;
    mlas_sgemm_compute("N",
                       "N",
                       M,
                       N,
                       K,
                       1.0f,
                       reinterpret_cast<float*>(src0MemPtr->getData()),
                       lda,
                       reinterpret_cast<float*>(mlasPackedPtr->getData()),
                       ldb,
                       0.0f,
                       reinterpret_cast<float*>(dstMemPtr->getData()),
                       ldc,
                       withBiases ? reinterpret_cast<float*>(biasMemPtr->getData()) : nullptr);
}

#endif

void FullyConnected::execute(dnnl::stream strm) {
#ifdef OV_CPU_WITH_MLAS
    if (useMlas) {
        executeMLAS();
        return;
    }
#endif
    auto thread_num = static_cast<size_t>(parallel_get_max_threads());
    std::atomic<size_t> dummy_parallel_get_idx{0};
    std::atomic<size_t> dummy_workload_idx{0};
    // std::cout << "[FullyConnected] thread_num: " << getName() << " :" << thread_num << "\n";

    //ov::parallel_nt_static(thread_num, [&](size_t workload_idx, size_t workload_total) {
    /*tbb::parallel_for(0, static_cast<int>(thread_num), [&](int workload_idx) {
        // std::this_thread::sleep_for(std::chrono::milliseconds(5));
        auto parallel_get_idx = parallel_get_thread_num();
        // std::cout << "-----parallel_nt_static----parallel_get_idx: " << parallel_get_idx
        //           << " workload_idx:" << workload_idx << "\n";
        // pid_t cur_sys_tid = syscall(__NR_gettid);
        // else if (cur_sys_tid != infos[idx].pid)
        //     printf("idx %d cur_sys_tid pid %d, first pid %d\n", static_cast<int>(idx),
        //     static_cast<int>(cur_sys_tid), static_cast<int>(infos[idx].pid));
        // if (parallel_get_idx != workload_idx)
        //     printf("workload_idx %d parallel_get_idx %d\n",
        //            static_cast<int>(workload_idx),
        //            static_cast<int>(parallel_get_idx));
        // printf("workload_idx %d parallel_get_idx %d\n", static_cast<int>(workload_idx),
        // static_cast<int>(parallel_get_idx));

        cpu_set_t mask;
        long nproc, i;

        if (sched_getaffinity(0, sizeof(cpu_set_t), &mask) == -1) {
            perror("sched_getaffinity");
            assert(false);
        }
        for (i = 0; i < 400; i++)
            if (CPU_ISSET(i, &mask))
                break;
        if (infos[parallel_get_idx].core_id < 0)
            infos[parallel_get_idx].core_id = i;
        else if (infos[parallel_get_idx].core_id != i)
            printf("parallel_get_idx %d actual core %d, first core %d\n",
                   static_cast<int>(parallel_get_idx),
                   static_cast<int>(i),
                   static_cast<int>(infos[parallel_get_idx].core_id));
        dummy_parallel_get_idx += parallel_get_idx;
        dummy_workload_idx += workload_idx;
        // std::this_thread::sleep_for(std::chrono::milliseconds(5));
    });

    auto expected = (thread_num - 1) * thread_num / 2;
    if (expected != dummy_parallel_get_idx) {
        std::cout << getName() << " use default fc, thread: " << thread_num
                  << " dummy_parallel_get_idx: " << dummy_parallel_get_idx
                  << " dummy_workload_idx: " << dummy_workload_idx << " expected: " << expected << "\n";
    }*/

    if (!execPtr) {
        IE_THROW() << "Can't execute FullyConnected node with name: " << getName() << ", because executor is not compiled";
    }

    // in cases parameter -> FullyConnected or dynamic shapes
    // we keep old pointer to data in primArgs on second iteration with same input shapes
    auto updateMemoryPtr = [this](int argType) {
        auto param = primArgs.find(argType);
        if (param != primArgs.end()) {
            if (argType == DNNL_ARG_SRC && (getInputShapeAtPort(DATA_ID).getRank() == 3 || useConv1x1)) {
                primArgs.at(argType).set_data_handle(getParentEdgesAtPort(0)[0]->getMemoryPtr()->getData());
            }
            if (argType == DNNL_ARG_DST && (getOutputShapeAtPort(0).getRank() == 3 || useConv1x1)) {
                primArgs.at(argType).set_data_handle(getChildEdgesAtPort(0)[0]->getMemoryPtr()->getData());
            }
        }
    };

    parallel_mt_sockets(
        thread_num,
        [&](size_t workload_idx, size_t workload_total) {
#define PI 3.14159265
            auto parallel_get_idx = parallel_get_thread_num();
            auto param = 0.0;
            auto res = 0.0;
            for (auto i = 0; i < 1000; i++) {
                param += 0.01;
                auto s = sin(param * PI / 180.0);
                auto c = cos(param * PI / 180.0);
                auto t = tan(param * PI / 180.0);
                res += s * c * t;
            }
            if (res > 10.0 && res < 10.1) {
                std::cout << "xxx\n";
            }

            cpu_set_t mask;
            long nproc, i;

            if (sched_getaffinity(0, sizeof(cpu_set_t), &mask) == -1) {
                perror("sched_getaffinity");
                assert(false);
            }
            for (i = 0; i < 400; i++)
                if (CPU_ISSET(i, &mask))
                    break;
            if (infos[parallel_get_idx].core_id < 0)
                infos[parallel_get_idx].core_id = i;
            else if (infos[parallel_get_idx].core_id != i)
                // printf("parallel_get_idx %d actual core %d, first core %d\n",
                //     static_cast<int>(parallel_get_idx),
                //     static_cast<int>(i),
                //     static_cast<int>(infos[parallel_get_idx].core_id));
                // std::cout << "[cpu id] workload_idx: " << static_cast<int>(workload_idx) << " cpu: " << i << " tid: "
                // << tid << "\n";
                dummy_parallel_get_idx += parallel_get_idx;
            dummy_workload_idx += workload_idx;
        },
        2);

    // updateMemoryPtr(DNNL_ARG_SRC);
    // updateMemoryPtr(DNNL_ARG_DST);

    execPtr->exec(primArgs, strm);
}

void FullyConnected::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

bool FullyConnected::canFuse(const NodePtr& node) const {
    return canFuseSimpleOperation(node);
}

void FullyConnected::setPostOps(dnnl::primitive_attr& attr, const VectorDims& dims_ext) {
    dnnl::post_ops ops;

    // accoridng to https://oneapi-src.github.io/oneDNN/dev_guide_inner_product.html
    // oneDNN inner product primitive's input & output tensors are always 2D:
    //   input: [N, IC]  weight: [OC, IC]   bias: [OC]   output:[N,OC]
    //
    // when input output tensors have spatial dimensions, they are flattened to 2D.
    // and following type of MatMul will be converted into FullyConnected inside CPU plugin:
    //    2D:   [X,Y] [Y,Z] =>   [X,Z]   with    N=X,IC=Y,OC=Z
    //    3D: [B,X,Y] [Y,Z] => [B,X,Z]   with  N=B*X,IC=Y,OC=Z

    VectorDims dims;
    if (dims_ext.size() == 2) {
        // 2D
        dims = dims_ext;
    } else if (dims_ext.size() == 3) {
        // 3D
        dims.push_back(dims_ext[0] * dims_ext[1]);
        dims.push_back(dims_ext[2]);
    } else {
        IE_THROW() << "Unexpected rank(" << dims_ext.size() << ") for output tensor of node: " << getName();
    }

    DnnlPostOpsComposer dnnlpoc(getEngine(), attr, ops, postOpsArgs, dims, dims.size() - 1, canBeExecutedInInt8(),
                                    1 << 0,  getDQScales(), withBiases);

    NodeDesc *selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        IE_THROW() << "Preferable primitive descriptor is not set for node " << getName() << ".";
    // OneDNN API doesn't provide an abilitiy to query optimal layout for runtime attributes
    // As workaround we assume that all AMX IP implementations use equal internal IC block size for weights layout
    // and prepack runtime attributes accordingly for better performance
    bool withAMX = selected_pd->getImplementationType() & impl_desc_type::amx;
    int icBlock = withAMX ? 2 : 1;
    if (!decompressionMultiply.empty())
        dnnlpoc.appendDecompressionScales(decompressionMultiply, icBlock);
    if (!decompressionSubtract.empty())
        dnnlpoc.appendDecompressionZeroPoints(decompressionSubtract, icBlock);

    for (size_t i = 0; i < fusedWith.size(); ++i) {
        auto& node = fusedWith[i];
        bool isLastPostOp = (i == (fusedWith.size() - 1));

        if (auto* fakeQuantizeNode = dynamic_cast<FakeQuantize*>(node.get())) {
            fakeQuantizeNode->appendAttrPostOps(dnnlpoc, isLastPostOp, outputDataType);
            continue;
        }

        if (auto* eltwiseNode = dynamic_cast<Eltwise*>(node.get())) {
            eltwiseNode->appendAttrPostOps(dnnlpoc, isLastPostOp, outputDataType);
            continue;
        }

        IE_THROW() << "Fusing of " << NameFromType(node->getType()) << " operation to " << NameFromType(this->getType())
                   << " node is not implemented";
    }

    attr.set_post_ops(ops);
}

bool FullyConnected::created() const {
    return getType() == Type::FullyConnected;
}

const std::vector<impl_desc_type>& FullyConnected::getDefaultImplPriority() {
    static const std::vector<impl_desc_type> priorities = {
        impl_desc_type::unknown,
        impl_desc_type::acl,
        impl_desc_type::brgemm_sparse_avx512_amx,
        impl_desc_type::brgemm_avx512_amx,
        impl_desc_type::brgemm_avx512,
        impl_desc_type::brgemm_avx2,
        impl_desc_type::gemm_blas,
        impl_desc_type::gemm_avx512,
        impl_desc_type::gemm_avx2,
        impl_desc_type::gemm_avx,
        impl_desc_type::gemm_sse42,
        impl_desc_type::gemm_any,
        impl_desc_type::gemm,
        impl_desc_type::jit_gemm,
        impl_desc_type::jit_uni_dw,
        impl_desc_type::jit_uni_1x1,
        impl_desc_type::jit_uni,
        impl_desc_type::jit_avx512_dw,
        impl_desc_type::jit_avx512_1x1,
        impl_desc_type::jit_avx512,
        impl_desc_type::jit_avx2_dw,
        impl_desc_type::jit_avx2_1x1,
        impl_desc_type::jit_avx2,
        impl_desc_type::jit_avx_dw,
        impl_desc_type::jit_avx_1x1,
        impl_desc_type::jit_avx,
        impl_desc_type::jit_sse42_dw,
        impl_desc_type::jit_sse42_1x1,
        impl_desc_type::jit_sse42,
        impl_desc_type::ref,
    };

    return priorities;
}

// WA: creation DnnlMemoryDesc with format == any is prohibited
// so we create dnnl::memory::desc directly
// we need specific method and can't remove createDescriptor from base class because its used into initDescriptor
void FullyConnected::createDescriptorInternal(const dnnl::memory::desc &inputDesc,
                                              const dnnl::memory::desc &outputDesc) {
    auto create2Dcandidate = [](const dnnl::memory::desc &desc) {
        if (desc.get_dims().size() != 3) // already 2D
            return desc;

        auto inDims = desc.get_dims();
        auto normalizedInDims = {inDims[0] * inDims[1], inDims[2]};

        return dnnl::memory::desc(normalizedInDims, desc.get_data_type(),
                                  DnnlExtensionUtils::GetPlainFormatByRank(normalizedInDims.size()));
    };

    const auto in_candidate  = create2Dcandidate(inputDesc);
    const auto out_candidate = create2Dcandidate(outputDesc);

    const dnnl::memory::data_type indt = inputDesc.get_data_type();
    const dnnl::memory::data_type outdt = outputDesc.get_data_type();
    dnnl::memory::data_type wdt = indt;
    dnnl::memory::data_type bdt = outdt;

    if (useWeightsDecompressionImpl) {
        // Weights decompression case
        wdt = DnnlExtensionUtils::IEPrecisionToDataType(getOriginalInputPrecisionAtPort(WEIGHTS_ID));
    } else if (one_of(indt, dnnl::memory::data_type::bf16, dnnl::memory::data_type::f16)) {
#if defined(OPENVINO_ARCH_X86_64)
        bdt = dnnl::memory::data_type::f32;
#else
        // oneDNN ARM InnerProduct primitive supports only identical in/out data types
        bdt = dnnl::memory::data_type::f16;
#endif
    } else if (indt == dnnl::memory::data_type::u8 || indt == dnnl::memory::data_type::s8) {
        wdt = memory::data_type::s8;
        if (withBiases)
            bdt = DnnlExtensionUtils::IEPrecisionToDataType(getOriginalInputPrecisionAtPort(BIAS_ID));
    }
    // We need to explicitly specify the memory descriptor to use sparse weights decompression
    dnnl::memory::desc wgh_candidate;
    if (useSparseWeights) {
        wgh_candidate = wgh_candidate.sparse_desc(DnnlExtensionUtils::convertToDnnlDims(getInputShapeAtPort(WEIGHTS_ID).getStaticDims()),
                                                  wdt);
    } else {
        wgh_candidate = { DnnlExtensionUtils::convertToDnnlDims(getInputShapeAtPort(WEIGHTS_ID).getStaticDims()),
                                        wdt, dnnl::memory::format_tag::any };
    }

    const dnnl::primitive_attr attr;

    if (withBiases) {
        dnnl::memory::desc bias_candidate(DnnlExtensionUtils::convertToDnnlDims(expectedBiasDims), bdt,
                                            dnnl::memory::format_tag::any);
        auto desc = inner_product_forward::primitive_desc(
            getEngine(),
            prop_kind::forward_inference,
            in_candidate,
            wgh_candidate,
            bias_candidate,
            out_candidate,
            attr,
            true);

        descs.push_back(desc);
    } else {
        auto desc = inner_product_forward::primitive_desc(
            getEngine(),
            prop_kind::forward_inference,
            in_candidate,
            wgh_candidate,
            out_candidate,
            attr,
            true);

        descs.push_back(desc);
    }
}

void FullyConnected::createDescriptor(const std::vector<MemoryDescPtr> &inputDesc,
                                      const std::vector<MemoryDescPtr> &outputDesc) {
    MemoryDescPtr inpDesc;
    if (inputDesc[0]->isDefined()) {
        inpDesc = inputDesc[0];
    } else {
        inpDesc = inputDesc[0]->cloneWithNewDims(inDims);
    }

    MemoryDescPtr outDesc;
    if (outputDesc[0]->isDefined()) {
        outDesc = outputDesc[0];
    } else {
        outDesc = outputDesc[0]->cloneWithNewDims(outDims);
    }
    createDescriptorInternal(MemoryDescUtils::convertToDnnlMemoryDesc(inpDesc)->getDnnlDesc(),
                             MemoryDescUtils::convertToDnnlMemoryDesc(outDesc)->getDnnlDesc());
}

void FullyConnected::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;
    if (useMlas) {
        auto dataPrecision = getOriginalInputPrecisionAtPort(0);
        if (withBiases) {
            addSupportedPrimDesc({{LayoutType::ncsp, dataPrecision},
                            {LayoutType::ncsp, dataPrecision},
                            {LayoutType::ncsp, dataPrecision}},
                            {{LayoutType::ncsp, dataPrecision}},
                            impl_desc_type::gemm_mlas);
        } else {
            addSupportedPrimDesc({{LayoutType::ncsp, dataPrecision},
                {LayoutType::ncsp, dataPrecision}},
                {{LayoutType::ncsp, dataPrecision}},
                impl_desc_type::gemm_mlas);
        }
        return;
    }
    // 3D FC requires implicit reshape so strides should be defined
    auto supportsUndefStridesAndOffset = [&]() {
        return getOutputShapeAtPort(0).getRank() == 2;
    };

    auto addSupportedPrimitiveDescriptor = [&](const dnnl::primitive_desc& prim_desc) {
        std::vector<PortConfig> inConfs, outConfs;
        const int inPlaceOutPort = canBeInPlace() ? 0 : -1;

        for (size_t i = 0; i < descInputNumbers(); i++) {
            auto desc = getSrcMemDesc(prim_desc, i);
            const auto inputBlockedMask = (supportsUndefStridesAndOffset() && !(i == WEIGHTS_ID && useSparseWeights)) ?
                BlockedMemoryDesc::EMPTY_MASK :
                BlockedMemoryDesc::FULL_MASK;

            inConfs.emplace_back(desc, inputBlockedMask);
        }

        const auto outputBlockedMask = supportsUndefStridesAndOffset() ? BlockedMemoryDesc::EMPTY_MASK : BlockedMemoryDesc::FULL_MASK;

        for (size_t i = 0; i < descOutputNumbers(); i++) {
            auto desc = getDstMemDesc(prim_desc, i);

            outConfs.emplace_back(desc, outputBlockedMask, inPlaceOutPort);
        }

        const NodeConfig config(inConfs, outConfs);
        const impl_desc_type impl_type = parse_impl_name(prim_desc.impl_info_str());

        supportedPrimitiveDescriptors.emplace_back(config, impl_type);
    };

    for (auto& desc : descs) {
        auto first_desc = dnnl::primitive_desc(DnnlExtensionUtils::clone_primitive_desc(desc.get()));
        const bool first_match = customImplPriorities.empty();
        DnnlExtensionUtils::for_each_implementation(desc,
                                                    first_match,
                                                    [&](impl_desc_type implType) {
                                                        return contains(getImplPriority(), implType);
                                                    },
                                                    [&](dnnl::primitive_desc& desc) {
                                                        addSupportedPrimitiveDescriptor(desc);
                                                    });

        // fallback. if none of the primitive types is present in the priority list just add first implementation
        // @todo this fallback is not necessary if primitive priority list is filled correctly
        if (supportedPrimitiveDescriptors.empty())
            addSupportedPrimitiveDescriptor(first_desc);
    }
}

std::shared_ptr<MemoryDesc> FullyConnected::getSrcMemDesc(const dnnl::primitive_desc &prim_desc, size_t idx) const {
    auto desc = idx > 0 ? prim_desc.weights_desc(idx - 1) : prim_desc.src_desc(idx);

    if (getInputShapeAtPort(idx).getRank() == 3
        // report original plain layout for weight since it needs to be reordered dynamically at runtime
        || idx == 1) {
        return std::make_shared<CpuBlockedMemoryDesc>(
            DnnlExtensionUtils::DataTypeToIEPrecision(desc.get_data_type()), getInputShapeAtPort(idx));
    }

    if (getInputShapeAtPort(idx).isDynamic()) {
        return DnnlExtensionUtils::makeUndefinedDesc(desc, getInputShapeAtPort(idx));
    }

    return DnnlExtensionUtils::makeDescriptor(desc);
}

std::shared_ptr<MemoryDesc> FullyConnected::getDstMemDesc(const dnnl::primitive_desc &prim_desc, size_t idx) const {
    auto desc = prim_desc.dst_desc(idx);

    if (getOutputShapeAtPort(idx).getRank() == 3) {
        return std::make_shared<CpuBlockedMemoryDesc>(
            DnnlExtensionUtils::DataTypeToIEPrecision(desc.get_data_type()), getOutputShapeAtPort(idx));
    }

    if (getOutputShapeAtPort(idx).isDynamic()) {
        return DnnlExtensionUtils::makeUndefinedDesc(desc, getOutputShapeAtPort(idx));
    }

    return DnnlExtensionUtils::makeDescriptor(desc);
}

InferenceEngine::Precision FullyConnected::getRuntimePrecision() const {
    std::vector<InferenceEngine::Precision> inputPrecisions;
    // Don't take bias precision into account
    size_t inputsNumLimit = 2;
    for (size_t i = 0; i < std::min(getParentEdges().size(), inputsNumLimit); i++) {
        auto parentEdge = getParentEdgeAt(i);
        if (parentEdge && parentEdge->getStatus() == Edge::Status::Validated) {
            inputPrecisions.emplace_back(DnnlExtensionUtils::DataTypeToIEPrecision((parentEdge->getMemoryPtr()->getDataType())));
        }
    }

    return getMaxPrecision(inputPrecisions);
}

void FullyConnected::initOptimalPrimitiveDescriptor() {
    Node::initOptimalPrimitiveDescriptor();
    auto selectedPD = getSelectedPrimitiveDescriptor();
    implementationTypeIP = selectedPD->getImplementationType();
    // if convolution selected the reorder for ip is useless. Will do the reoder for ip in prepareParams
    auto constParent = getParentEdgeAt(1)->getParent();
    auto selectedParentPD = constParent->getSelectedPrimitiveDescriptor();
    auto config = selectedPD->getConfig();
    weightDescIP = config.inConfs[1].getMemDesc();
    config.inConfs[1].setMemDesc(selectedParentPD->getConfig().outConfs[0].getMemDesc());
    selectedPD->setConfig(config);
}

dnnl::convolution_forward::primitive_desc
FullyConnected::createDescriptorInternalForConv(DnnlMemoryDescCPtr inputDescPtr,
                                                DnnlMemoryDescCPtr weightDescPtr,
                                                DnnlMemoryDescCPtr biasDescPtr,
                                                DnnlMemoryDescCPtr outputDescPtr,
                                                const dnnl::primitive_attr& attr,
                                                const dnnl::engine& engine) {
    const dnnl::memory::desc &inputDesc  = inputDescPtr->getDnnlDesc();
    const dnnl::memory::desc &outputDesc = outputDescPtr->getDnnlDesc();
    const dnnl::memory::desc &weightDesc = weightDescPtr->getDnnlDesc();
    // make a fake shape: N, IC, W
    auto inDims = inputDesc.get_dims();
    dnnl::memory::dims normalizedInDims;
    if (inDims.size() == 3) {
        normalizedInDims = {inDims[0], inDims[2], inDims[1]};
    } else if (inDims.size() == 2) {
        normalizedInDims = {dnnl::memory::dim{1}, inDims[1], inDims[0]};
    }
    auto convInDesc = dnnl::memory::desc(normalizedInDims, inputDesc.get_data_type(), memory::format_tag::nwc);

    // make a fake shape: N, OC, W
    const auto& outDims = outputDesc.get_dims();
    dnnl::memory::dims normalizedOutDims;
    if (outDims.size() == 3) {
        normalizedOutDims = { outDims[0], outDims[2], outDims[1]};
    } else if (outDims.size() == 2) {
        normalizedOutDims = { dnnl::memory::dim{1}, outDims[1], outDims[0]};
    }
    auto convOutDesc = dnnl::memory::desc(normalizedOutDims, outputDesc.get_data_type(), memory::format_tag::nwc);

    // make a fake shape: OC, IC, 1
    auto weightDims = weightDesc.get_dims();
    dnnl::memory::dims normalizedWeightDims;
    normalizedWeightDims = {static_cast<dnnl::memory::dim>(weightDims[0]),
                            static_cast<dnnl::memory::dim>(weightDims[1]),
                            dnnl::memory::dim{1}};
    auto convWeightDescAny = dnnl::memory::desc(normalizedWeightDims, weightDesc.get_data_type(), dnnl::memory::format_tag::any);

    if (biasDescPtr) {
        return dnnl::convolution_forward::primitive_desc(
            engine,
            prop_kind::forward_inference,
            dnnl::algorithm::convolution_direct,
            convInDesc, convWeightDescAny, biasDescPtr->getDnnlDesc(), convOutDesc,
            dnnl::memory::dims{1},   // stride
            dnnl::memory::dims{0},   // dilation
            dnnl::memory::dims{0},   // paddingL
            dnnl::memory::dims{0},   // paddingR
            attr);
    } else {
        return dnnl::convolution_forward::primitive_desc(
            engine,
            prop_kind::forward_inference, dnnl::algorithm::convolution_direct,
            convInDesc, convWeightDescAny, convOutDesc,
            dnnl::memory::dims{1},   // stride
            dnnl::memory::dims{0},   // dilation
            dnnl::memory::dims{0},   // paddingL
            dnnl::memory::dims{0},   // paddingR
            attr);
    }
}

bool FullyConnected::canBeExecutedInConv1x1() const {
    bool retVal = false;
    const auto inRank = getInputShapeAtPort(DATA_ID).getRank();
    const auto weightRank = getInputShapeAtPort(WEIGHTS_ID).getRank();
    if (useWeightsDecompressionImpl) {
        return false;
    }
    // disable rank=4:
    // if layout is nhwc:
    //   A matrix: N * IC * H * W --> N * (IC*H*W), the M, N', K of matrix multiply will be:
    //   M = 1, K = (IC*H*W), when M = 1 it should not be efficient since acts as a vector multiply
    // if layout is nchw/nChw16c: brg1x1 not support. Although jit supports, it should have similar
    //   problems with the above.
    if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core) &&
        getOriginalInputPrecisionAtPort(DATA_ID) == InferenceEngine::Precision::FP32 &&
        one_of(inRank, 2u, 3u) && weightRank == 2) {
        auto dstMemPtr = getChildEdgesAtPort(0)[0]->getMemoryPtr();
        DnnlMemoryDescCPtr outDesc = dstMemPtr->getDescWithType<DnnlMemoryDesc>();
        // brg convolution does not support stride
        dnnl::impl::memory_desc_wrapper wrapped(outDesc->getDnnlDesc().get());
        if (wrapped.offset0() == 0)
            retVal = true;
    }

    if (retVal) {
        auto srcMemPtr = getParentEdgesAtPort(0)[0]->getMemoryPtr();
        const auto& srcDims = srcMemPtr->getStaticDims();
        auto weightMemPtr = getParentEdgesAtPort(1)[0]->getMemoryPtr();
        const auto& weightDims = weightMemPtr->getStaticDims();
        // for original inner product semantics:
        //  when input is 2D tensor
        //    M in oneDNN will map to widthInConv
        //  when input is 3D tensor
        //    M in oneDNN will map to widthInConv*minibatch
        // currently nwc mapping in brg:
        //  when input is 2D tensor
        //    widthInConv will map to 'w', 'n' will be 1
        //  when input is 3D tensor
        //    widthInConv will map to 'w', 'n' will be minibatch
        Dim widthInConv, N, K;
        widthInConv = srcDims[inRank - 2];
        K = srcDims[inRank - 1];
        N = weightDims[0];

        if (!(widthInConv >= 2 && widthInConv <= 3136 &&
              K >= 96 && K <= 4096 &&
              N >= 96 && N <= K * 4))
            retVal = false;
    }

    return retVal;
}

bool FullyConnected::useSparseWeightsDecompression() {
    // minSparseRate == 1 means that sparse feature is switched off
    if (minSparseRate == 1.f) {
        return false;
    }

    if (!impl::cpu::x64::mayiuse(impl::cpu::x64::avx512_core_amx))
        return false;

    auto weiDims = getInputShapeAtPort(WEIGHTS_ID).getStaticDims();
    if (weiDims.size() != 2 || weiDims[0] % 64 != 0 || weiDims[1] % 64 != 0) {
        return false;
    }

    auto inputPrecision = getOriginalInputPrecisionAtPort(DATA_ID);
    auto weightsPrecision = getOriginalInputPrecisionAtPort(WEIGHTS_ID);
    if (!one_of(inputPrecision , Precision::U8, Precision::I8) || weightsPrecision != Precision::I8) {
        return false;
    }

    // calculate sparse rate
    const auto constNode = std::dynamic_pointer_cast<Input>(getParentEdgeAt(WEIGHTS_ID)->getParent());
    if (!constNode) {
        return false;
    }
    auto blb = constNode->getMemoryPtr();
    if (blb == nullptr)
        IE_THROW() << "Cannot get const blob for node " << getName() << ".";

    auto weightsData = reinterpret_cast<const int8_t*>(blb->getData());
    auto elementsCount = blb->getDescWithType<BlockedMemoryDesc>()->getPaddedElementsCount();
    size_t zerosCounts = 0;
    for (size_t i = 0; i < elementsCount; i++) {
        if (weightsData[i] == 0) {
            zerosCounts++;
        }
    }

    DEBUG_LOG(getName(), ", elementsCount = ", elementsCount, ", zerosCounts = ",
        zerosCounts, ", nnzCount = ", elementsCount - zerosCounts);

    weiSparseRate = static_cast<float>(zerosCounts) / static_cast<float>(elementsCount);

    DEBUG_LOG(getName(), " | sparse rate = ", weiSparseRate * 100, "%, min sparse rate = ",
        minSparseRate * 100, "%, use sparse weights = ", weiSparseRate >= minSparseRate);

    if (weiSparseRate < minSparseRate) {
        return false;
    }

    return true;
}

void FullyConnected::fuseDecompressionMultiply(const NodePtr& constData) {
    fuseDecompressionConstant(constData, decompressionMultiply);
}

void FullyConnected::fuseDecompressionSubtract(const NodePtr& constData) {
    fuseDecompressionConstant(constData, decompressionSubtract);
}

void FullyConnected::fuseDecompressionConstant(const NodePtr& constData, std::vector<float>& decompressionValues) {
    auto *constInputNode = dynamic_cast<node::Input *>(constData.get());
    if (!constInputNode) {
        IE_THROW() << "Cannot cast " << constData->getName() << " to Input";
    }
    auto constBlob = constInputNode->getMemoryPtr();
    const auto elementsCount = constBlob->getDescWithType<BlockedMemoryDesc>()->getPaddedElementsCount();
    decompressionValues.resize(elementsCount);
    cpu_convert(constBlob->getData(),
                &decompressionValues[0],
                DnnlExtensionUtils::DataTypeToIEPrecision(constBlob->getDataType()),
                Precision::FP32,
                elementsCount);
}

DnnlMemoryDescPtr FullyConnected::makeTransposedWeightDescriptor() {
    if (!getParentEdgeAt(1)->getParent()->isConstant())
        IE_THROW() << "Weight input is not const for node " << getName() << ".";
    auto edgeMem = getParentEdgeAt(1)->getMemoryPtr();
    if (!edgeMem)
        IE_THROW() << "Cannot get const weights edgeMem for node " << getName() << ".";

    auto constDnnlMemOutDesc = edgeMem->getDescWithType<DnnlMemoryDesc>();
    auto weightSrcDesc = constDnnlMemOutDesc->getDnnlDesc();
    weightSrcDesc = {weightSrcDesc.get_dims(), weightSrcDesc.get_data_type(), memory::format_tag::ba};
    weightSrcDesc = weightSrcDesc.reshape(execPtr->getWeightDesc()->getDnnlDesc().get_dims());

    return DnnlExtensionUtils::makeDescriptor(weightSrcDesc);
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
