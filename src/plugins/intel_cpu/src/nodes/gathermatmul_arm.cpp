#include <openvino/core/visibility.hpp>
#ifdef OPENVINO_ARCH_ARM64
#    include <oneapi/dnnl/dnnl_common_types.h>
#    include <oneapi/dnnl/dnnl_types.h>

#    include <bitset>
#    include <common/primitive_hashing_utils.hpp>
#    include <common/utils.hpp>
#    include <cstddef>
#    include <cstdint>
#    include <cstring>
#    include <memory>
#    include <oneapi/dnnl/dnnl.hpp>
#    include <oneapi/dnnl/dnnl_common.hpp>
#    include <string>
#    include <tuple>
#    include <unordered_map>
#    include <utility>
#    include <vector>

#    include "common/blocked_desc_creator.h"
#    include "config.h"
#    include "cpu/x64/cpu_isa_traits.hpp"
#    include "cpu_memory.h"
#    include "cpu_types.h"
#    include "dnnl_extension_utils.h"
#    include "gathermatmul.h"
#    include "graph_context.h"
#    include "memory_desc/blocked_memory_desc.h"
#    include "memory_desc/cpu_memory_desc.h"
#    include "memory_desc/cpu_memory_desc_utils.h"
#    include "memory_desc/dnnl_memory_desc.h"
#    include "node.h"
#    include "node_config.h"
#    include "nodes/executors/executor.hpp"
#    include "onednn/iml_type_mapper.h"
#    include "openvino/core/except.hpp"
#    include "openvino/core/node.hpp"
#    include "openvino/core/parallel.hpp"
#    include "openvino/core/type.hpp"
#    include "openvino/core/type/element_type.hpp"
#    include "openvino/op/constant.hpp"
#    include "shape_inference/custom/gathermatmul.hpp"
#    include "transformations/cpu_opset/common/op/batch_gather_matmul.hpp"
#    include "transformations/cpu_opset/common/op/batch_gather_matmul_compressed.hpp"
#    include "transformations/utils/utils.hpp"
#    include "utils/general_utils.h"

//  Tensors:
//    A : Activations  [B, M, K]
//    B : Weights      [G, N, K]   (transposed, gather axis = G)
//    C : Indices      [M, B]
//    D : Output       [B, M, N]
//
//  create_primitive (once):
//    For g = 0 .. G-1:
//      Executor[g] with Weights [N,K] and Bias [N]
//
//  execute():
//    For g = 0 .. G-1:
//      gather activations for expert g
//      Executor[g] → compute
//      scatter results to D

namespace ov::intel_cpu::node {

namespace {
class OffsetHelper {
public:
    static OffsetHelper createOffsetHelper(const MemoryPtr& mem) {
        static const VectorDims empty_dims;
        std::bitset<2> broadcast_mask;
        if (nullptr == mem || mem->getDesc().empty()) {
            return {nullptr, empty_dims, broadcast_mask, 0};
        }
        return createOffsetHelper(*mem);
    }

    static OffsetHelper createOffsetHelper(const IMemory& mem) {
        std::bitset<2> broadcast_mask;
        auto* base_ptr = static_cast<uint8_t*>(mem.getData());
        auto desc = mem.getDescWithType<BlockedMemoryDesc>();
        const auto& strides = desc->getStrides();
        const auto prc = desc->getPrecision();
        const auto& shape = desc->getShape().getStaticDims();
        for (size_t i = 0; i < shape.size() && i < 2; i++) {
            if (shape[i] == 1) {
                broadcast_mask.set(i);
            }
        }
        return {base_ptr, strides, broadcast_mask, prc.bitwidth()};
    }

    void* operator()(size_t i0) const {
        if (!m_base_ptr) {
            return nullptr;
        }
        if (m_broadcast_mask.test(0)) {
            i0 = 0;
        }
        const size_t offset_bits = i0 * m_strides[0] * m_num_bits;
        const size_t offset = div_up(offset_bits, 8);  // 8 bits in byte
        return m_base_ptr + offset;
    }

    void* operator()(size_t i0, size_t i1) const {
        if (!m_base_ptr) {
            return nullptr;
        }
        if (m_broadcast_mask.test(0)) {
            i0 = 0;
        }
        if (m_broadcast_mask.test(1)) {
            i1 = 0;
        }
        const size_t offset_bits = i0 * m_strides[0] * m_num_bits + i1 * m_strides[1] * m_num_bits;
        const size_t offset = div_up(offset_bits, 8);  // 8 bits in byte
        return m_base_ptr + offset;
    }

    [[nodiscard]] void* get_base() const {
        return m_base_ptr;
    }

private:
    OffsetHelper(uint8_t* base_ptr, const VectorDims& strides, std::bitset<2> broadcast_mask, size_t num_bits)
        : m_base_ptr(base_ptr),
          m_strides(strides),
          m_num_bits(num_bits),
          m_broadcast_mask(broadcast_mask) {}

    uint8_t* m_base_ptr = nullptr;
    const VectorDims& m_strides;
    size_t m_num_bits;
    std::bitset<2> m_broadcast_mask;
};
}  // namespace

bool GatherMatmul::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const bool isBatchGatherMatmul = ov::is_type<ov::intel_cpu::BatchGatherMatmul>(op);
        if (!isBatchGatherMatmul) {
            errorMessage =
                "Only BatchGatherMatmul operations are supported. Got: " + std::string(op->get_type_info().name);
            return false;
        }

        // Check that weights input (port 1) is constant
        if (!ov::op::util::is_on_path<ov::op::v0::Constant>(op->input_value(WEIGHTS))) {
            errorMessage = "Only constant weights are supported for GatherMatmul operation";
            return false;
        }

        if (op->get_input_size() > BIAS) {
            const auto& biasInput = op->input_value(BIAS);
            // Skip validation if bias is dynamic (empty constant)
            if (biasInput.get_element_type() != ov::element::dynamic) {
                if (!ov::op::util::is_on_path<ov::op::v0::Constant>(biasInput)) {
                    errorMessage = "Only constant bias is supported for GatherMatmul operation";
                    return false;
                }
            }
        }
    } catch (...) {
        return false;
    }

    return true;
}

bool GatherMatmul::isSupportedCompressedOperation([[maybe_unused]] const std::shared_ptr<ov::Node>& op,
                                                  [[maybe_unused]] size_t IC,
                                                  [[maybe_unused]] size_t OC,
                                                  [[maybe_unused]] size_t G,
                                                  [[maybe_unused]] const Config& config) noexcept {
    // TODO: Support Compressed operation after enabling at IR level for ARM
    return false;
}

ov::element::TypeVector GatherMatmul::getSupportedCompressedWeightsTypes([[maybe_unused]] bool apply_fp8) {
    return {};
}

ov::element::TypeVector GatherMatmul::getSupportedCompressedActivationsTypes() {
    using ov::element::Type_t;
    return {};
}

void GatherMatmul::initSupportedPrimitiveDescriptors() {
    const auto& srcTypes = getOriginalInputPrecisions();
    auto dstTypes = getOriginalOutputPrecisions();

    if (!fusedWith.empty()) {
        dstTypes = fusedWith.back()->getOriginalOutputPrecisions();
    }

    NodeConfig nodeConfig;

    const auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    for (size_t i = 0; i < srcTypes.size(); i++) {
        if (srcTypes[i] == element::dynamic) {
            nodeConfig.inConfs.emplace_back(MemoryDescUtils::makeEmptyDesc());
            continue;
        }
        const auto srcDesc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(srcTypes[i], getInputShapeAtPort(i));
        nodeConfig.inConfs.emplace_back(srcDesc);
    }

    for (size_t i = 0; i < dstTypes.size(); i++) {
        const auto dstDesc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(dstTypes[i], getOutputShapeAtPort(i));
        nodeConfig.outConfs.emplace_back(dstDesc);
    }

    supportedPrimitiveDescriptors.emplace_back(nodeConfig, impl_desc_type::undef);
}

void GatherMatmul::createPrimitive() {
    auto srcMemoryDesc = getBaseMemDescAtInputPort(DATA);
    auto weiMemoryDesc = getBaseMemDescAtInputPort(WEIGHTS);
    auto biasMemoryDesc = getBaseMemDescAtInputPort(BIAS);
    const auto& weiDims = weiMemoryDesc->getShape().getStaticDims();
    auto weiPrec = weiMemoryDesc->getPrecision();
    auto SrcPrec = srcMemoryDesc->getPrecision();
    size_t N = weiDims[weiDims.size() - 2];
    size_t K = weiDims[weiDims.size() - 1];
    numExperts = weiDims[0];

    CPU_NODE_ASSERT(weiMemoryDesc->isDefined(), "Weights memory descriptor is not defined");
    CPU_NODE_ASSERT(weiPrec == ov::element::f32, "Weights currently supported only in f32");
    CPU_NODE_ASSERT(SrcPrec == ov::element::f32, "Weights currently supported only in f32");

    const auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    auto expertWeiDesc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(weiPrec, Shape({N, K}));

    auto addBatchDim = [](const BlockedMemoryDescPtr& desc, size_t batchDim) -> DnnlMemoryDescPtr {
        const auto& weightsDims = desc->getShape().getStaticDims();
        const auto& weightsBlockDims = desc->getBlockDims();
        const auto& weightsOrder = desc->getOrder();
        // at this point we assume that the tensors are dense and have no padded dims
        VectorDims newDims = {batchDim};
        newDims.insert(newDims.end(), weightsDims.begin(), weightsDims.end());
        VectorDims newBlockDims = {batchDim};
        newBlockDims.insert(newBlockDims.end(), weightsBlockDims.begin(), weightsBlockDims.end());
        VectorDims newOrder(weightsOrder.size() + 1);
        newOrder[0] = 0;
        for (size_t i = 0; i < weightsOrder.size(); i++) {
            newOrder[i + 1] = weightsOrder[i] + 1;
        }
        auto targetDesc =
            std::make_shared<CpuBlockedMemoryDesc>(desc->getPrecision(), Shape(newDims), newBlockDims, newOrder);
        return MemoryDescUtils::convertToDnnlMemoryDesc(targetDesc);
    };
    auto targetWeightsDesc = addBatchDim(expertWeiDesc, numExperts);
    m_weightsMemory = prepareWeightMemory(targetWeightsDesc, MemoryDescUtils::convertToDnnlMemoryDesc(weiMemoryDesc));

    MemoryDescArgs memDescArgs;
    // Only weights and bias are required at this stage.
    // Others are just placeholders, data is only available at execution stage.
    memDescArgs[ARG_SRC] = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(SrcPrec, Shape({1, K}));
    memDescArgs[ARG_DST] = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(SrcPrec, Shape({1, N}));
    memDescArgs[ARG_WEI] = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(weiPrec, Shape({N, K}));
    if (biasMemoryDesc && !biasMemoryDesc->empty()) {
        memDescArgs[ARG_BIAS] = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(weiPrec, Shape({N}));
        ;
    } else {
        memDescArgs[ARG_BIAS] = MemoryDescUtils::makeEmptyDesc();
    }

    auto executionContext = std::make_shared<ExecutorContext>(context, getImplPriority(), privateWeightCache);
    memArgsFC.reserve(numExperts);
    MemoryFormatFilter filter{};
    auto factory = std::make_shared<ExecutorFactory<FCAttrs>>(FCAttrs(),
                                                              executionContext,
                                                              memDescArgs,
                                                              filter,
                                                              std::string("fullyconnected_kleidiai"));
    for (size_t expert = 0; expert < numExperts; ++expert) {
        MemoryArgs FCArgs;
        FCArgs[ARG_WEI] = split_horizontal(context->getEngine(), m_weightsMemory, 0, expert, numExperts, true);
        // wei_shape shape becomes: [1, N, K] --> redefine desc to [N, K]
        FCArgs[ARG_WEI]->redefineDesc(memDescArgs[ARG_WEI]);
        if (biasMemoryDesc && !biasMemoryDesc->empty()) {
            auto bias = getSrcMemoryAtPort(BIAS);
            FCArgs[ARG_BIAS] = split_horizontal(context->getEngine(), bias, 0, expert, numExperts, true);
            FCArgs[ARG_BIAS]->redefineDesc(memDescArgs[ARG_BIAS]);
        } else {
            FCArgs[ARG_BIAS] = std::make_shared<Memory>(context->getEngine(), MemoryDescUtils::makeEmptyDesc());
        }
        FCArgs[ARG_SRC] = std::make_shared<Memory>(context->getEngine(), memDescArgs[ARG_SRC]);
        memArgsFC.emplace_back(FCArgs);
        // Currectly support only KleidiAI Executor.
        executor.push_back(factory->make(memArgsFC.back()));
    }
    Node::createPrimitive();

    // set the actual implementation type to Kleidiai
    getSelectedPrimitiveDescriptor()->setImplementationType(ov::intel_cpu::impl_desc_type::kleidiai);
}

bool GatherMatmul::needPrepareParams() const {
    return true;
}

void GatherMatmul::prepareParams() {
    auto srcMem = getSrcMemoryAtPort(DATA);
    auto indMem = getSrcMemoryAtPort(INDICES);
    auto dstMem = getDstMemoryAtPort(0);
    const auto& srcShape = srcMem->getStaticDims();
    const auto& dstShape = dstMem->getStaticDims();

    const Dim M = srcShape[1];

    const auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    const auto srcPrc = srcMem->getDesc().getPrecision();

    m_tmpInputDesc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(srcPrc, Shape({M, srcShape[2]}));
    m_tmpOutputDesc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(srcPrc, Shape({M, dstShape[2]}));

    auto srcSize = rnd_up(m_tmpInputDesc->getCurrentMemSize(), 64);  // 64 bytes is the cache line size
    const size_t totalSize = srcSize + m_tmpOutputDesc->getCurrentMemSize();
    auto scratchPadDesc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(ov::element::u8, Shape({totalSize}));
    m_tmpInpBuffer = getScratchPadMem(scratchPadDesc);
}

bool GatherMatmul::isExecutable() const {
    return !isInputTensorAtPortEmpty(0);  // only data shape matters
}

void GatherMatmul::execute(const dnnl::stream& strm) {
    const auto& srcMem = getParentEdgeAt(DATA)->getMemoryPtr();
    const auto& indexMem = getParentEdgeAt(INDICES)->getMemoryPtr();
    const auto& dstMem = getChildEdgeAt(0)->getMemoryPtr();

    auto src_offset = OffsetHelper::createOffsetHelper(srcMem);
    auto index_offset = OffsetHelper::createOffsetHelper(indexMem);
    auto dst_offset = OffsetHelper::createOffsetHelper(dstMem);

    const auto& indexShape = indexMem->getStaticDims();
    size_t M = indexShape[0];
    size_t B = indexShape[1];

    const auto& srcShape = srcMem->getStaticDims();
    const auto srcPrc = srcMem->getDesc().getPrecision();
    const Dim K = srcShape[2];
    const Dim N = dstMem->getStaticDims()[2];

    CPU_NODE_ASSERT(m_tmpInpBuffer, "Temporary input/output memory is not created");
    CPU_NODE_ASSERT(m_tmpInputDesc, "Temporary input memory desc is not created");
    CPU_NODE_ASSERT(m_tmpOutputDesc, "Temporary output memory desc is not created");

    // all the gather idx for corresponding m index
    const size_t gather_axis_size = numExperts;
    std::vector<std::pair<int32_t, int32_t>> gather_idx_map(gather_axis_size * M);
    std::vector<int32_t> elements_per_gather_indx(gather_axis_size, 0);
    for (size_t m = 0; m < M; m++) {
        const auto* gather_ids = static_cast<const int32_t*>(index_offset(m));
        for (size_t i = 0; i < B; i++) {
            int32_t gather_axis_index = gather_ids[i];
            CPU_NODE_ASSERT(gather_axis_index >= 0 && static_cast<size_t>(gather_axis_index) < gather_axis_size,
                            "Invalid gather_id ",
                            gather_axis_index,
                            " for m ",
                            m);
            auto& index = elements_per_gather_indx[gather_axis_index];
            gather_idx_map[gather_axis_index * M + index] = {m, i};
            index++;
        }
    }

    const auto element_size = m_tmpInputDesc->getPrecision().size();
    auto* input_ptr = m_tmpInpBuffer->getDataAs<uint8_t>();
    auto* output_ptr = input_ptr + rnd_up(m_tmpInputDesc->getCurrentMemSize(), 64);  // 64 bytes is the cache line size

    Memory tmpInput(getEngine(), m_tmpInputDesc, input_ptr);
    Memory tmpOutput(getEngine(), m_tmpOutputDesc, output_ptr);

    auto tmp_input_offset = OffsetHelper::createOffsetHelper(tmpInput);
    auto tmp_dst_offset = OffsetHelper::createOffsetHelper(tmpOutput);

    for (size_t gather_axis_index = 0; gather_axis_index < gather_axis_size; gather_axis_index++) {
        const size_t num_valid_rows = elements_per_gather_indx[gather_axis_index];
        if (0 == num_valid_rows) {
            continue;
        }

        parallel_for(num_valid_rows, [&](size_t m) {
            auto* dst_row = tmp_input_offset(m);
            const auto row_id = gather_idx_map[gather_axis_index * M + m].first;
            const auto batch_index = gather_idx_map[gather_axis_index * M + m].second;
            const auto* src_data = src_offset(batch_index, row_id);
            std::memcpy(dst_row, src_data, K * element_size);
        });

        const auto& creatorsMap = BlockedDescCreator::getCommonCreators();
        auto SrcDesc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(srcPrc, Shape({num_valid_rows, K}));
        auto DstDesc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(srcPrc, Shape({num_valid_rows, N}));

        auto* srcPtr = tmp_input_offset.get_base();
        auto* dstPtr = tmp_dst_offset.get_base();

        memArgsFC[gather_axis_index][ARG_SRC] = std::make_shared<Memory>(context->getEngine(), SrcDesc, srcPtr);
        memArgsFC[gather_axis_index][ARG_DST] = std::make_shared<Memory>(context->getEngine(), DstDesc, dstPtr);
        executor[gather_axis_index]->update(memArgsFC[gather_axis_index]);
        executor[gather_axis_index]->execute(memArgsFC[gather_axis_index]);

        // Immediately scatter results while they're hot in cache
        parallel_for(num_valid_rows, [&](size_t m) {
            const auto* src_row = tmp_dst_offset(m);
            const auto row_id = gather_idx_map[gather_axis_index * M + m].first;
            const auto batch_index = gather_idx_map[gather_axis_index * M + m].second;
            auto* dst_row = dst_offset(batch_index, row_id);
            std::memcpy(dst_row, src_row, N * element_size);
        });
    }
}

void GatherMatmul::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

bool GatherMatmul::created() const {
    return getType() == Type::GatherMatmul;
}

GatherMatmul::GatherMatmul(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, GatherMatmulShapeInferFactory(op)) {
    // Graph_context = context;
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    // Determine the algorithm type
    if (ov::is_type<ov::intel_cpu::BatchGatherMatmulCompressed>(op)) {
        algorithm = Algorithm::GatherMatmulCompressed;
    } else {
        algorithm = Algorithm::GatherMatmulDefault;
    }
}

}  // namespace ov::intel_cpu::node

#endif