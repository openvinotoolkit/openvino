// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "block_sparse_attention.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <tuple>

#include "cpu_types.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/op/block_sparse_attention.hpp"
#include "openvino/reference/block_sparse_attention.hpp"
#include "selective_build.h"
#include "shape_inference/shape_inference_cpu.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu::node {

BlockSparseAttention::BlockSparseAttention(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    const auto bsa_op = ov::as_type_ptr<const ov::op::v17::BlockSparseAttention>(op);
    m_block_size = bsa_op->get_block_size();
    m_causal = bsa_op->get_causal();
    m_has_mask = op->get_input_size() >= 5;
    m_has_scale = op->get_input_size() == 6;
}

bool BlockSparseAttention::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                                std::string& errorMessage) noexcept {
    try {
        if (!ov::is_type<ov::op::v17::BlockSparseAttention>(op)) {
            errorMessage = "Only opset17 BlockSparseAttention operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

void BlockSparseAttention::getSupportedDescriptors() {
    // Validation is already done in ov::op::v17::BlockSparseAttention::validate_and_infer_types().
}

void BlockSparseAttention::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    ov::element::Type dataPrec = getOriginalInputPrecisionAtPort(0);
    if (none_of(dataPrec, ov::element::f32, ov::element::bf16)) {
        dataPrec = ov::element::f32;
    }

    ov::element::Type indexPrec = getOriginalInputPrecisionAtPort(3);
    if (none_of(indexPrec, ov::element::i32, ov::element::i64)) {
        indexPrec = ov::element::i64;
    }

    std::vector<PortConfigurator> inConfs{{LayoutType::ncsp, dataPrec},
                                          {LayoutType::ncsp, dataPrec},
                                          {LayoutType::ncsp, dataPrec},
                                          {LayoutType::ncsp, indexPrec}};
    if (m_has_mask) {
        // ov::pass::ConvertPrecision (run by most plugins, including CPU) normalizes
        // `boolean`-typed graph tensors to `u8` storage before an op ever executes, splicing in
        // a Convert on that edge -- mirrors intel_cpu's own ScaledDotProductAttention node, which
        // negotiates the same port as u8 whenever the *original* (pre-conversion) input precision
        // was already u8, and as boolean otherwise.
        ov::element::Type maskPrec = getOriginalInputPrecisionAtPort(4);
        if (maskPrec != ov::element::u8) {
            maskPrec = ov::element::boolean;
        }
        inConfs.emplace_back(LayoutType::ncsp, maskPrec);
    }
    if (m_has_scale) {
        inConfs.emplace_back(LayoutType::ncsp, dataPrec);
    }

    addSupportedPrimDesc(inConfs, {{LayoutType::ncsp, dataPrec}}, impl_desc_type::ref);
}

bool BlockSparseAttention::created() const {
    return getType() == Type::BlockSparseAttention;
}

template <typename T, typename TIndex>
void BlockSparseAttention::executeImpl() {
    const auto* query = getSrcDataAtPortAs<const T>(0);
    const auto* key = getSrcDataAtPortAs<const T>(1);
    const auto* value = getSrcDataAtPortAs<const T>(2);
    const auto* blockIndices = getSrcDataAtPortAs<const TIndex>(3);
    const char* mask = m_has_mask ? getSrcDataAtPortAs<const char>(4) : nullptr;
    const T* scale = m_has_scale ? getSrcDataAtPortAs<const T>(5) : nullptr;
    auto* output = getDstDataAtPortAs<T>(0);

    const ov::Shape queryShape{getSrcMemoryAtPort(0)->getStaticDims()};
    const ov::Shape keyShape{getSrcMemoryAtPort(1)->getStaticDims()};
    const ov::Shape valueShape{getSrcMemoryAtPort(2)->getStaticDims()};
    const ov::Shape blockIndicesShape{getSrcMemoryAtPort(3)->getStaticDims()};

    const auto B = static_cast<int64_t>(queryShape[0]);
    const auto H = static_cast<int64_t>(queryShape[1]);
    const auto L = static_cast<int64_t>(queryShape[2]);
    const auto E = static_cast<int64_t>(queryShape[3]);
    const auto Hk = static_cast<int64_t>(keyShape[1]);
    const auto S = static_cast<int64_t>(keyShape[2]);
    const auto Ev = static_cast<int64_t>(valueShape[3]);
    const auto Hb = static_cast<int64_t>(blockIndicesShape[1]);
    const auto numQBlocks = static_cast<int64_t>(blockIndicesShape[2]);
    const auto kBlocks = static_cast<int64_t>(blockIndicesShape[3]);

    const auto broadcastHead = [](int64_t h, int64_t dimSize) {
        return dimSize == 1 ? int64_t{0} : h;
    };

    const ov::Shape sliceQueryShape{1, 1, static_cast<size_t>(L), static_cast<size_t>(E)};
    const ov::Shape sliceKeyShape{1, 1, static_cast<size_t>(S), static_cast<size_t>(E)};
    const ov::Shape sliceValueShape{1, 1, static_cast<size_t>(S), static_cast<size_t>(Ev)};
    const ov::Shape sliceBlockIndicesShape{1, 1, static_cast<size_t>(numQBlocks), static_cast<size_t>(kBlocks)};

    // Parallelize across (batch, head): every (b, h) pair is fully independent, so each task
    // slices out exactly one head's worth of query/key/value/block_indices/output and delegates
    // to the shared reference kernel with B=H=1. This reuses the exact same math already
    // validated by the core unit tests and the Template-plugin functional tests, adding
    // multi-core scaling on top without introducing any new attention/softmax logic here.
    ov::parallel_for2d(B, H, [&](int64_t b, int64_t h) {
        const T* qSlice = query + static_cast<size_t>((b * H + h) * L * E);
        const T* kSlice = key + static_cast<size_t>((b * Hk + broadcastHead(h, Hk)) * S * E);
        const T* vSlice = value + static_cast<size_t>((b * Hk + broadcastHead(h, Hk)) * S * Ev);
        const TIndex* biSlice =
            blockIndices + static_cast<size_t>((b * Hb + broadcastHead(h, Hb)) * numQBlocks * kBlocks);
        const char* maskSlice =
            mask ? mask + static_cast<size_t>((b * Hb + broadcastHead(h, Hb)) * numQBlocks * kBlocks) : nullptr;
        T* outSlice = output + static_cast<size_t>((b * H + h) * L * Ev);

        ov::reference::block_sparse_attention<T, TIndex>(qSlice,
                                                          kSlice,
                                                          vSlice,
                                                          biSlice,
                                                          maskSlice,
                                                          scale,
                                                          outSlice,
                                                          m_causal,
                                                          m_block_size,
                                                          sliceQueryShape,
                                                          sliceKeyShape,
                                                          sliceValueShape,
                                                          sliceBlockIndicesShape);
    });
}

namespace {
struct BlockSparseAttentionContext {
    BlockSparseAttention& node;
};
}  // namespace

template <typename T>
struct BlockSparseAttention::Execute {
    using TData = typename std::tuple_element<0, T>::type;
    using TIndexType = typename std::tuple_element<1, T>::type;

    void operator()(BlockSparseAttentionContext& ctx) {
        ctx.node.executeImpl<TData, TIndexType>();
    }
};

void BlockSparseAttention::execute([[maybe_unused]] const dnnl::stream& strm) {
    auto dataPrecision = getParentEdgeAt(0)->getMemory().getDesc().getPrecision();
    auto indexPrecision = getParentEdgeAt(3)->getMemory().getDesc().getPrecision();

    BlockSparseAttentionContext ctx = {*this};

#define CASE(OV_TYPE)                                                                             \
    OV_CASE2(OV_TYPE, ov::element::i32, ov::element_type_traits<OV_TYPE>::value_type, int32_t),  \
        OV_CASE2(OV_TYPE, ov::element::i64, ov::element_type_traits<OV_TYPE>::value_type, int64_t)

    OV_SWITCH(intel_cpu,
              Execute,
              ctx,
              std::tie(dataPrecision, indexPrecision),
              CASE(ov::element::f32),
              CASE(ov::element::bf16))

#undef CASE
}

}  // namespace ov::intel_cpu::node
