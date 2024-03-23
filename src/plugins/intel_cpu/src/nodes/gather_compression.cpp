// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_compression.h"

#include "common/cpu_memcpy.h"
#include "ov_ops/gather_compressed.hpp"
#include "utils/ngraph_utils.hpp"

using namespace dnnl::impl::cpu;

#define THROW_ERROR(...) OPENVINO_THROW(getTypeStr(), " node with name '", getName(), "' ", __VA_ARGS__)

namespace ov {
namespace intel_cpu {
namespace node {

#define PRINT(X) std::cout << #X << " = " << X << std::endl

bool GatherCompression::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto gather_compression = std::dynamic_pointer_cast<const ov::op::internal::GatherCompressed>(op);
        if (!gather_compression) {
            errorMessage = "Only GatherCompression operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

GatherCompression::GatherCompression(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    if ((op->get_input_size() != 4u && op->get_input_size() != 5u) || op->get_output_size() != 1u)
        THROW_ERROR("has incorrect number of input/output[",
                    op->get_input_size(),
                    ",",
                    op->get_output_size(),
                    "] edges!");

    if (ov::is_type<ov::op::internal::GatherCompressed>(op)) {
        m_batchDims = static_cast<int>(ov::as_type_ptr<ov::op::internal::GatherCompressed>(op)->get_batch_dims());
    }
}

void GatherCompression::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    ov::element::Type dataPrecision = getOriginalInputPrecisionAtPort(GATHER_DATA);
    if (!one_of(dataPrecision, ov::element::u8, ov::element::u4, ov::element::i8, ov::element::i4)) {
        THROW_ERROR("has unsupported 'data' input precision: ", dataPrecision);
    }

    ov::element::Type scalePrecision = getOriginalInputPrecisionAtPort(GATHER_SCALE);
    if (scalePrecision != ov::element::f32) {
        THROW_ERROR("has unsupported 'scale' input precision: ", scalePrecision);
    }

    ov::element::Type outPrecision = getOriginalOutputPrecisionAtPort(0);
    if (!one_of(outPrecision, ov::element::f32, ov::element::f16)) {
        THROW_ERROR("has unsupported out precision: ", outPrecision);
    }

    if (getOriginalInputsNumber() == 5u) {
        ov::element::Type zpPrecision = getOriginalInputPrecisionAtPort(GATHER_ZP);
        if (zpPrecision != ov::element::f32) {
            THROW_ERROR("has unsupported 'zp' input precision: ", zpPrecision);
        }
        addSupportedPrimDesc({{LayoutType::ncsp, dataPrecision},
                              {LayoutType::ncsp, ov::element::i32},
                              {LayoutType::ncsp, ov::element::i32},
                              {LayoutType::ncsp, scalePrecision},
                              {LayoutType::ncsp, zpPrecision}},
                             {{LayoutType::ncsp, outPrecision}},
                             ref_any);
    } else {
        addSupportedPrimDesc({{LayoutType::ncsp, dataPrecision},
                              {LayoutType::ncsp, ov::element::i32},
                              {LayoutType::ncsp, ov::element::i32},
                              {LayoutType::ncsp, scalePrecision}},
                             {{LayoutType::ncsp, outPrecision}},
                             ref_any);
    }
}

bool GatherCompression::needPrepareParams() const {
    return false;
}

void GatherCompression::execute(dnnl::stream strm) {
    execReference();
}

void GatherCompression::executeDynamicImpl(dnnl::stream strm) {
    execReference();
}

template <typename OUT_TYPE>
void GatherCompression::execReferenceU4() {
    std::cout << "--->5:GatherCompression::execReferenceU4()" << std::endl;
}

template <typename OUT_TYPE>
void GatherCompression::execReferenceI4() {
    std::cout << "--->5:GatherCompression::execReferenceI4()" << std::endl;
    // DEBUG_LOG(getName(), "execReference4bit");
    // auto data_mem_ptr = getParentEdgeAt(GATHER_DATA)->getMemoryPtr();
    // auto ind_mem_ptr = getParentEdgeAt(GATHER_INDICES)->getMemoryPtr();
    // const auto* psrc = data_mem_ptr->getDataAs<uint8_t>();
    // const auto* pidx = ind_mem_ptr->getDataAs<int32_t>();

    // bool one_dim_zp = getParentEdgeAt(GATHER_ZP)->getMemoryPtr()->getShape().getRank() == 1;
    // const auto* zp = getSrcDataAtPortAs<float_t>(GATHER_ZP);
    // const auto* scale = getSrcDataAtPortAs<float_t>(GATHER_SCALE);
    // auto* pdst = getDstDataAtPortAs<float>(0);

    // const auto& idxDims = ind_mem_ptr->getStaticDims();
    // const auto batch = idxDims[0];
    // const auto seqLen = idxDims[1];

    // auto axisDim = data_mem_ptr->getStaticDims()[0];
    // auto groupDim = data_mem_ptr->getStaticDims().size() == 2 ? 1 : data_mem_ptr->getStaticDims()[1];
    // auto feaDim = data_mem_ptr->getStaticDims().size() == 2 ? data_mem_ptr->getStaticDims()[1] : data_mem_ptr->getStaticDims()[2];

    // parallel_for2d(batch, seqLen, [&](size_t b, size_t s) {
    //     auto dstIdx = b * seqLen + s;
    //     auto ii = pidx[dstIdx];
    //     if (ii < 0) {
    //         if (reverseIndexing)
    //             ii += axisDim;
    //         else
    //             ii = axisDim;
    //     }

    //     auto* dst = pdst + dstIdx * feaDim * groupDim;
    //     auto* src = psrc + ii * feaDim * groupDim / 2;

    //     for (size_t g = 0; g < groupDim; g++) {
    //         // auto& deq_zp = zp[ii];
    //         // auto& deq_scale = scale[ii];
    //         auto& deq_zp = one_dim_zp ? zp[0] : zp[ii * groupDim + g];
    //         auto& deq_scale = scale[ii * groupDim + g];

    //         size_t k = 0;
    //         for (; k < feaDim; k += 2) {
    //             auto x = src[0];
    //             dst[0] = ((x & 0x0F) - deq_zp) * deq_scale;
    //             dst[1] = ((x >> 4) - deq_zp) * deq_scale;
    //             dst += 2;
    //             src++;
    //         }
    //         // Process last one if feaDim is odd
    //         for (; k < feaDim; k++) {
    //             auto x = src[0];
    //             dst[0] = ((x & 0x0F) - deq_zp) * deq_scale;
    //             dst++;
    //             src++;
    //         }
    //     }
    // });
}

static std::string shape2str(ov::intel_cpu::Shape shape) {
    std::string str = "[";
    for (auto s : shape.getStaticDims()) {
        str += std::to_string(s) + ",";
    }
    return str + "]";
}

template <typename IN_TYPE, typename OUT_TYPE>
void GatherCompression::execReference8bit() {
    std::cout << "GatherCompression::execReference8bit()\n";

    const int32_t* srcIndices = getSrcDataAtPortAs<const int32_t>(GATHER_INDICES);
    const IN_TYPE* srcData = getSrcDataAtPortAs<const IN_TYPE>(GATHER_DATA);
    OUT_TYPE* dstData = getDstDataAtPortAs<OUT_TYPE>(0);

    // PRINT(getParentEdgeAt(GATHER_DATA)->getMemoryPtr()->getPrecision());
    // PRINT(getChildEdgeAt(0)->getMemoryPtr()->getPrecision());
    // PRINT(shape2str(getChildEdgeAt(0)->getMemoryPtr()->getShape()));

    auto dataMemPtr = getSrcMemoryAtPort(GATHER_DATA);
    // [4,2]
    const auto& dataDims = dataMemPtr->getStaticDims();
    // 0, or 1
    auto axis = (getSrcDataAtPortAs<const int32_t>(GATHER_AXIS))[0];
    //

    PRINT(shape2str(dataMemPtr->getShape()));
    PRINT(axis);

    // zp/scale
    bool have_zp = getOriginalInputsNumber() > 4u;
    float const_zp = 0;
    const auto* zp = have_zp ? getSrcDataAtPortAs<float_t>(GATHER_ZP) : &const_zp;
    const auto* scale = getSrcDataAtPortAs<float_t>(GATHER_SCALE);
    PRINT(getParentEdgeAt(GATHER_SCALE)->getMemoryPtr()->getPrecision());

    auto scale_group_size = getParentEdgeAt(GATHER_DATA)->getMemoryPtr()->getShape().getElementsCount() /
                            getParentEdgeAt(GATHER_SCALE)->getMemoryPtr()->getShape().getElementsCount();
    PRINT(scale_group_size);
    auto zp_group_size = 1;
    if (have_zp)
        zp_group_size = getParentEdgeAt(GATHER_DATA)->getMemoryPtr()->getShape().getElementsCount() /
                        getParentEdgeAt(GATHER_ZP)->getMemoryPtr()->getShape().getElementsCount();

    auto betweenBatchAndAxisSize =
        std::accumulate(dataDims.begin() + m_batchDims, dataDims.begin() + axis, 1lu, std::multiplies<uint64_t>());

    auto idxMemPtr = getSrcMemoryAtPort(GATHER_INDICES);
    const auto& idxDims = idxMemPtr->getStaticDims();
    auto specIndicesSize =
        std::accumulate(idxDims.begin() + m_batchDims, idxDims.end(), 1lu, std::multiplies<uint64_t>());

    auto afterAxisSize = std::accumulate(dataDims.begin() + axis + 1, dataDims.end(), 1lu, std::multiplies<uint64_t>());

    auto dataTypeSize = getOriginalInputPrecisionAtPort(GATHER_DATA).size();
    auto afterAxisSizeInBytes = afterAxisSize * dataTypeSize;
    auto specIdxAndAfterAxSizeB = specIndicesSize * afterAxisSizeInBytes;

    auto beforeBatchSize = std::accumulate(dataDims.begin(), dataDims.begin() + m_batchDims, 1lu, std::multiplies<Dim>());

    const size_t dstAfterBatchSize = betweenBatchAndAxisSize * specIdxAndAfterAxSizeB;

    auto axisDim = dataDims[axis];
    PRINT(axisDim);

    auto axisAndAfterAxisSizeInBytes = axisDim * afterAxisSizeInBytes;
    auto srcAfterBatchSizeInBytes = betweenBatchAndAxisSize * axisAndAfterAxisSizeInBytes;

    parallel_for2d(beforeBatchSize, specIndicesSize, [&](const size_t b, const size_t j) {
        int ii = srcIndices[b * specIndicesSize + j];
        if (ii < 0) {
            if (reverseIndexing)
                ii += axisDim;
            else
                ii = axisDim;
        }
        const size_t idx = ii;
        const size_t c2 = dstAfterBatchSize * b + afterAxisSizeInBytes * j;
        if (idx < static_cast<size_t>(axisDim)) {
            size_t c1 = srcAfterBatchSizeInBytes * b + afterAxisSizeInBytes * idx;
            for (size_t i = 0; i < betweenBatchAndAxisSize; i++) {
                size_t srcIdx = c1 + axisAndAfterAxisSizeInBytes * i;
                size_t dstIdx = c2 + specIdxAndAfterAxSizeB * i;

                // cpu_memcpy(&dstData[dstIdx], &srcData[srcIdx], afterAxisSizeInBytes);
                const IN_TYPE* psrc = &srcData[srcIdx];
                OUT_TYPE* pdst = &dstData[dstIdx];

                const uint scale_offset = srcIdx / scale_group_size;
                auto cur_zp = have_zp ? zp[srcIdx / zp_group_size] : 0;
                for (size_t p = 0; p < afterAxisSize; p++) {
                    pdst[p] = static_cast<OUT_TYPE>((static_cast<float>(psrc[p]) - cur_zp) * scale[scale_offset]);
                }
            }
        } else {
            for (size_t i = 0; i < betweenBatchAndAxisSize; i++) {
                size_t dstIdx = c2 + specIdxAndAfterAxSizeB * i;
                for (size_t p = 0; p < afterAxisSize; p++)
                    dstData[dstIdx] = 0;
            }
        }
    });
}

void GatherCompression::execReference() {
    auto in_precison = getParentEdgeAt(GATHER_DATA)->getMemoryPtr()->getPrecision();
    auto out_precision = getChildEdgeAt(0)->getMemoryPtr()->getPrecision();

    if (out_precision == ov::element::f16) {
        switch (in_precison) {
        case ov::element::u8:
            return execReference8bit<uint8_t, float16>();
        case ov::element::i8:
            return execReference8bit<int8_t, float16>();
        case ov::element::u4:
            return execReferenceU4<float16>();
        case ov::element::i4:
            return execReferenceI4<float16>();
        default:
            break;
        }
    } else if (out_precision == ov::element::f32) {
        switch (in_precison) {
        case ov::element::u8:
            return execReference8bit<uint8_t, float>();
        case ov::element::i8:
            return execReference8bit<int8_t, float>();
        case ov::element::u4:
            return execReferenceU4<float>();
        case ov::element::i4:
            return execReferenceI4<float>();
        default:
            break;
        }
    }

    THROW_ERROR("only support in precision(u4/i4/u8/i8), out precision(f32/f16), in_precison=",
                in_precison,
                ", out_precision=",
                out_precision);
}

bool GatherCompression::created() const {
    return getType() == Type::GatherCompression;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov