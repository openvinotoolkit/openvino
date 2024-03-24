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
static std::string shape2str(ov::intel_cpu::Shape shape) {
    std::string str = "[";
    for (auto s : shape.getStaticDims()) {
        str += std::to_string(s) + ",";
    }
    return str + "]";
}
static std::string dims2str(ov::intel_cpu::VectorDims dims) {
    std::string str = "[";
    for (auto s : dims) {
        str += std::to_string(s) + ",";
    }
    return str + "]";
}

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
        batchDims = static_cast<int>(ov::as_type_ptr<ov::op::internal::GatherCompressed>(op)->get_batch_dims());
    }

    const auto& dataShape = getInputShapeAtPort(GATHER_DATA);
    isDataShapeStat = dataShape.isStatic();
    dataSrcRank = dataShape.getRank();

    const auto& idxShape = getInputShapeAtPort(GATHER_INDICES);
    isIdxShapeStat = idxShape.isStatic();
    const auto indicesRank = idxShape.getRank();
    if (dataSrcRank == 0lu || indicesRank == 0lu)
        THROW_ERROR("has incorrect input parameters ranks.");

    if (batchDims < 0)
        batchDims += indicesRank;
    if (batchDims < 0 || batchDims > std::min(static_cast<int>(dataSrcRank), static_cast<int>(indicesRank)))
        THROW_ERROR("has incorrect batch_dims ", batchDims, "!");

    if (ov::is_type<ov::op::v0::Constant>(op->get_input_node_ptr(GATHER_AXIS))) {
        isAxisInputConst = true;
        axis = ov::as_type<ov::op::v0::Constant>(op->get_input_node_ptr(GATHER_AXIS))->cast_vector<int>()[0];
        if (axis < 0)
            axis += dataSrcRank;
        if (axis < 0 || axis >= dataSrcRank || batchDims > axis)
            THROW_ERROR("has incorrect input parameter axis value: ", axis);
    }

    if (auto indices = ov::as_type<ov::op::v0::Constant>(op->get_input_node_ptr(GATHER_INDICES))) {
        constIndices = indices->cast_vector<int>();
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

    scale_group_size =
        getInputShapeAtPort(GATHER_DATA).getElementsCount() / getInputShapeAtPort(GATHER_SCALE).getElementsCount();
    PRINT(scale_group_size);

    dataTypeSize = getOriginalInputPrecisionAtPort(GATHER_DATA).size();
    PRINT(dataTypeSize);

    const auto& dataDims = getInputShapeAtPort(GATHER_DATA).getDims();
    if (isAxisInputConst && isDataShapeStat) {
        axisDim = dataDims[axis];
        beforeAxisSize = std::accumulate(dataDims.begin(), dataDims.begin() + axis, 1lu, std::multiplies<Dim>());
        betweenBatchAndAxisSize =
            std::accumulate(dataDims.begin() + batchDims, dataDims.begin() + axis, 1lu, std::multiplies<Dim>());
        afterAxisSize = std::accumulate(dataDims.begin() + axis + 1, dataDims.end(), 1lu, std::multiplies<Dim>());

        afterAxisSizeInBytes = afterAxisSize * dataTypeSize;
        axisAndAfterAxisSizeInBytes = axisDim * afterAxisSizeInBytes;
        srcAfterBatchSizeInBytes = betweenBatchAndAxisSize * axisAndAfterAxisSizeInBytes;
    }
    if (isDataShapeStat) {
        beforeBatchSize = std::accumulate(dataDims.begin(), dataDims.begin() + batchDims, 1lu, std::multiplies<Dim>());
    }
    if (isIdxShapeStat) {
        const auto& idxDims = getInputShapeAtPort(GATHER_INDICES).getDims();
        specIndicesSize = std::accumulate(idxDims.begin() + batchDims, idxDims.end(), 1lu, std::multiplies<Dim>());

        if (isDataShapeStat) {
            specIdxAndAfterAxSizeB = specIndicesSize * afterAxisSizeInBytes;
            totalWork = beforeBatchSize * betweenBatchAndAxisSize * specIndicesSize * afterAxisSize;
        }
    }

    if (getOriginalInputsNumber() == 5u) {
        ov::element::Type zpPrecision = getOriginalInputPrecisionAtPort(GATHER_ZP);
        if (zpPrecision != ov::element::f32) {
            THROW_ERROR("has unsupported 'zp' input precision: ", zpPrecision);
        }

        have_zp = true;
        zp_group_size =
            getInputShapeAtPort(GATHER_DATA).getElementsCount() / getInputShapeAtPort(GATHER_ZP).getElementsCount();
        PRINT(zp_group_size);
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
    if (isInPlace()) {
        return false;
    }
    bool result = inputShapesModified();
    if (!isAxisInputConst)
        result = result || axis != (getSrcDataAtPortAs<const int32_t>(GATHER_AXIS))[0];
    return result;
}

void GatherCompression::prepareParams() {
    auto dataMemPtr = getSrcMemoryAtPort(GATHER_DATA);
    if (!dataMemPtr || !dataMemPtr->isAllocated())
        THROW_ERROR(" has not allocated input data memory.");
    auto idxMemPtr = getSrcMemoryAtPort(GATHER_INDICES);
    if (!idxMemPtr || !idxMemPtr->isAllocated())
        THROW_ERROR(" has not allocated input indices memory.");
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_ERROR(" has unidentified preferable primitive descriptor.");

    if (!isAxisInputConst) {
        axis = (getSrcDataAtPortAs<const int32_t>(GATHER_AXIS))[0];
        if (axis < 0)
            axis += dataSrcRank;
        if (axis < 0 || axis >= dataSrcRank || batchDims > axis)
            THROW_ERROR("has incorrect input parameter axis value: ", axis);
    }

    if (!isDataShapeStat || !isAxisInputConst) {
        const auto& dataDims = dataMemPtr->getStaticDims();
        axisDim = dataDims[axis];
        beforeBatchSize = std::accumulate(dataDims.begin(), dataDims.begin() + batchDims, 1lu, std::multiplies<uint64_t>());
        betweenBatchAndAxisSize = std::accumulate(dataDims.begin() + batchDims, dataDims.begin() + axis, 1lu, std::multiplies<uint64_t>());
        afterAxisSize = std::accumulate(dataDims.begin() + axis + 1, dataDims.end(), 1lu, std::multiplies<uint64_t>());

        afterAxisSizeInBytes = afterAxisSize * dataTypeSize;
        axisAndAfterAxisSizeInBytes = axisDim * afterAxisSizeInBytes;
        srcAfterBatchSizeInBytes = betweenBatchAndAxisSize * axisAndAfterAxisSizeInBytes;

        if (isIdxShapeStat) {
            specIdxAndAfterAxSizeB = specIndicesSize * afterAxisSizeInBytes;
            totalWork = beforeBatchSize * betweenBatchAndAxisSize * specIndicesSize * afterAxisSize;
        }
    }

    if (!isIdxShapeStat) {
        const auto& idxDims = idxMemPtr->getStaticDims();
        specIndicesSize = std::accumulate(idxDims.begin() + batchDims, idxDims.end(), 1lu, std::multiplies<uint64_t>());

        specIdxAndAfterAxSizeB = specIndicesSize * afterAxisSizeInBytes;
        totalWork = beforeBatchSize * betweenBatchAndAxisSize * specIndicesSize * afterAxisSize;
    }
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
    const int32_t* srcIndices = getSrcDataAtPortAs<const int32_t>(GATHER_INDICES);
    const uint8_t* srcData = getSrcDataAtPortAs<const uint8_t>(GATHER_DATA);
    OUT_TYPE* dstData = getDstDataAtPortAs<OUT_TYPE>(0);

    // zp/scale
    float const_zp = 0;
    const auto* zp = have_zp ? getSrcDataAtPortAs<float_t>(GATHER_ZP) : &const_zp;
    const auto* scale = getSrcDataAtPortAs<float_t>(GATHER_SCALE);
    PRINT(getParentEdgeAt(GATHER_SCALE)->getMemoryPtr()->getPrecision());

    const size_t dstAfterBatchSize = betweenBatchAndAxisSize * specIdxAndAfterAxSizeB;

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
                PRINT(srcIdx);
                const uint8_t* psrc = &srcData[srcIdx];
                OUT_TYPE* pdst = &dstData[dstIdx];

                const uint scale_offset = srcIdx / scale_group_size;
                auto cur_zp = have_zp ? zp[srcIdx / zp_group_size] : 0;
                size_t p = srcIdx;
                size_t dst_idx = 0;

                for (; p < srcIdx + afterAxisSize; p++) {
                    if (p % 2 == 0) {
                        auto val = srcData[p >> 1];
                        pdst[dst_idx] =
                            static_cast<OUT_TYPE>((static_cast<float>(val & 0xF) - cur_zp) * scale[scale_offset]);
                    } else {
                        auto val = srcData[p >> 1];
                        pdst[dst_idx] = static_cast<OUT_TYPE>((static_cast<float>((val >> 4) & 0xF) - cur_zp) *
                                                              scale[scale_offset]);
                    }
                    dst_idx++;
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


template <typename OUT_TYPE>
void GatherCompression::execReferenceI4() {
    std::cout << "--->5:GatherCompression::execReferenceI4()" << std::endl;
    const int32_t* srcIndices = getSrcDataAtPortAs<const int32_t>(GATHER_INDICES);
    const uint8_t* srcData = getSrcDataAtPortAs<const uint8_t>(GATHER_DATA);
    OUT_TYPE* dstData = getDstDataAtPortAs<OUT_TYPE>(0);

    // zp/scale
    float const_zp = 0;
    const auto* zp = have_zp ? getSrcDataAtPortAs<float_t>(GATHER_ZP) : &const_zp;
    const auto* scale = getSrcDataAtPortAs<float_t>(GATHER_SCALE);
    PRINT(getParentEdgeAt(GATHER_SCALE)->getMemoryPtr()->getPrecision());

    const size_t dstAfterBatchSize = betweenBatchAndAxisSize * specIdxAndAfterAxSizeB;

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
                PRINT(srcIdx);
                const uint8_t* psrc = &srcData[srcIdx];
                OUT_TYPE* pdst = &dstData[dstIdx];

                const uint scale_offset = srcIdx / scale_group_size;
                auto cur_zp = have_zp ? zp[srcIdx / zp_group_size] : 0;
                size_t p = srcIdx;
                size_t dst_idx = 0;

                auto cvt_i4_low = [](const uint8_t& val) {
                    if (val & 0x8) {
                        // Just fill in the high 4 bits with 1
                        return static_cast<int8_t>((val & 0x7) | 0xf8);
                    } else {
                        return static_cast<int8_t>(val & 0xF);
                    }
                };
                auto cvt_i4_high = [](const uint8_t& val) {
                    if (val & 0x80) {
                        return static_cast<int8_t>(((val >> 4) & 0x7) | 0xf8);
                    } else {
                        return static_cast<int8_t>((val & 0xF) >> 4);
                    }
                };

                for (; p < srcIdx + afterAxisSize; p++) {
                    if (p % 2 == 0) {
                        auto val = srcData[p >> 1];
                        pdst[dst_idx] =
                            static_cast<OUT_TYPE>((static_cast<float>(cvt_i4_low(val)) - cur_zp) * scale[scale_offset]);
                    } else {
                        auto val = srcData[p >> 1];
                        pdst[dst_idx] = static_cast<OUT_TYPE>((static_cast<float>(cvt_i4_high(val)) - cur_zp) *
                                                              scale[scale_offset]);
                    }
                    dst_idx++;
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

template <typename IN_TYPE, typename OUT_TYPE>
void GatherCompression::execReference8bit() {
    std::cout << "GatherCompression::execReference8bit()\n";

    const int32_t* srcIndices = getSrcDataAtPortAs<const int32_t>(GATHER_INDICES);
    const IN_TYPE* srcData = getSrcDataAtPortAs<const IN_TYPE>(GATHER_DATA);
    OUT_TYPE* dstData = getDstDataAtPortAs<OUT_TYPE>(0);

    // zp/scale
    float const_zp = 0;
    const auto* zp = have_zp ? getSrcDataAtPortAs<float_t>(GATHER_ZP) : &const_zp;
    const auto* scale = getSrcDataAtPortAs<float_t>(GATHER_SCALE);
    PRINT(getParentEdgeAt(GATHER_SCALE)->getMemoryPtr()->getPrecision());

    const size_t dstAfterBatchSize = betweenBatchAndAxisSize * specIdxAndAfterAxSizeB;

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