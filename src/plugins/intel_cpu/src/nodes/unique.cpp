// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "unique.hpp"

#include <openvino/op/constant.hpp>
#include <openvino/op/unique.hpp>

#include "common/cpu_memcpy.h"
#include "openvino/core/parallel.hpp"
#include "shape_inference/shape_inference_internal_dyn.hpp"

using namespace ov::intel_cpu;
using namespace ov::intel_cpu::node;

bool Unique::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!ov::is_type<op::v10::Unique>(op)) {
            errorMessage = "Not supported Unique operation version. CPU plug-in supports only 10th version.";
            return false;
        }
        if (op->get_input_size() > AXIS && !ov::is_type<op::v0::Constant>(op->get_input_node_ptr(AXIS))) {
            errorMessage = "CPU plug-in supports only constant Axis input.";
            return false;
        }
    } catch (...) {
        return false;
    }

    return true;
}

Unique::Unique(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, InternalDynShapeInferFactory()) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    if (!one_of(op->get_input_size(), 1u, 2u) || op->get_output_size() != 4) {
        THROW_CPU_NODE_ERR("has incorrect number of input/output edges.");
    }

    for (int i = 0; i < 4; i++) {
        definedOutputs[i] = !op->get_output_target_inputs(i).empty();
    }

    sorted = ov::as_type_ptr<op::v10::Unique>(op)->get_sorted();
    if (op->get_input_size() > AXIS) {
        flattened = false;
        axis = ov::as_type<op::v0::Constant>(op->get_input_node_ptr(AXIS))->cast_vector<int>()[0];
        if (axis < 0) {
            axis += op->get_input_partial_shape(IN_DATA).rank().get_length();
        }
        if (axis < 0 || axis >= op->get_input_partial_shape(IN_DATA).rank().get_length()) {
            THROW_CPU_NODE_ERR("has invalid axis value: ",
                               ov::as_type<op::v0::Constant>(op->get_input_node_ptr(AXIS))->cast_vector<int>()[0]);
        }
    } else {
        flattened = true;
    }
}

void Unique::initSupportedPrimitiveDescriptors() {
    dataPrecision = getOriginalInputPrecisionAtPort(IN_DATA);
    if (dataPrecision != ov::element::i32 && dataPrecision != ov::element::i8 && dataPrecision != ov::element::u8) {
        dataPrecision = ov::element::f32;
    }
    dataTypeSize = dataPrecision.size();
    const ov::element::Type axisPrecision = ov::element::i32;

    impl_desc_type implType = ref;

    std::vector<PortConfigurator> inPortConfigs = {{LayoutType::ncsp, dataPrecision}};
    if (!flattened) {
        inPortConfigs.emplace_back(LayoutType::ncsp, axisPrecision);
    }
    std::vector<PortConfigurator> outPortConfigs;
    outPortConfigs.reserve(4);
    for (int i = 0; i < 4; i++) {
        outPortConfigs.emplace_back(LayoutType::ncsp, i == 0 ? dataPrecision : axisPrecision);
    }

    addSupportedPrimDesc(inPortConfigs, outPortConfigs, implType);
}

void Unique::createPrimitive() {
    Node::createPrimitive();
}

void Unique::prepareParams() {
    auto dataMemPtr = getSrcMemoryAtPort(IN_DATA);
    if (!dataMemPtr) {
        THROW_CPU_NODE_ERR("has null input data memory.");
    }
    for (int i = 0; i < 4; i++) {
        if (definedOutputs[i]) {
            auto dstMemPtr = getDstMemoryAtPort(i);
            if (!dstMemPtr) {
                THROW_CPU_NODE_ERR("has null output memory at port ", i);
            }
        }
    }
    if (getSelectedPrimitiveDescriptor() == nullptr) {
        THROW_CPU_NODE_ERR("has unidentified preferable primitive descriptor.");
    }

    size_t srcLen = 1;
    if (flattened) {
        srcLen = getSrcMemoryAtPort(IN_DATA)->getSize() / dataTypeSize;
    } else {
        auto dstDataShape = getSrcMemoryAtPort(IN_DATA)->getStaticDims();
        srcLen = dstDataShape[axis];
    }
    firstUniTmp.resize(srcLen, 0);
    inToOutTmp.resize(srcLen);
    occurTmp.resize(srcLen);
}

template <typename T>
struct Unique::flattenExec {
    void operator()(Unique* node) {
        node->flattenTensorExec<T>();
    }
};

template <typename T>
struct Unique::slicedExec {
    void operator()(Unique* node) {
        node->slicedTensorExec<T>();
    }
};

void Unique::execute(const dnnl::stream& strm) {
    if (flattened) {
        OV_SWITCH(intel_cpu,
                  flattenExec,
                  this,
                  dataPrecision,
                  OV_CASE(ov::element::f32, float),
                  OV_CASE(ov::element::i32, int32_t),
                  OV_CASE(ov::element::i8, int8_t),
                  OV_CASE(ov::element::u8, uint8_t))
        return;
    }
    OV_SWITCH(intel_cpu,
              slicedExec,
              this,
              dataPrecision,
              OV_CASE(ov::element::f32, float),
              OV_CASE(ov::element::i32, int32_t),
              OV_CASE(ov::element::i8, int8_t),
              OV_CASE(ov::element::u8, uint8_t))
}

void Unique::executeDynamicImpl(const dnnl::stream& strm) {
    const auto& srcDataDims = getSrcMemoryAtPort(IN_DATA)->getStaticDims();
    VectorDims dstDataDims;
    Dim uniqLen = 1;
    if (flattened) {
        uniqLen = std::accumulate(srcDataDims.begin(), srcDataDims.end(), 1, std::multiplies<>());
        dstDataDims = {uniqLen};
    } else {
        uniqLen = srcDataDims[axis];
        dstDataDims = srcDataDims;
    }
    redefineOutputMemory({dstDataDims, {uniqLen}, {uniqLen}, {uniqLen}});

    execute(strm);
}

template <typename T>
void Unique::flattenTensorExec() {
    const T* srcDataPtr = getSrcDataAtPortAs<const T>(IN_DATA);
    const size_t inputLen = getSrcMemoryAtPort(IN_DATA)->getSize() / sizeof(T);
    std::vector<T> uniDataTmp(inputLen);
    auto uniDataTmpPtr = uniDataTmp.data();
    int *firstTmpPtr = nullptr, *inToOutTmpPtr = nullptr, *occurTmpPtr = nullptr;
    if (definedOutputs[FIRST_UNIQUE_IDX]) {
        firstTmpPtr = firstUniTmp.data();
    }
    if (definedOutputs[INPUT_TO_UNIQ_IDX]) {
        inToOutTmpPtr = inToOutTmp.data();
    }
    if (definedOutputs[OCCURRENCES_NUM]) {
        occurTmpPtr = occurTmp.data();
    }
    uniqueLen = inputLen;

    if (sorted) {
        cpu_parallel_memcpy(uniDataTmpPtr, srcDataPtr, inputLen * sizeof(T));
        std::sort(uniDataTmpPtr, uniDataTmpPtr + inputLen);
        auto last = std::unique(uniDataTmpPtr, uniDataTmpPtr + inputLen);
        uniqueLen = last - uniDataTmpPtr;

        if (definedOutputs[FIRST_UNIQUE_IDX]) {
            T* first = uniDataTmpPtr;
            for (T* it = first; it < last; it++) {
                for (size_t i = 0; i < inputLen; i++) {
                    if (srcDataPtr[i] == *it) {
                        *firstTmpPtr++ = i;
                        first++;
                        break;
                    }
                }
            }
        }
        if (definedOutputs[INPUT_TO_UNIQ_IDX]) {
            for (size_t i = 0; i < inputLen; i++) {
                if (i > 0 && srcDataPtr[i] == srcDataPtr[i - 1]) {
                    inToOutTmpPtr[i] = inToOutTmpPtr[i - 1];
                    continue;
                }
                for (size_t j = 0; j < uniqueLen; j++) {
                    if (srcDataPtr[i] == uniDataTmpPtr[j]) {
                        inToOutTmpPtr[i] = j;
                        break;
                    }
                }
            }
        }
        if (definedOutputs[OCCURRENCES_NUM]) {
            std::fill(occurTmpPtr, occurTmpPtr + uniqueLen, 0);
            for (size_t j = 0; j < uniqueLen; j++) {
                for (size_t i = 0; i < inputLen; i++) {
                    if (srcDataPtr[i] == uniDataTmpPtr[j]) {
                        occurTmpPtr[j]++;
                    }
                }
            }
        }
    } else {
        std::unordered_map<T, int32_t> uniq;
        uniq.reserve(inputLen);

        if (definedOutputs[OCCURRENCES_NUM]) {
            std::fill(occurTmpPtr, occurTmpPtr + inputLen, 1);
        }

        for (size_t i = 0, j = 0; i < inputLen; ++i) {
            auto it = uniq.emplace(srcDataPtr[i], j);
            if (definedOutputs[INPUT_TO_UNIQ_IDX]) {
                inToOutTmpPtr[i] = it.first->second;
                if (it.second) {
                    if (definedOutputs[FIRST_UNIQUE_IDX]) {
                        firstTmpPtr[j] = i;
                    }
                    ++j;
                } else {
                    if (definedOutputs[OCCURRENCES_NUM]) {
                        occurTmpPtr[inToOutTmpPtr[i]]++;
                    }
                }
            }
        }

        uniqueLen = static_cast<int64_t>(uniq.size());
        for (const auto& it : uniq) {
            uniDataTmpPtr[it.second] = it.first;
        }
    }

    redefineOutputMemory({{uniqueLen}, {uniqueLen}, {inputLen}, {uniqueLen}});

    T* uniDataPtr = getDstDataAtPortAs<T>(UNIQUE_DATA);
    cpu_parallel_memcpy(uniDataPtr, uniDataTmpPtr, uniqueLen * sizeof(T));
    if (definedOutputs[FIRST_UNIQUE_IDX]) {
        auto* firstPtr = getDstDataAtPortAs<int>(FIRST_UNIQUE_IDX);
        cpu_parallel_memcpy(firstPtr, firstUniTmp.data(), uniqueLen * sizeof(int));
    }
    if (definedOutputs[INPUT_TO_UNIQ_IDX]) {
        auto inToOutPtr = getDstDataAtPortAs<int>(INPUT_TO_UNIQ_IDX);
        cpu_parallel_memcpy(inToOutPtr, inToOutTmp.data(), inputLen * sizeof(int));
    }
    if (definedOutputs[OCCURRENCES_NUM]) {
        auto occurPtr = getDstDataAtPortAs<int>(OCCURRENCES_NUM);
        cpu_parallel_memcpy(occurPtr, occurTmp.data(), uniqueLen * sizeof(int));
    }
}

template <typename T>
void Unique::slicedTensorExec() {
    auto inDataMemPtr = getSrcMemoryAtPort(IN_DATA);
    auto srcDataPtr = inDataMemPtr->getDataAs<const T>();
    int *firstTmpPtr = nullptr, *inToOutTmpPtr = nullptr, *occurTmpPtr = nullptr;
    if (definedOutputs[FIRST_UNIQUE_IDX]) {
        firstTmpPtr = firstUniTmp.data();
    }
    if (definedOutputs[INPUT_TO_UNIQ_IDX]) {
        inToOutTmpPtr = inToOutTmp.data();
    }
    if (definedOutputs[OCCURRENCES_NUM]) {
        occurTmpPtr = occurTmp.data();
    }

    const auto& srcDataShape = inDataMemPtr->getStaticDims();

    const auto axisDim = srcDataShape[axis];
    int64_t outerLen = 1lu;
    if (axis > 0) {
        outerLen = std::accumulate(srcDataShape.begin(), srcDataShape.begin() + axis, 1, std::multiplies<>());
    }
    int64_t innerLen = 1;
    if (static_cast<size_t>(axis) < srcDataShape.size() - 1) {
        innerLen = std::accumulate(srcDataShape.begin() + axis + 1, srcDataShape.end(), 1, std::multiplies<>());
    }
    const auto innerSizeB = innerLen * sizeof(T);
    const auto srcOuterStep = innerLen * axisDim;

    if (definedOutputs[FIRST_UNIQUE_IDX]) {
        firstTmpPtr[0] = 0;
    }
    if (definedOutputs[INPUT_TO_UNIQ_IDX]) {
        inToOutTmpPtr[0] = 0;
    }
    if (definedOutputs[OCCURRENCES_NUM]) {
        occurTmpPtr[0] = 1;
        std::fill(occurTmpPtr, occurTmpPtr + axisDim, 1);
    }

    uniqueLen = 1lu;
    std::vector<size_t> uniqIdx(axisDim, 0lu);
    // Search for unique slices.
    for (size_t a = 1lu; a < axisDim; a++) {
        auto first1 = srcDataPtr + a * innerLen;
        auto last1 = srcDataPtr + (a + 1lu) * innerLen;
        bool equal = true;
        size_t uIdx = 0lu;
        // Compare with unique blocks.
        for (; uIdx < uniqueLen; uIdx++) {
            auto first2 = srcDataPtr + uniqIdx[uIdx] * innerLen;
            equal = true;
            for (int64_t o = 0lu; o < outerLen; o++) {
                equal = std::equal(first1, last1, first2);
                if (!equal) {
                    break;
                }
                first1 += srcOuterStep;
                last1 += srcOuterStep;
                first2 += srcOuterStep;
            }
            if (equal) {
                break;
            }
        }
        if (!equal) {
            if (definedOutputs[FIRST_UNIQUE_IDX]) {
                firstTmpPtr[uniqueLen] = a;
            }

            uniqIdx[uniqueLen++] = a;
        } else {
            if (definedOutputs[OCCURRENCES_NUM]) {
                occurTmpPtr[uIdx]++;
            }
        }
        if (definedOutputs[INPUT_TO_UNIQ_IDX]) {
            inToOutTmpPtr[a] = uIdx;
        }
    }

    // Redefinition of output shapes.
    auto dstDataShape = srcDataShape;
    dstDataShape[axis] = uniqueLen;
    redefineOutputMemory({dstDataShape, {uniqueLen}, {axisDim}, {uniqueLen}});

    int *firstPtr = nullptr, *inToOutPtr = nullptr, *occurNPtr = nullptr;
    if (definedOutputs[FIRST_UNIQUE_IDX]) {
        firstPtr = getDstDataAtPortAs<int>(FIRST_UNIQUE_IDX);
    }
    if (definedOutputs[INPUT_TO_UNIQ_IDX]) {
        inToOutPtr = getDstDataAtPortAs<int>(INPUT_TO_UNIQ_IDX);
    }
    if (definedOutputs[OCCURRENCES_NUM]) {
        occurNPtr = getDstDataAtPortAs<int>(OCCURRENCES_NUM);
    }

    T* dstDataPtr = getDstDataAtPortAs<T>(UNIQUE_DATA);
    const auto dstOuterStep = innerLen * uniqueLen;
    // Filling of the first output if needed.
    if (sorted || definedOutputs[UNIQUE_DATA]) {
        parallel_for(uniqueLen, [&](size_t u) {
            auto first1 = srcDataPtr + uniqIdx[u] * innerLen;
            auto first2 = dstDataPtr + u * innerLen;
            for (int64_t p = 0lu; p < outerLen; p++) {
                cpu_memcpy(first2, first1, innerSizeB);
                first1 += srcOuterStep;
                first2 += dstOuterStep;
            }
        });
    }

    const auto uniqueLenIB = uniqueLen * sizeof(T);

    if (sorted) {
        const auto dstUniDataLen = dstOuterStep * outerLen;
        std::vector<T> vDstBuff(dstUniDataLen);
        auto dstBuff = vDstBuff.data();

        struct OrdEl {
            T val;
            int64_t idx;
        };

        std::vector<OrdEl> colToSort(uniqueLen);
        T *dst1 = dstDataPtr, *dst2 = dstBuff;
        int *first1 = firstPtr, *first2 = firstTmpPtr;
        int *occurN1 = occurNPtr, *occurN2 = occurTmpPtr;
        int *inToOut1 = inToOutPtr, *inToOut2 = inToOutTmpPtr;

        const bool defined3outputs =
            definedOutputs[FIRST_UNIQUE_IDX] || definedOutputs[OCCURRENCES_NUM] || definedOutputs[INPUT_TO_UNIQ_IDX];

        for (int64_t o = outerLen - 1; o >= 0; o--) {  // Backward loop through the outer block.
            const int64_t pos1Lim = o * dstOuterStep;
            int64_t pos1 = pos1Lim + innerLen - 1;
            for (; pos1 >= pos1Lim; pos1--) {  // Backward loop through the inner block.
                int64_t pos2 = pos1;
                for (int64_t k = 0; k < static_cast<int64_t>(uniqueLen); k++, pos2 += innerLen) {
                    colToSort[k] = {dst1[pos2], k};
                }
                std::stable_sort(colToSort.begin(), colToSort.end(), [](const OrdEl& el1, const OrdEl& el2) {
                    return el1.val < el2.val;
                });

                // Permutation
                parallel_for2d(outerLen, uniqueLen, [&](int64_t ot, size_t u) {
                    auto src = dst1 + ot * dstOuterStep + colToSort[u].idx * innerLen;
                    auto dst = dst2 + ot * dstOuterStep + u * innerLen;

                    cpu_memcpy(dst, src, innerSizeB);
                });

                if (defined3outputs) {
                    parallel_for(uniqueLen, [&](size_t u) {
                        if (definedOutputs[FIRST_UNIQUE_IDX]) {
                            first1[u] = first2[colToSort[u].idx];
                        }
                        if (definedOutputs[OCCURRENCES_NUM]) {
                            occurN1[u] = occurN2[colToSort[u].idx];
                        }
                        if (definedOutputs[INPUT_TO_UNIQ_IDX]) {
                            for (size_t ax = 0; ax < axisDim; ax++) {
                                if (inToOut2[ax] == colToSort[u].idx) {
                                    inToOut1[ax] = u;
                                }
                            }
                        }
                    });
                }

                std::swap(dst1, dst2);
                if (definedOutputs[FIRST_UNIQUE_IDX]) {
                    std::swap(first1, first2);
                }
                if (definedOutputs[OCCURRENCES_NUM]) {
                    std::swap(occurN1, occurN2);
                }
                if (definedOutputs[INPUT_TO_UNIQ_IDX]) {
                    std::swap(inToOut1, inToOut2);
                }
            }
        }

        if (definedOutputs[UNIQUE_DATA] && dst1 != dstDataPtr) {
            cpu_parallel_memcpy(dstDataPtr, dst1, dstUniDataLen * sizeof(T));
        }
        if (definedOutputs[FIRST_UNIQUE_IDX] && first2 != firstPtr) {
            cpu_parallel_memcpy(firstPtr, first2, uniqueLenIB);
        }
        if (definedOutputs[INPUT_TO_UNIQ_IDX] && inToOut2 != inToOutPtr) {
            cpu_parallel_memcpy(inToOutPtr, inToOut2, axisDim * sizeof(int));
        }
        if (definedOutputs[OCCURRENCES_NUM] && occurN2 != occurNPtr) {
            cpu_parallel_memcpy(occurNPtr, occurN2, uniqueLenIB);
        }
    } else {
        if (definedOutputs[FIRST_UNIQUE_IDX]) {
            cpu_parallel_memcpy(firstPtr, firstUniTmp.data(), uniqueLenIB);
        }
        if (definedOutputs[INPUT_TO_UNIQ_IDX]) {
            cpu_parallel_memcpy(inToOutPtr, inToOutTmp.data(), axisDim * sizeof(int));
        }
        if (definedOutputs[OCCURRENCES_NUM]) {
            cpu_parallel_memcpy(occurNPtr, occurTmp.data(), uniqueLenIB);
        }
    }
}
