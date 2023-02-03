// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "unique.hpp"

#include "ie_parallel.hpp"
#include "common/cpu_memcpy.h"
#include <utils/shape_inference/shape_inference_internal_dyn.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/unique.hpp>

using namespace InferenceEngine;
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

Unique::Unique(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context) :
        Node(op, context, InternalDynShapeInferFactory()) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    if (!one_of(op->get_input_size(), 1u, 2u) || op->get_output_size() != 4)
        THROW_CPU_NODE_ERR << "has incorrect number of input/output edges.";

    for (int i = 0; i < 4; i++) {
        definedOutputs[i] = !op->get_output_target_inputs(i).empty();
    }

    sorted = ov::as_type_ptr<op::v10::Unique>(op)->get_sorted();
    auto dataShapeRank = op->get_input_partial_shape(IN_DATA).rank().get_length();
    if (op->get_input_size() > AXIS && dataShapeRank > 1) {
        flattened = false;
        axis = ov::as_type<op::v0::Constant>(op->get_input_node_ptr(AXIS))->cast_vector<int>()[0];
        if (axis < 0) {
            axis += dataShapeRank;
        }
        if (axis < 0 || axis >= dataShapeRank) {
            THROW_CPU_NODE_ERR << "has invalid axis value: " << ov::as_type<op::v0::Constant>(op->get_input_node_ptr(AXIS))->cast_vector<int>()[0];
        }
    } else {
        flattened = true;
    }
}

void Unique::initSupportedPrimitiveDescriptors() {
    dataPrecision = getOriginalInputPrecisionAtPort(IN_DATA);
    if (dataPrecision != Precision::I64 && dataPrecision != Precision::I32 && dataPrecision != Precision::I8 && dataPrecision != Precision::U8) {
        dataPrecision = Precision::FP32;
    }
    dataTypeSize = dataPrecision.size();
    Precision axisPrecision = Precision::I64;

    impl_desc_type implType = ref;

    std::vector<PortConfigurator> inPortConfigs = { {LayoutType::ncsp, dataPrecision} };
    if (getOriginalInputsNumber() > AXIS) {
        axisPrecision = getOriginalInputPrecisionAtPort(AXIS);
        inPortConfigs.push_back({LayoutType::ncsp, axisPrecision});
    }
    std::vector<PortConfigurator> outPortConfigs;
    for (int i = 0; i < 4; i++) {
        outputsPrc[i] = getOriginalOutputPrecisionAtPort(i);
        outPortConfigs.push_back({LayoutType::ncsp, i == 0 ? dataPrecision : getOriginalOutputPrecisionAtPort(i)});
    }

    addSupportedPrimDesc(inPortConfigs, outPortConfigs, implType);
}

void Unique::createPrimitive() {
    Node::createPrimitive();
}

void Unique::prepareParams() {
    auto dataMemPtr = getParentEdgeAt(IN_DATA)->getMemoryPtr();
    if (!dataMemPtr || !dataMemPtr->isAllocated()) {
        THROW_CPU_NODE_ERR << " has not allocated input data memory.";
    }
    for (int i = 0; i < 4; i++) {
        if (definedOutputs[i]) {
            auto dstMemPtr = getChildEdgeAt(i)->getMemoryPtr();
            if (!dstMemPtr || !dstMemPtr->isAllocated()) {
                THROW_CPU_NODE_ERR << " has not allocated output memory at port " << i;
            }
        }
    }
    if (getSelectedPrimitiveDescriptor() == nullptr) {
        THROW_CPU_NODE_ERR << " has unidentified preferable primitive descriptor.";
    }

    size_t srcLen = 1;
    if (flattened) {
        srcLen = getParentEdgeAt(IN_DATA)->getMemoryPtr()->getSize() / dataTypeSize;
    } else {
        auto dstDataShape = getParentEdgeAt(IN_DATA)->getMemoryPtr()->getStaticDims();
        srcLen = dstDataShape[axis];
    }
    if (definedOutputs[FIRST_UNIQUE_IDX]) {
        firstUniTmp.resize(srcLen, 0);
    }
    if (definedOutputs[INPUT_TO_UNIQ_IDX]) {
        inToOutTmp.resize(srcLen);
    }
    if (definedOutputs[OCCURRENCES_NUM]) {
        occurTmp.resize(srcLen);
    }
}

template<typename T>
struct Unique::flattenExec {
    void operator()(Unique *node) {
        node->flattenTensorExec<T>();
    }
};

template<typename T>
struct Unique::slicedExec {
    void operator()(Unique *node) {
        node->slicedTensorExec<T>();
    }
};

void Unique::execute(dnnl::stream strm) {
    if (flattened) {
        OV_SWITCH(intel_cpu, flattenExec, this, dataPrecision,
              OV_CASE(Precision::FP32, float),
              OV_CASE(Precision::I32, int32_t),
              OV_CASE(Precision::I64, int64_t),
              OV_CASE(Precision::I8, int8_t),
              OV_CASE(Precision::U8, uint8_t))
    } else {
        OV_SWITCH(intel_cpu, slicedExec, this, dataPrecision,
              OV_CASE(Precision::FP32, float),
              OV_CASE(Precision::I32, int32_t),
              OV_CASE(Precision::I64, int64_t),
              OV_CASE(Precision::I8, int8_t),
              OV_CASE(Precision::U8, uint8_t))
    }
}

void Unique::executeDynamicImpl(dnnl::stream strm) {
    const auto& srcDataDims = getParentEdgeAt(IN_DATA)->getMemoryPtr()->getStaticDims();
    VectorDims dstDataDims;
    Dim uniqLen = 1;
    if (flattened) {
        uniqLen = std::accumulate(srcDataDims.begin(), srcDataDims.end(), 1, std::multiplies<Dim>());
        dstDataDims = { uniqLen };
    } else {
        uniqLen = srcDataDims[axis];
        dstDataDims = srcDataDims;
    }
    redefineOutputMemory({ dstDataDims, {uniqLen}, {uniqLen}, {uniqLen}});

    execute(strm);
}

template <typename T>
void Unique::flattenTensorExec() {
    const T* srcDataPtr = reinterpret_cast<const T*>(getParentEdgeAt(IN_DATA)->getMemoryPtr()->getData());
    const size_t inputLen = getParentEdgeAt(IN_DATA)->getMemoryPtr()->getSize() / sizeof(T);
    std::vector<T> uniDataTmp(inputLen);
    auto uniDataTmpPtr = uniDataTmp.data();
    int64_t *firstTmpPtr = nullptr, *inToOutTmpPtr = nullptr, *occurTmpPtr = nullptr;
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
        uniDataTmpPtr[0] = srcDataPtr[0];
        if (definedOutputs[FIRST_UNIQUE_IDX]) {
            firstTmpPtr[0] = 0;
        }
        if (definedOutputs[INPUT_TO_UNIQ_IDX]) {
            inToOutTmpPtr[0] = 0;
        }
        if (definedOutputs[OCCURRENCES_NUM]) {
            std::fill(occurTmpPtr, occurTmpPtr + inputLen, 1);
        }
        uniqueLen = 1;

        for (size_t i = 1; i < inputLen; i++) {
            bool found = false;
            size_t j = 0;
            for (; j < uniqueLen; j++) {
                if (uniDataTmpPtr[j] == srcDataPtr[i]) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                uniDataTmpPtr[uniqueLen] = srcDataPtr[i];
                if (definedOutputs[FIRST_UNIQUE_IDX]) {
                    firstTmpPtr[uniqueLen] = i;
                }
                uniqueLen++;
            } else {
                if (definedOutputs[OCCURRENCES_NUM]) {
                    occurTmpPtr[j]++;
                }
            }
            if (definedOutputs[INPUT_TO_UNIQ_IDX]) {
                inToOutTmpPtr[i] = j;
            }
        }
    }

    redefineOutputMemory({ {uniqueLen}, {uniqueLen}, {inputLen}, {uniqueLen}});

    T* uniDataPtr = reinterpret_cast<T*>(getChildEdgesAtPort(UNIQUE_DATA)[0]->getMemoryPtr()->getData());
    cpu_parallel_memcpy(uniDataPtr, uniDataTmpPtr, uniqueLen * sizeof(T));
    if (definedOutputs[FIRST_UNIQUE_IDX]) {
        copyOutput(FIRST_UNIQUE_IDX, firstUniTmp.data(), uniqueLen);
    }
    if (definedOutputs[INPUT_TO_UNIQ_IDX]) {
        copyOutput(INPUT_TO_UNIQ_IDX, inToOutTmpPtr, inputLen);
    }
    if (definedOutputs[OCCURRENCES_NUM]) {
        copyOutput(OCCURRENCES_NUM, occurTmpPtr, uniqueLen);
    }
}

template <typename T>
void Unique::slicedTensorExec() {
    auto inDataMemPtr = getParentEdgeAt(IN_DATA)->getMemoryPtr();
    auto srcDataPtr = reinterpret_cast<const T*>(inDataMemPtr->getData());

    uint8_t *firstTmpPtr = nullptr, *inToOutTmpPtr = nullptr, *occurTmpPtr = nullptr;
     if (definedOutputs[FIRST_UNIQUE_IDX]) {
         firstTmpPtr = reinterpret_cast<uint8_t*>(firstUniTmp.data());
     }
     if (definedOutputs[INPUT_TO_UNIQ_IDX]) {
         inToOutTmpPtr = reinterpret_cast<uint8_t*>(inToOutTmp.data());
     }
     if (definedOutputs[OCCURRENCES_NUM]) {
         occurTmpPtr = reinterpret_cast<uint8_t*>(occurTmp.data());
     }

    const auto& srcDataShape = inDataMemPtr->getStaticDims();

    const auto axisDim = srcDataShape[axis];
    int64_t outerLen = 1lu;
    if (axis > 0) {
        outerLen = std::accumulate(srcDataShape.begin(), srcDataShape.begin() + axis, 1, std::multiplies<Dim>());
    }
    int64_t innerLen = 1;
    if (static_cast<size_t>(axis) < srcDataShape.size() - 1) {
        innerLen = std::accumulate(srcDataShape.begin() + axis + 1, srcDataShape.end(), 1, std::multiplies<Dim>());
    }
    const auto innerSizeB = innerLen * sizeof(T);
    const auto srcOuterStep = innerLen * axisDim;

    if (definedOutputs[FIRST_UNIQUE_IDX]) {
        if (outputsPrc[FIRST_UNIQUE_IDX] == Precision::I32) {
            reinterpret_cast<int32_t*>(firstTmpPtr)[0] = 0;
        } else {
            reinterpret_cast<int64_t*>(firstTmpPtr)[0] = 0;
        }
    }
    if (definedOutputs[INPUT_TO_UNIQ_IDX]) {
        if (outputsPrc[INPUT_TO_UNIQ_IDX] == Precision::I32) {
            reinterpret_cast<int32_t*>(inToOutTmpPtr)[0] = 0;
        } else {
            reinterpret_cast<int64_t*>(inToOutTmpPtr)[0] = 0;
        }
    }
    if (definedOutputs[OCCURRENCES_NUM]) {
        if (outputsPrc[OCCURRENCES_NUM] == Precision::I32) {
            auto dstMem = reinterpret_cast<int32_t*>(occurTmpPtr);
            std::fill(dstMem, dstMem + axisDim, 1);
        } else {
            auto dstMem = reinterpret_cast<int64_t*>(occurTmpPtr);
            std::fill(dstMem, dstMem + axisDim, 1);
        }
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
                last1  += srcOuterStep;
                first2 += srcOuterStep;
            }
            if (equal) {
                break;
            }
        }
        if (!equal) {
            if (definedOutputs[FIRST_UNIQUE_IDX]) {
                if (outputsPrc[FIRST_UNIQUE_IDX] == Precision::I32) {
                    reinterpret_cast<int32_t*>(firstTmpPtr)[uniqueLen] = static_cast<int32_t>(a);
                } else {
                    reinterpret_cast<int64_t*>(firstTmpPtr)[uniqueLen] = static_cast<int64_t>(a);
                }
            }

            uniqIdx[uniqueLen++] = a;
        } else {
            if (definedOutputs[OCCURRENCES_NUM]) {
                if (outputsPrc[OCCURRENCES_NUM] == Precision::I32) {
                    reinterpret_cast<int32_t*>(occurTmpPtr)[uIdx]++;
                } else {
                    reinterpret_cast<int64_t*>(occurTmpPtr)[uIdx]++;
                }
            }
        }
        if (definedOutputs[INPUT_TO_UNIQ_IDX]) {
            if (outputsPrc[INPUT_TO_UNIQ_IDX] == Precision::I32) {
                reinterpret_cast<int32_t*>(inToOutTmpPtr)[a] = uIdx;
            } else {
                reinterpret_cast<int64_t*>(inToOutTmpPtr)[a] = uIdx;
            }
        }
    }

    // Redefinition of output shapes.
    auto dstDataShape = srcDataShape;
    dstDataShape[axis] = uniqueLen;
    redefineOutputMemory({ dstDataShape, {uniqueLen}, {axisDim}, {uniqueLen}});

    uint8_t *firstPtr = nullptr, *inToOutPtr = nullptr, *occurNPtr = nullptr;
    if (definedOutputs[FIRST_UNIQUE_IDX]) {
        firstPtr = reinterpret_cast<uint8_t*>(getChildEdgesAtPort(FIRST_UNIQUE_IDX)[0]->getMemoryPtr()->getData());
    }
    if (definedOutputs[INPUT_TO_UNIQ_IDX]) {
        inToOutPtr = reinterpret_cast<uint8_t*>(getChildEdgesAtPort(INPUT_TO_UNIQ_IDX)[0]->getMemoryPtr()->getData());
    }
    if (definedOutputs[OCCURRENCES_NUM]) {
        occurNPtr = reinterpret_cast<uint8_t*>(getChildEdgesAtPort(OCCURRENCES_NUM)[0]->getMemoryPtr()->getData());
    }

    T* dstDataPtr = reinterpret_cast<T*>(getChildEdgesAtPort(UNIQUE_DATA)[0]->getMemoryPtr()->getData());
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
        uint8_t *first1 = firstPtr, *first2 = firstTmpPtr;
        uint8_t *occurN1 = occurNPtr, *occurN2 = occurTmpPtr;
        uint8_t *inToOut1 = inToOutPtr, *inToOut2 = inToOutTmpPtr;

        const bool defined3outputs = definedOutputs[FIRST_UNIQUE_IDX] || definedOutputs[OCCURRENCES_NUM] || definedOutputs[INPUT_TO_UNIQ_IDX];

        for (int64_t o = outerLen - 1; o >= 0; o--) { // Backward loop through the outer block.
            const int64_t pos1Lim = o * dstOuterStep;
            int64_t pos1 = pos1Lim + innerLen - 1;
            for (; pos1 >= pos1Lim ; pos1--) { // Backward loop through the inner block.
                int64_t pos2 = pos1;
                for (int64_t k = 0; k < static_cast<int64_t>(uniqueLen); k++, pos2 += innerLen) {
                    colToSort[k] = { dst1[pos2], k };
                }
                std::stable_sort(colToSort.begin(), colToSort.end(), [](const OrdEl &el1, const OrdEl &el2) { return el1.val < el2.val; });

                // Permutation
                parallel_for2d(outerLen, uniqueLen, [&](int64_t ot, size_t u) {
                    auto src = dst1 + ot * dstOuterStep + colToSort[u].idx * innerLen;
                    auto dst = dst2 + ot * dstOuterStep + u * innerLen;

                    cpu_memcpy(dst, src, innerSizeB);
                });

                if (defined3outputs) {
                    parallel_for(uniqueLen, [&](size_t u) {
                        if (definedOutputs[FIRST_UNIQUE_IDX]) {
                            if (outputsPrc[FIRST_UNIQUE_IDX] == Precision::I32) {
                                reinterpret_cast<int32_t*>(first1)[u] = reinterpret_cast<int32_t*>(first2)[colToSort[u].idx];
                            } else {
                                reinterpret_cast<int64_t*>(first1)[u] = reinterpret_cast<int64_t*>(first2)[colToSort[u].idx];
                            }
                        }
                        if (definedOutputs[OCCURRENCES_NUM]) {
                            if (outputsPrc[OCCURRENCES_NUM] == Precision::I32) {
                                reinterpret_cast<int32_t*>(occurN1)[u] = reinterpret_cast<int32_t*>(occurN2)[colToSort[u].idx];
                            } else {
                                reinterpret_cast<int64_t*>(occurN1)[u] = reinterpret_cast<int64_t*>(occurN2)[colToSort[u].idx];
                            }
                        }
                        if (definedOutputs[INPUT_TO_UNIQ_IDX]) {
                            if (outputsPrc[INPUT_TO_UNIQ_IDX] == Precision::I32) {
                                auto inToOut1_i32 = reinterpret_cast<int32_t*>(inToOut1);
                                auto inToOut2_i32 = reinterpret_cast<int32_t*>(inToOut2);
                                for (size_t ax = 0; ax < axisDim; ax++) {
                                    if (inToOut2_i32[ax] == colToSort[u].idx) {
                                        inToOut1_i32[ax] = static_cast<int32_t>(u);
                                    }
                                }
                            } else {
                                auto inToOut1_i64 = reinterpret_cast<int64_t*>(inToOut1);
                                auto inToOut2_i64 = reinterpret_cast<int64_t*>(inToOut2);
                                for (size_t ax = 0; ax < axisDim; ax++) {
                                    if (inToOut2_i64[ax] == colToSort[u].idx) {
                                        inToOut1_i64[ax] = static_cast<int64_t>(u);
                                    }
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
            const auto cpyLen = uniqueLen * (outputsPrc[FIRST_UNIQUE_IDX] == Precision::I32 ? sizeof(int32_t) : sizeof(int64_t));
            cpu_parallel_memcpy(firstPtr, first2, cpyLen);
        }
        if (definedOutputs[INPUT_TO_UNIQ_IDX] && inToOut2 != inToOutPtr) {
            const auto cpyLen = axisDim * (outputsPrc[INPUT_TO_UNIQ_IDX] == Precision::I32 ? sizeof(int32_t) : sizeof(int64_t));
            cpu_parallel_memcpy(inToOutPtr, inToOut2, cpyLen);
        }
        if (definedOutputs[OCCURRENCES_NUM] && occurN2 != occurNPtr) {
            const auto cpyLen = uniqueLen * (outputsPrc[OCCURRENCES_NUM] == Precision::I32 ? sizeof(int32_t) : sizeof(int64_t));
            cpu_parallel_memcpy(occurNPtr, occurN2, cpyLen);
        }
    } else {
        if (definedOutputs[FIRST_UNIQUE_IDX]) {
            const auto cpyLen = uniqueLen * (outputsPrc[FIRST_UNIQUE_IDX] == Precision::I32 ? sizeof(int32_t) : sizeof(int64_t));
            cpu_parallel_memcpy(firstPtr, firstUniTmp.data(), cpyLen);
        }
        if (definedOutputs[INPUT_TO_UNIQ_IDX]) {
            const auto cpyLen = axisDim * (outputsPrc[INPUT_TO_UNIQ_IDX] == Precision::I32 ? sizeof(int32_t) : sizeof(int64_t));
            cpu_parallel_memcpy(inToOutPtr, inToOutTmp.data(), cpyLen);
        }
        if (definedOutputs[OCCURRENCES_NUM]) {
            const auto cpyLen = uniqueLen * (outputsPrc[OCCURRENCES_NUM] == Precision::I32 ? sizeof(int32_t) : sizeof(int64_t));
            cpu_parallel_memcpy(occurNPtr, occurTmp.data(), cpyLen);
        }
    }
}

void Unique::copyOutput(size_t outIdx, const int64_t* srcPtr, size_t len) {
    const auto outMem = getChildEdgesAtPort(outIdx)[0]->getMemoryPtr();
    if (outMem->getDataType() == dnnl::memory::data_type::s64) {
        cpu_parallel_memcpy(outMem->getData(), srcPtr, len * sizeof(int64_t));
    } else if (outMem->getDataType() == dnnl::memory::data_type::s32) {
        auto outPtr = reinterpret_cast<int32_t *>(outMem->getData());
        parallel_for(len, [&](size_t i) {
            outPtr[i] = static_cast<int32_t>(srcPtr[i]);
        });
    }
}
