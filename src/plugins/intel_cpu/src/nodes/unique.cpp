// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "unique.hpp"
#include <ngraph/opsets/opset1.hpp>
#include <utils/shape_inference/shape_inference_internal_dyn.hpp>

using namespace InferenceEngine;
using namespace ov::intel_cpu;
using namespace ov::intel_cpu::node;

#define THROW_ERROR IE_THROW() << getTypeStr() << " node with name '" << getName() << "' "

bool Unique::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!ov::is_type<op::v10::Unique>(op)) {
            errorMessage = "Not supported Unique operation version. CPU plug-in supports only 10th version.";
            return false;
        }
        if (op->get_input_size() > AXIS && !ov::is_type<ov::op::v0::Constant>(op->get_input_node_ptr(AXIS))) {
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
        THROW_ERROR << "has incorrect number of input/output edges.";

    for (int i = 0; i < 4; i++) {
        definedOutputs[i] = !op->get_output_target_inputs(i).empty();
    }

    sorted = ov::as_type_ptr<ov::op::v10::Unique>(op)->get_sorted();
    if (op->get_input_size() > AXIS) {
        flattened = false;
        axis = ov::as_type<ov::op::v0::Constant>(op->get_input_node_ptr(AXIS))->cast_vector<int>()[0];
        if (axis < 0) {
            axis += op->get_input_partial_shape(IN_DATA).rank().get_length();
        }
        if (axis < 0 || axis >= op->get_input_partial_shape(IN_DATA).rank().get_length()) {
            THROW_ERROR << "has invalid axis value: " << ov::as_type<ov::op::v0::Constant>(op->get_input_node_ptr(AXIS))->cast_vector<int>()[0];
        }
    } else {
        flattened = true;
    }
}

void Unique::initSupportedPrimitiveDescriptors() {
    dataPrecision = getOriginalInputPrecisionAtPort(IN_DATA);
    if (dataPrecision != Precision::I32 && dataPrecision != Precision::I8 && dataPrecision != Precision::U8) {
        dataPrecision = Precision::FP32;
    }
    dataTypeSize = dataPrecision.size();
    const InferenceEngine::Precision axisPrecision = Precision::I32;

    impl_desc_type implType = ref;

    std::vector<PortConfigurator> inPortConfigs = { {LayoutType::ncsp, dataPrecision} };
    if (!flattened) {
        inPortConfigs.push_back({LayoutType::ncsp, axisPrecision});
    }
    std::vector<PortConfigurator> outPortConfigs;
    for (int i = 0; i < 4; i++) {
        outPortConfigs.push_back({LayoutType::ncsp, i == 0 ? dataPrecision : axisPrecision});
    }

    addSupportedPrimDesc(inPortConfigs, outPortConfigs, implType, isDynamicNode());
}

void Unique::createPrimitive() {
    Node::createPrimitive();
}

void Unique::prepareParams() {
    auto& dataMemPtr = getParentEdgeAt(IN_DATA)->getMemoryPtr();
    if (!dataMemPtr || !dataMemPtr->isAllocated()) {
        THROW_ERROR << " has not allocated input data memory.";
    }
    for (int i = 0; i < 4; i++) {
        if (definedOutputs[i]) {
            auto& dstMemPtr = getChildEdgeAt(i)->getMemoryPtr();
            if (!dstMemPtr || !dstMemPtr->isAllocated()) {
                THROW_ERROR << " has not allocated output memory at port " << i;
            }
        }
    }
    if (getSelectedPrimitiveDescriptor() == nullptr) {
        THROW_ERROR << " has unidentified preferable primitive descriptor.";
    }

    size_t srcLen = 1;
    if (flattened) {
        srcLen = getParentEdgeAt(IN_DATA)->getMemoryPtr()->GetSize() / dataTypeSize;
    } else {
        auto dstDataShape = getParentEdgeAt(IN_DATA)->getMemoryPtr()->getStaticDims();
        srcLen = dstDataShape[axis];
    }
    firstUniTmp.resize(srcLen, 0);
    inToOutTmp.resize(srcLen);
    occurTmp.resize(srcLen);
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
              OV_CASE(Precision::I8, int8_t),
              OV_CASE(Precision::U8, uint8_t))
    } else {
        OV_SWITCH(intel_cpu, slicedExec, this, dataPrecision,
              OV_CASE(Precision::FP32, float),
              OV_CASE(Precision::I32, int32_t),
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
    const T* srcDataPtr = reinterpret_cast<const T*>(getParentEdgeAt(IN_DATA)->getMemoryPtr()->GetPtr());
    const size_t inputLen = getParentEdgeAt(IN_DATA)->getMemoryPtr()->GetSize() / sizeof(T);
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
        std::memcpy(uniDataTmpPtr, srcDataPtr, inputLen * sizeof(T));
        std::sort(uniDataTmpPtr, uniDataTmpPtr + inputLen);
        auto last = std::unique(uniDataTmpPtr, uniDataTmpPtr + inputLen);
        uniqueLen = last - uniDataTmpPtr;

        if (definedOutputs[FIRST_UNIQUE_IDX]) {
            T* first = uniDataTmpPtr;
            for (T* it = first; it < last; it++) {
                for (int i = 0; i < inputLen; i++) {
                    if (srcDataPtr[i] == *it) {
                        *firstTmpPtr++ = i;
                        first++;
                        break;
                    }
                }
            }
        }
        if (definedOutputs[INPUT_TO_UNIQ_IDX]) {
            for (int i = 0; i < inputLen; i++) {
                if (i > 0 && srcDataPtr[i] == srcDataPtr[i - 1]) {
                    inToOutTmpPtr[i] = inToOutTmpPtr[i - 1];
                    continue;
                }
                for (int j = 0; j < uniqueLen; j++) {
                    if (srcDataPtr[i] == uniDataTmpPtr[j]) {
                        inToOutTmpPtr[i] = j;
                        break;
                    }
                }
            }
        }
        if (definedOutputs[OCCURRENCES_NUM]) {
            std::fill(occurTmpPtr, occurTmpPtr + uniqueLen, 0);
            for (int j = 0; j < uniqueLen; j++) {
                for (int i = 0; i < inputLen; i++) {
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

        for (int i = 1; i < inputLen; i++) {
            bool found = false;
            int j = 0;
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

    T* uniDataPtr = reinterpret_cast<T*>(getChildEdgesAtPort(UNIQUE_DATA)[0]->getMemoryPtr()->GetPtr());
    memcpy(uniDataPtr, uniDataTmpPtr, uniqueLen * sizeof(T));
    if (definedOutputs[FIRST_UNIQUE_IDX]) {
        int *firstPtr = reinterpret_cast<int*>(getChildEdgesAtPort(FIRST_UNIQUE_IDX)[0]->getMemoryPtr()->GetPtr());
        memcpy(firstPtr, firstUniTmp.data(), uniqueLen * sizeof(int));
    }
    if (definedOutputs[INPUT_TO_UNIQ_IDX]) {
        auto inToOutPtr = reinterpret_cast<int*>(getChildEdgesAtPort(INPUT_TO_UNIQ_IDX)[0]->getMemoryPtr()->GetPtr());
        memcpy(inToOutPtr, inToOutTmp.data(), inputLen * sizeof(int));
    }
    if (definedOutputs[OCCURRENCES_NUM]) {
        auto occurPtr = reinterpret_cast<int*>(getChildEdgesAtPort(OCCURRENCES_NUM)[0]->getMemoryPtr()->GetPtr());
        memcpy(occurPtr, occurTmp.data(), uniqueLen * sizeof(int));
    }
}

template <typename T>
void Unique::slicedTensorExec() {
    const T* srcDataPtr = reinterpret_cast<const T*>(getParentEdgeAt(IN_DATA)->getMemoryPtr()->GetPtr());
    const size_t inputLen = getParentEdgeAt(IN_DATA)->getMemoryPtr()->GetSize() / sizeof(T);
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

    const auto& srcDataShape = getParentEdgeAt(IN_DATA)->getMemoryPtr()->getStaticDims();

    const auto cmpBlNum = srcDataShape[axis]; // Blocks to compare.
    int64_t partsInBl = 1; // Parts in block
    if (axis > 0) {
        partsInBl = std::accumulate(srcDataShape.begin(), srcDataShape.begin() + axis, 1, std::multiplies<Dim>());
    }
    int64_t elPerPart = 1; // Elements number in part.
    if (axis < srcDataShape.size() - 1) {
        elPerPart = std::accumulate(srcDataShape.begin() + axis + 1, srcDataShape.end(), 1, std::multiplies<Dim>());
    }
    const auto partLenB = elPerPart * dataPrecision.size();
    const auto partStep = elPerPart * cmpBlNum;

    if (definedOutputs[FIRST_UNIQUE_IDX]) {
        firstTmpPtr[0] = 0;
    }
    if (definedOutputs[INPUT_TO_UNIQ_IDX]) {
        inToOutTmpPtr[0] = 0;
    }
    if (definedOutputs[OCCURRENCES_NUM]) {
        occurTmpPtr[0] = 1;
        std::fill(occurTmpPtr, occurTmpPtr + cmpBlNum, 1);
    }

    uniqueLen = 1;
    std::vector<int64_t> uniqIdx(cmpBlNum, 0);
    for (int b1 = 1; b1 < cmpBlNum; b1++) {
        auto first1 = srcDataPtr + b1 * elPerPart;
        auto last1 = srcDataPtr + (b1 + 1) * elPerPart;
        bool equal = true;
        int b2 = 0;
        // Compare with unique blocks.
        for (; b2 < uniqueLen; b2++) {
            auto first2 = srcDataPtr + uniqIdx[b2] * elPerPart;
            equal = true;
            for (int p = 0; p < partsInBl; p++) {
                equal = std::equal(first1, last1, first2);
                if (!equal) {
                    break;
                }
                first1 += partStep;
                last1  += partStep;
                first2 += partStep;
            }
            if (equal) {
                break;
            }
        }
        if (!equal) {
            if (definedOutputs[FIRST_UNIQUE_IDX]) {
                firstTmpPtr[uniqueLen] = b1;
            }

            uniqIdx[uniqueLen++] = b1;
        } else {
            if (definedOutputs[OCCURRENCES_NUM]) {
                occurTmpPtr[b2]++;
            }
        }
        if (definedOutputs[INPUT_TO_UNIQ_IDX]) {
            inToOutTmpPtr[b1] = b2;
        }
    }

    const auto dstPrtStep = elPerPart * uniqueLen;
    for (int b1 = 0; b1 < uniqueLen; b1++) {
        auto first1 = srcDataPtr + uniqIdx[b1] * elPerPart;
        auto first2 = uniDataTmpPtr + b1 * elPerPart;
        for (int p = 0; p < partsInBl; p++) {
            memcpy(first2, first1, partLenB);
            first1 += partStep;
            first2 += dstPrtStep;
        }
    }

    if (sorted) {
        const auto elInBl = elPerPart * partsInBl;
        struct OrdEl {
            T val;
            int64_t idx;
        };

        std::vector<OrdEl> colToSort(uniqueLen);
        std::vector<int64_t> moveTo(uniqueLen);
        for (int k = 0; k < uniqueLen; k++) {
            moveTo[k] = k;
        }
        std::vector<T> buff1(elPerPart);
        std::vector<T> buff2(elPerPart);
        for (int64_t p = partsInBl - 1; p >= 0; p--) {
            for (int64_t e = elPerPart - 1; e >= 0 ; e--) {
                int64_t pos1 = p * dstPrtStep + e;
                for (int64_t i = 0; i < static_cast<int64_t>(uniqueLen); i++) {
                    int64_t pos2 = i * elInBl + pos1;
                    colToSort[i] = {uniDataTmpPtr[pos2], i};
                }
                std::stable_sort(colToSort.begin(), colToSort.end(), [](const OrdEl &el1, const OrdEl &el2) { return el1.val < el2.val; });
                for (int k = 0; k < uniqueLen; k++) {
                    moveTo[colToSort[k].idx] = k;
                }

                // perm
                for (int64_t pb = 0; pb < partsInBl; pb++) {
                    auto currDst = uniDataTmpPtr + pb * dstPrtStep;
                    memcpy(buff1.data(), currDst, partLenB);
                    auto dstIdx = moveTo[0];
                    for (size_t b = 0; b < uniqueLen; b++) {
                        if (dstIdx == moveTo[dstIdx]) {
                            dstIdx = moveTo[dstIdx + 1];
                            continue;
                        }
                        T* dst = currDst + dstIdx * elPerPart;

                        auto& bSrc = b % 2 == 0 ? buff1 : buff2;
                        auto& bDst = b % 2 == 0 ? buff2 : buff1;
                        memcpy(bDst.data(), dst, partLenB);
                        memcpy(dst, bSrc.data(), partLenB);

                        dstIdx = moveTo[dstIdx];
                    }
                }

                auto mPos = moveTo[0];
                int32_t firstSrc = 0, firstDst = 0, ocSrc = 0, ocDst = 0;
                if (definedOutputs[FIRST_UNIQUE_IDX]) {
                    firstSrc = firstTmpPtr[0];
                }
                if (definedOutputs[OCCURRENCES_NUM]) {
                    ocSrc = occurTmpPtr[0];
                }
                for (int k = 0; k < uniqueLen; k++) {
                    if (mPos == moveTo[mPos]) {
                        mPos = moveTo[mPos + 1];
                        continue;
                    }

                    if (definedOutputs[FIRST_UNIQUE_IDX]) {
                        auto& fSrc = k % 2 == 0 ? firstSrc : firstDst;
                        auto& fDst = k % 2 == 0 ? firstDst : firstSrc;
                        fDst = firstTmpPtr[mPos];
                        firstTmpPtr[mPos] = fSrc;
                    }
                    if (definedOutputs[OCCURRENCES_NUM]) {
                        auto& oSrc = k % 2 == 0 ? ocSrc : ocDst;
                        auto& oDst = k % 2 == 0 ? ocDst : ocSrc;
                        oDst = occurTmpPtr[mPos];
                        occurTmpPtr[mPos] = oSrc;
                    }

                    mPos = moveTo[mPos];
                }
            }
        }

        if (definedOutputs[INPUT_TO_UNIQ_IDX]) {
            for (int b1 = 0; b1 < cmpBlNum; b1++) {
                auto first1 = srcDataPtr + b1 * elPerPart;
                auto last1 = srcDataPtr + (b1 + 1) * elPerPart;
                bool equal = true;
                for (int b2 = 0; b2 < uniqueLen; b2++) {
                    auto first2 = uniDataTmpPtr + b2 * elPerPart;
                    equal = true;
                    for (int p = 0; p < partsInBl; p++) {
                        equal = std::equal(first1, last1, first2);
                        if (!equal) {
                            break;
                        }
                        first2 += dstPrtStep;
                    }
                    if (equal) {
                        inToOutTmpPtr[b1] = b2;
                    }
                }
            }
        }
    }

    auto dstDataShape = srcDataShape;
    dstDataShape[axis] = uniqueLen;
    redefineOutputMemory({ dstDataShape, {uniqueLen}, {cmpBlNum}, {uniqueLen}});

    T* uniDataPtr = reinterpret_cast<T*>(getChildEdgesAtPort(UNIQUE_DATA)[0]->getMemoryPtr()->GetPtr());
    memcpy(uniDataPtr, uniDataTmpPtr, getChildEdgesAtPort(UNIQUE_DATA)[0]->getMemoryPtr()->GetSize());
    if (definedOutputs[FIRST_UNIQUE_IDX]) {
        int *firstPtr = reinterpret_cast<int*>(getChildEdgesAtPort(FIRST_UNIQUE_IDX)[0]->getMemoryPtr()->GetPtr());
        memcpy(firstPtr, firstUniTmp.data(), uniqueLen * sizeof(int));
    }
    if (definedOutputs[INPUT_TO_UNIQ_IDX]) {
        auto inToOutPtr = reinterpret_cast<int*>(getChildEdgesAtPort(INPUT_TO_UNIQ_IDX)[0]->getMemoryPtr()->GetPtr());
        memcpy(inToOutPtr, inToOutTmp.data(), cmpBlNum * sizeof(int));
    }
    if (definedOutputs[OCCURRENCES_NUM]) {
        auto occurPtr = reinterpret_cast<int*>(getChildEdgesAtPort(OCCURRENCES_NUM)[0]->getMemoryPtr()->GetPtr());
        memcpy(occurPtr, occurTmp.data(), uniqueLen * sizeof(int));
    }
}
