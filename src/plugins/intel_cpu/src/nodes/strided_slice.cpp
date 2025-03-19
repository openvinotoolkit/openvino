// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "strided_slice.h"

#include <cmath>
#include <string>

#include "common/cpu_memcpy.h"
#include "input.h"
#include "openvino/core/parallel.hpp"
#include "openvino/opsets/opset1.hpp"
#include "shape_inference/custom/strided_slice.hpp"
#include "slice_shape_inference_utils.hpp"

using namespace dnnl;

namespace ov::intel_cpu::node {

bool StridedSlice::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!ov::is_type_any_of<ov::op::v1::StridedSlice, ov::op::v8::Slice, ov::op::v15::SliceScatter>(op)) {
            errorMessage = "Only StridedSlice from opset1, Slice from opset8 and SliceScatter from opset15 operations "
                           "are supported.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

StridedSlice::StridedSlice(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, StridedSliceShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    attrs.isStridedSliceOp = ov::is_type<ov::op::v1::StridedSlice>(op);
    attrs.isSliceScatterOp = ov::is_type<ov::op::v15::SliceScatter>(op);
    if (!attrs.isSliceScatterOp) {
        attrs.BEGIN_ID = 1;
        attrs.END_ID = 2;
        attrs.STRIDE_ID = 3;
        attrs.AXES_ID = 4;
    } else {
        attrs.BEGIN_ID = 2;
        attrs.END_ID = 3;
        attrs.STRIDE_ID = 4;
        attrs.AXES_ID = 5;
    }

    if ((attrs.isStridedSliceOp && (inputShapes.size() < 3 || inputShapes.size() > 4)) ||
        (!attrs.isStridedSliceOp &&
         (inputShapes.size() < (attrs.STRIDE_ID + 1) || inputShapes.size() > (attrs.AXES_ID + 1)))) {
        THROW_CPU_NODE_ERR("has incorrect number of input edges");
    }
    if (outputShapes.size() != 1) {
        THROW_CPU_NODE_ERR("has incorrect number of output edges");
    }

    if (inputShapes.size() > attrs.STRIDE_ID) {
        isStrideSpecified = true;
    }

    if (inputShapes.size() > attrs.AXES_ID) {
        isAxesSpecified = true;
    }

    for (size_t i = 0lu; i < op->get_input_size(); i++) {
        isConstantInput[i] = ov::is_type<ov::op::v0::Constant>(op->get_input_node_shared_ptr(i));
        if (!isConstantInput[i] && one_of(i, attrs.BEGIN_ID, attrs.END_ID, attrs.STRIDE_ID) &&
            !attrs.isSliceScatterOp) {
            shapeHasDataDependency = true;
        }
    }
    hasConstAttrInputs = !shapeHasDataDependency;
    if (isAxesSpecified) {
        hasConstAttrInputs &= isConstantInput[attrs.AXES_ID];
    }

    const size_t inputRank = getInputShapeAtPort(attrs.DATA_ID).getRank();
    const size_t outputRank = getOutputShapeAtPort(0).getRank();
    const size_t nDims = std::max(inputRank, outputRank);

    if (attrs.isStridedSliceOp) {
        auto ss = ov::as_type_ptr<const ov::op::v1::StridedSlice>(op);

        auto createMask = [&](const std::vector<int64_t>& origMask, const int bit = 0, bool needReverse = false) {
            std::vector<int> mask(origMask.size());
            for (size_t i = 0; i < mask.size(); i++) {
                mask[i] = static_cast<int>(origMask[i]);
                if (needReverse) {
                    mask[i] = 1 - mask[i];
                }
            }
            for (size_t i = mask.size(); i < nDims; ++i) {
                mask.push_back(bit);
            }
            return mask;
        };

        attrs.beginMask = createMask(ss->get_begin_mask(), 1, true);
        attrs.endMask = createMask(ss->get_end_mask(), 1, true);
        attrs.newAxisMask = createMask(ss->get_new_axis_mask());
        attrs.shrinkAxisMask = createMask(ss->get_shrink_axis_mask());
        attrs.ellipsisMask = createMask(ss->get_ellipsis_mask());
    } else {
        const size_t length = outputShapes[0].getRank();
        if (inputShapes.size() > attrs.AXES_ID) {
            attrs.beginMask = std::vector<int>(length, 0);
            attrs.endMask = std::vector<int>(length, 0);
        } else {
            attrs.beginMask = std::vector<int>(length, 1);
            attrs.endMask = std::vector<int>(length, 1);
        }
        attrs.newAxisMask = std::vector<int>(length, 0);
        attrs.shrinkAxisMask = std::vector<int>(length, 0);
        attrs.ellipsisMask = std::vector<int>(length, 0);
    }

    if (attrs.isStridedSliceOp) {
        for (size_t i = 0; i < attrs.ellipsisMask.size(); i++) {
            attrs.ellipsisMaskCounter += attrs.ellipsisMask[i];
            attrs.ellipsisPos1 = attrs.ellipsisMask[i] == 1 && attrs.ellipsisPos1 == -1 ? i : attrs.ellipsisPos1;
        }
        if (attrs.ellipsisMaskCounter > 1) {
            THROW_CPU_NODE_ERR("has incorrect 'Ellipsis_mask'. Only one non-zero bit is allowed");
        }

        int newAxis = std::accumulate(attrs.newAxisMask.begin(), attrs.newAxisMask.end(), 0);
        int shrinkAxis = std::accumulate(attrs.shrinkAxisMask.begin(), attrs.shrinkAxisMask.end(), 0);
        attrs.equalDims = newAxis == 0 && shrinkAxis == 0;
    } else {
        attrs.equalDims = true;
    }

    auto fillingInParameters = [&](std::vector<int>& parameter, const size_t type, const int value) {
        if (!isConstantInput[type]) {
            return;
        }

        const auto constNode = ov::as_type_ptr<const ov::opset1::Constant>(op->get_input_node_shared_ptr(type));
        parameter = constNode->cast_vector<int>();

        auto size = constNode->get_shape()[0];
        if (type != attrs.AXES_ID && attrs.ellipsisMaskCounter == 0 && size < nDims) {
            for (size_t i = size; i < nDims; i++) {
                parameter.push_back(value);
            }
        }
    };

    fillingInParameters(attrs.begin, attrs.BEGIN_ID, 0);
    fillingInParameters(attrs.end, attrs.END_ID, 0);
    if (inputShapes.size() > attrs.STRIDE_ID) {
        fillingInParameters(attrs.stride, attrs.STRIDE_ID, 1);
    }
    if (inputShapes.size() > attrs.AXES_ID) {
        fillingInParameters(attrs.axes, attrs.AXES_ID, 0);
    }
}

void StridedSlice::getSupportedDescriptors() {}

static void addHiddenDims(StridedSlice::StridedSliceAttributes& attrs,
                          const size_t inputRank,
                          const size_t outputRank,
                          bool withAxis) {
    if (withAxis) {
        std::vector<int> beginTmp(outputRank, 0);
        std::vector<int> endTmp(outputRank, -1);
        std::vector<int> strideTmp(outputRank, 1);
        size_t i = 0lu;
        for (auto& a : attrs.axes) {
            if (a < 0) {
                a += outputRank;
            }
            beginTmp[a] = attrs.begin[i];
            endTmp[a] = attrs.end[i];
            strideTmp[a] = attrs.stride[i++];
            attrs.beginMask[a] = 1;
            attrs.endMask[a] = 1;
        }
        attrs.begin = beginTmp;
        attrs.end = endTmp;
        attrs.stride = strideTmp;
    }

    if (inputRank > 3 && attrs.equalDims && attrs.ellipsisMaskCounter == 1) {
        // all masks and input parameters are for planar layouts. So if we use blocked or per channel layout and
        // there is ellipsis should to add default values in hidden dimensions to know real order of mask or parameter
        // values
        size_t afterDims = attrs.begin.size() - attrs.ellipsisPos1 - 1;
        size_t ellipsisPos2 = inputRank - afterDims - 1;

        auto addHiddenDims = [&](std::vector<int>& data, const int bit = 0) {
            std::vector<int> temp;
            temp.reserve(attrs.ellipsisPos1);
            for (int i = 0; i < attrs.ellipsisPos1; i++) {
                temp.push_back(data[i]);
            }
            for (size_t i = attrs.ellipsisPos1; i < ellipsisPos2 + 1; i++) {
                temp.push_back(bit);
            }
            for (size_t i = 1; i < inputRank - ellipsisPos2; i++) {
                temp.push_back(data[i + attrs.ellipsisPos1]);
            }
            data = temp;
        };

        addHiddenDims(attrs.begin);
        addHiddenDims(attrs.end);
        addHiddenDims(attrs.stride, 1);
        addHiddenDims(attrs.beginMask);
        addHiddenDims(attrs.endMask);
        addHiddenDims(attrs.ellipsisMask);
        addHiddenDims(attrs.newAxisMask);
        addHiddenDims(attrs.shrinkAxisMask);
    }
}

void StridedSlice::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    const ov::element::Type dataPrecision = getOriginalInputPrecisionAtPort(attrs.DATA_ID);
    const ov::element::Type iPrecision = ov::element::i32;
    attrs.dataSize = dataPrecision.size();

    const size_t nDims = getInputShapeAtPort(attrs.DATA_ID).getRank();

    NodeConfig config;
    config.inConfs.resize(getParentEdges().size());
    config.inConfs[attrs.DATA_ID].inPlace(-1);
    config.inConfs[attrs.BEGIN_ID].inPlace(-1);
    config.inConfs[attrs.END_ID].inPlace(-1);
    config.inConfs[attrs.DATA_ID].constant(isConstantInput[attrs.DATA_ID]);
    config.inConfs[attrs.BEGIN_ID].constant(isConstantInput[attrs.BEGIN_ID]);
    config.inConfs[attrs.END_ID].constant(isConstantInput[attrs.END_ID]);
    if (isStrideSpecified) {
        config.inConfs[attrs.STRIDE_ID].inPlace(-1);
        config.inConfs[attrs.STRIDE_ID].constant(isConstantInput[attrs.STRIDE_ID]);
    }
    if (isAxesSpecified) {
        config.inConfs[attrs.AXES_ID].inPlace(-1);
        config.inConfs[attrs.AXES_ID].constant(isConstantInput[attrs.AXES_ID]);
    }
    if (attrs.isSliceScatterOp) {
        config.inConfs[attrs.UPDATES_ID].inPlace(-1);
        config.inConfs[attrs.UPDATES_ID].constant(isConstantInput[attrs.UPDATES_ID]);
    }
    config.outConfs.resize(1);

    std::vector<LayoutType> supportedTypes;
    if (nDims > 2 && attrs.equalDims) {
        auto canUseBlocked = [&](StridedSliceAttributes& tmpAttrs, const size_t blockSize) {
            if (attrs.isSliceScatterOp) {
                return false;
            }
            if (!isConstantInput[attrs.BEGIN_ID]) {
                return false;
            }
            const auto& srcDims = getInputShapeAtPort(attrs.DATA_ID).getDims();
            if (srcDims[1] == Shape::UNDEFINED_DIM) {
                return false;
            }
            auto channelBeginNormalized =
                tmpAttrs.begin[1] > 0 ? tmpAttrs.begin[1] : tmpAttrs.begin[1] + static_cast<std::int64_t>(srcDims[1]);
            return srcDims[1] % blockSize == 0 && abs(tmpAttrs.stride[1]) == 1 &&
                   (channelBeginNormalized > static_cast<int64_t>(srcDims[1]) ||
                    channelBeginNormalized % blockSize == 0 || channelBeginNormalized < 0 ||
                    tmpAttrs.beginMask[1] == 0);
        };

        supportedTypes.push_back(LayoutType::nspc);

        if (hasConstAttrInputs) {
            auto tmpAttrs = attrs;
            addHiddenDims(tmpAttrs,
                          getInputShapeAtPort(attrs.DATA_ID).getRank(),
                          getOutputShapeAtPort(0).getRank(),
                          isAxesSpecified);
            if (canUseBlocked(tmpAttrs, 8lu)) {
                supportedTypes.push_back(LayoutType::nCsp8c);
            }
            if (canUseBlocked(tmpAttrs, 16lu)) {
                supportedTypes.push_back(LayoutType::nCsp16c);
            }
        }
    }
    supportedTypes.push_back(LayoutType::ncsp);
    auto creators = BlockedDescCreator::getCommonCreators();
    auto range = BlockedDescCreator::makeFilteredRange(creators, nDims, supportedTypes);

    for (auto itr = range.first; itr != range.second; ++itr) {
        config.inConfs[attrs.DATA_ID].setMemDesc(
            itr->second->createSharedDesc(dataPrecision, getInputShapeAtPort(attrs.DATA_ID)));
        config.inConfs[attrs.BEGIN_ID].setMemDesc(
            creators.at(LayoutType::ncsp)->createSharedDesc(iPrecision, getInputShapeAtPort(attrs.BEGIN_ID)));
        config.inConfs[attrs.END_ID].setMemDesc(
            creators.at(LayoutType::ncsp)->createSharedDesc(iPrecision, getInputShapeAtPort(attrs.END_ID)));
        if (isStrideSpecified) {
            config.inConfs[attrs.STRIDE_ID].setMemDesc(
                creators.at(LayoutType::ncsp)->createSharedDesc(iPrecision, getInputShapeAtPort(attrs.STRIDE_ID)));
        }
        if (isAxesSpecified) {
            config.inConfs[attrs.AXES_ID].setMemDesc(
                creators.at(LayoutType::ncsp)->createSharedDesc(iPrecision, getInputShapeAtPort(attrs.AXES_ID)));
        }
        if (attrs.isSliceScatterOp) {
            config.inConfs[attrs.UPDATES_ID].setMemDesc(
                itr->second->createSharedDesc(dataPrecision, getInputShapeAtPort(attrs.UPDATES_ID)));
        }

        config.outConfs[0].setMemDesc(
            itr->second->createSharedDesc(dataPrecision, getOutputShapeAtPort(attrs.DATA_ID)));
        supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::ref);
    }
}

bool StridedSlice::neverExecute() const {
    return getSelectedPrimitiveDescriptor()->hasZeroInputDimsAtPort(0) ||
           getSelectedPrimitiveDescriptor()->hasZeroOutputDimsAtPort(0);
}

bool StridedSlice::isExecutable() const {
    return !isInputTensorAtPortEmpty(0) && !isOutputTensorAtPortEmpty(0);
}

void StridedSlice::createPrimitive() {
    if (inputShapesDefined() && isExecutable() && !shapeHasDataDependency) {
        if (needPrepareParams()) {
            prepareParams();
        }
        updateLastInputDims();
    }
}

bool StridedSlice::needPrepareParams() const {
    return true;
}

void StridedSlice::prepareParams() {
    updateLastInputDims();

    if (srcMemory.empty()) {
        for (size_t i = 0; i < getOriginalInputsNumber(); i++) {
            srcMemory.push_back(getSrcMemoryAtPort(i));
        }
    }
    if (dstMemory.empty()) {
        for (size_t i = 0; i < getOriginalOutputsNumber(); i++) {
            dstMemory.push_back(getDstMemoryAtPort(i));
        }
    }
    execPtr = std::make_shared<StridedSliceCommonExecutor>(attrs, srcMemory, dstMemory);
}

bool StridedSlice::needShapeInfer() const {
    return Node::inputShapesModified() || shapeHasDataDependency;
}

void StridedSlice::execute(const dnnl::stream& strm) {
    if (!execPtr) {
        THROW_CPU_NODE_ERR("doesn't have compiled executor!");
    }

    execPtr->exec(srcMemory, dstMemory);
}

void StridedSlice::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

bool StridedSlice::created() const {
    return getType() == Type::StridedSlice;
}

StridedSlice::StridedSliceCommonExecutor::StridedSliceCommonExecutor(const StridedSliceAttributes& attrs,
                                                                     const std::vector<MemoryCPtr>& srcMemory,
                                                                     const std::vector<MemoryCPtr>& dstMemory)
    : StridedSliceExecutor(attrs, srcMemory, dstMemory) {
    paramsInitialization(attrs, srcMemory, dstMemory);
    dimsNormalization();
    dimsGluing();
    indicesCalculation();
}

void StridedSlice::StridedSliceCommonExecutor::orderParametersByLayouts(
    const BlockedMemoryDescCPtr& blockedMemoryDesc) {
    size_t blk = 1;
    bool isBlockedLayout = false;
    if (blockedMemoryDesc->hasLayoutType(LayoutType::nCsp16c)) {
        isBlockedLayout = true;
        blk = 16;
    } else if (blockedMemoryDesc->hasLayoutType(LayoutType::nCsp8c)) {
        isBlockedLayout = true;
        blk = 8;
    }
    const bool isPerChannelLayout = blockedMemoryDesc->hasLayoutType(LayoutType::nspc);
    auto srcOrder = blockedMemoryDesc->getOrder();

    if (isBlockedLayout) {
        params.attrs.begin[1] = params.attrs.begin[1] / blk;
        params.attrs.end[1] = std::ceil(params.attrs.end[1] / static_cast<float>(blk));
        params.attrs.begin.push_back(0);
        params.attrs.end.push_back(0);
        params.attrs.stride.push_back(1);
        params.attrs.beginMask.push_back(0);
        params.attrs.endMask.push_back(0);
        params.attrs.ellipsisMask.push_back(0);
        params.attrs.newAxisMask.push_back(0);
        params.attrs.shrinkAxisMask.push_back(0);
    } else if (isPerChannelLayout) {
        auto sortByOrder = [&](std::vector<int>& data) {
            std::vector<int> temp(srcOrder.size());
            for (size_t i = 0; i < srcOrder.size(); i++) {
                temp[i] = data[srcOrder[i]];
            }
            data = temp;
        };

        sortByOrder(params.attrs.begin);
        sortByOrder(params.attrs.end);
        sortByOrder(params.attrs.stride);
        sortByOrder(params.attrs.beginMask);
        sortByOrder(params.attrs.endMask);
        if (params.attrs.isStridedSliceOp) {
            sortByOrder(params.attrs.ellipsisMask);
            sortByOrder(params.attrs.newAxisMask);
            sortByOrder(params.attrs.shrinkAxisMask);
        }
    }
}

void StridedSlice::StridedSliceCommonExecutor::paramsInitialization(const StridedSliceAttributes& attrs,
                                                                    const std::vector<MemoryCPtr>& srcMemory,
                                                                    const std::vector<MemoryCPtr>& dstMemory) {
    const auto srcBlockedMemoryDesc = srcMemory[0]->getDescWithType<BlockedMemoryDesc>();
    const auto dstBlockedMemoryDesc = dstMemory[0]->getDescWithType<BlockedMemoryDesc>();

    params.attrs = attrs;
    params.srcBlockedDims = srcBlockedMemoryDesc->getBlockDims();
    params.srcOrder = srcBlockedMemoryDesc->getOrder();
    params.dstBlockedDims = dstBlockedMemoryDesc->getBlockDims();

    const size_t inputRank = srcMemory[0]->getShape().getRank();
    const size_t outputRank = dstMemory[0]->getShape().getRank();
    const size_t nDims = std::max(inputRank, outputRank);

    auto fillingInParameters = [&](std::vector<int>& parameter, const size_t type, const size_t size, const int value) {
        const auto* ptr = srcMemory[type]->getDataAs<const int32_t>();
        parameter.assign(ptr, ptr + size);

        if (type != attrs.AXES_ID && params.attrs.ellipsisMaskCounter == 0 && size < nDims) {
            for (size_t i = size; i < nDims; i++) {
                parameter.push_back(value);
            }
        }
    };

    params.attrs.beginDims = srcMemory[attrs.BEGIN_ID]->getShape().getStaticDims();
    params.attrs.endDims = srcMemory[attrs.END_ID]->getShape().getStaticDims();
    if (params.attrs.beginDims.size() != 1) {
        OPENVINO_THROW("Strided slice common executor should have begin vector with 1 dimension");
    }
    if (params.attrs.endDims.size() != 1) {
        OPENVINO_THROW("Strided slice common executor should have end vector with 1 dimension");
    }
    if (params.attrs.beginDims[0] != params.attrs.endDims[0]) {
        OPENVINO_THROW("Strided slice common executor should have begin vector with size equal to end vector size");
    }

    if (params.attrs.begin.empty()) {
        fillingInParameters(params.attrs.begin, attrs.BEGIN_ID, params.attrs.beginDims[0], 0);
    }
    if (params.attrs.end.empty()) {
        fillingInParameters(params.attrs.end, attrs.END_ID, params.attrs.endDims[0], 0);
    }

    if (srcMemory.size() > attrs.STRIDE_ID) {
        params.attrs.strideDims = srcMemory[attrs.STRIDE_ID]->getShape().getStaticDims();
        if (params.attrs.strideDims.size() > 1) {
            OPENVINO_THROW("Strided slice common executor should have stride vector with 1 dimension");
        }
        if (params.attrs.beginDims[0] != params.attrs.strideDims[0]) {
            OPENVINO_THROW(
                "Strided slice common executor should have stride vector with size equal to begin vector size");
        }

        if (params.attrs.stride.empty()) {
            fillingInParameters(params.attrs.stride, attrs.STRIDE_ID, params.attrs.strideDims[0], 1);
        }
    }

    if (srcMemory.size() > attrs.AXES_ID) {
        params.attrs.axesDims = srcMemory[attrs.AXES_ID]->getShape().getStaticDims();
        if (params.attrs.axesDims.size() != 1) {
            OPENVINO_THROW("Strided slice common executor should have axes vector with 1 dimension.");
        }
        if (params.attrs.beginDims[0] != params.attrs.axesDims[0]) {
            OPENVINO_THROW(
                "Strided slice common executor should have axes vector with size equal to begin vector size.");
        }

        if (params.attrs.axes.empty()) {
            fillingInParameters(params.attrs.axes, attrs.AXES_ID, params.attrs.axesDims[0], 0);
        }
    }

    addHiddenDims(params.attrs, inputRank, outputRank, srcMemory.size() > attrs.AXES_ID);

    if (!srcBlockedMemoryDesc->hasLayoutType(LayoutType::ncsp)) {
        orderParametersByLayouts(srcBlockedMemoryDesc);
    }
}

void StridedSlice::StridedSliceCommonExecutor::dimsNormalization() {
    // creating new src and dst dimensions and parameters of the same size using masks
    //
    // example 1: before srcDims = [5, 6, 8, 3, 2], begin = [1, 0], end = [4, 0], stride = [1, 1]
    //            beginMask = [0, 1], endMask = [0, 1], ellipsisMask = [1, 0], newAxisMas = [0, 0], shrinkAxisMask = [0,
    //            0] after srcDims = [5, 6, 8, 3, 2], begin = [1, 0, 0, 0, 0], end = [4, 5, 7, 2, 1], stride = [1, 1, 1,
    //            1, 1], dstDims = [4, 6, 8, 3, 2]
    //
    // example 2: before srcDims = [5, 6, 8, 3, 2], begin = [0, 3, 0, 0, 0], end = [0, 3, 0, 0, 0], stride = [1, 1, 1,
    // 1, 1]
    //            beginMask = [1, 0, 1, 1, 1], endMask = [1, 0, 1, 1, 1], ellipsisMask = [0, 0, 0, 0, 0], newAxisMask =
    //            [0, 0, 0, 0, 0], shrinkAxisMask = [0, 1, 0, 0, 0] after srcDims = [5, 6, 8, 3, 2], begin = [0, 3, 0,
    //            0, 0], end = [4, 3, 7, 2, 1], stride = [1, 1, 1, 1, 1], dstDims = [5, 1, 8, 3, 2]
    //
    // example 3: before srcDims = [5, 8, 3, 2], begin = [0, 0, 0, 0], end = [0, 0, 0, 0], stride = [1, 1, 1, 1]
    //            beginMask = [1, 0, 1, 1, 1], endMask = [1, 0, 1, 1, 1], ellipsisMask = [0, 0, 0, 0, 0], newAxisMask =
    //            [0, 1, 0, 0, 0], shrinkAxisMask = [0, 0, 0, 0, 0] after srcDims = [5, 1, 8, 3, 2], begin = [0, 0, 0,
    //            0, 0], end = [4, 0, 7, 2, 1], stride = [1, 1, 1, 1, 1], dstDims = [5, 1, 8, 3, 2]

    auto clipping = [](int& idx, const int min, const int max) {
        idx = (idx > min) ? idx : min;
        idx = (idx < max) ? idx : (max - 1);
    };

    auto correcting = [](int& dim, const size_t shift) {
        dim = dim >= 0 ? dim : shift + dim;
    };

    VectorDims newSrcDims, newDstDims;
    std::vector<int> beginTemp;
    std::vector<int> endTemp;
    std::vector<int> strideTemp;
    size_t srcIdx = 0;
    for (size_t axis = 0; axis < params.attrs.begin.size(); ++axis) {
        if (params.attrs.ellipsisMask[axis] == 1) {
            int nNewAxisAfterEllipses = 0;
            int nSrcAxisBeforeEllipses = 0;
            for (size_t i = 0; i < axis; ++i) {
                if (params.attrs.newAxisMask[i] != 1) {
                    nSrcAxisBeforeEllipses++;
                }
            }
            for (size_t i = axis + 1; i < params.attrs.begin.size(); ++i) {
                if (params.attrs.newAxisMask[i] == 1) {
                    nNewAxisAfterEllipses++;
                }
            }

            size_t nSrcAxisAfterEllipses = (params.attrs.begin.size() - axis - nNewAxisAfterEllipses - 1);
            size_t nHiddenDims = params.srcBlockedDims.size() - nSrcAxisAfterEllipses - nSrcAxisBeforeEllipses;
            for (size_t i = 0; i < nHiddenDims; ++i) {
                newSrcDims.push_back(params.srcBlockedDims[srcIdx]);
                newDstDims.push_back(params.srcBlockedDims[srcIdx]);
                beginTemp.push_back(0);
                endTemp.push_back(params.srcBlockedDims[srcIdx] - 1);
                strideTemp.push_back(1);

                srcIdx++;
            }
        } else {
            if (params.attrs.newAxisMask[axis] == 1) {
                beginTemp.push_back(0);
                endTemp.push_back(0);
                strideTemp.push_back(1);
                newSrcDims.push_back(1);
                newDstDims.push_back(1);
            } else if (params.attrs.shrinkAxisMask[axis] == 1) {
                int b = params.attrs.beginMask[axis] == 1 ? params.attrs.begin[axis] : 0;
                correcting(b, params.srcBlockedDims[srcIdx]);
                clipping(b, 0, params.srcBlockedDims[srcIdx]);
                beginTemp.push_back(b);
                endTemp.push_back(b);
                strideTemp.push_back(1);
                newSrcDims.push_back(params.srcBlockedDims[srcIdx]);
                newDstDims.push_back(1);

                srcIdx++;
            } else {
                int b = params.attrs.beginMask[axis] == 1 ? params.attrs.begin[axis]
                                                          : (params.attrs.stride[axis] > 0 ? 0 : -1);
                correcting(b, params.srcBlockedDims[srcIdx]);
                clipping(b, 0, params.srcBlockedDims[srcIdx]);

                int e = params.attrs.endMask[axis] == 1
                            ? (params.attrs.stride[axis] > 0 ? params.attrs.end[axis] - 1 : params.attrs.end[axis] + 1)
                            : (params.attrs.stride[axis] > 0 ? -1 : 0);
                correcting(e, params.srcBlockedDims[srcIdx]);
                clipping(e, 0, params.srcBlockedDims[srcIdx]);

                beginTemp.push_back(b);
                endTemp.push_back(e);
                strideTemp.push_back(params.attrs.stride[axis]);
                newSrcDims.push_back(params.srcBlockedDims[srcIdx]);
                newDstDims.push_back(
                    std::ceil(static_cast<float>(abs(e - b) + 1) / static_cast<float>(abs(strideTemp.back()))));

                srcIdx++;
            }
        }
    }

    params.attrs.begin = beginTemp;
    params.attrs.end = endTemp;
    params.attrs.stride = strideTemp;

    params.dstBlockedDims = newDstDims;
    params.srcBlockedDims = newSrcDims;
    params.dstStrides.resize(newDstDims.size());
    params.srcStrides.resize(newSrcDims.size());
    params.dstStrides[params.dstStrides.size() - 1] = params.srcStrides[params.srcStrides.size() - 1] = 1;
    for (int i = newDstDims.size() - 2; i >= 0; --i) {
        params.dstStrides[i] = params.dstStrides[i + 1] * params.dstBlockedDims[i + 1];
        params.srcStrides[i] = params.srcStrides[i + 1] * params.srcBlockedDims[i + 1];
    }
}

void StridedSlice::StridedSliceCommonExecutor::dimsGluing() {
    // gluing of dimensions if there aren't begin, end and stride != 1 on this axis
    // example: before gluing srcDims = [5, 6, 8, 3, 2], begin = [1, 0, 0, 0, 0], stride = [1, 1, 2, 1, 1], dstDims =
    // [4, 6, 4, 3, 2]
    //          after gluing  srcDims = [30, 8, 6],      begin = [6, 0, 0],       stride = [1, 2, 1],       dstDims =
    //          [24, 4, 6]

    size_t realNDims = params.dstBlockedDims.size();

    std::pair<size_t, size_t> secondDim = {0, params.attrs.begin.size()};
    VectorDims indexes(1, 0);
    for (size_t idx = 0; idx < params.attrs.begin.size(); idx++) {
        if (params.attrs.begin[idx] != 0 ||
            static_cast<size_t>(params.attrs.end[idx]) != params.srcBlockedDims[idx] - 1 ||
            params.attrs.stride[idx] != 1) {
            indexes.push_back(0u == idx ? 0 : idx - 1);
            indexes.push_back(params.attrs.stride[idx] == 1 ? idx : idx + 1);

            if (idx != 0 && secondDim.first == 0) {
                secondDim.first = idx;
            } else if (idx != 0 && secondDim.second == params.attrs.begin.size()) {
                secondDim.second = idx;
            }
        }
    }

    if (indexes.back() < 2) {
        indexes[indexes.size() - 1] = 1;
        secondDim.first = 1;
    }

    const VectorDims srcBlockedDimsBefore = params.srcBlockedDims;
    const VectorDims dstBlockedDimsBefore = params.dstBlockedDims;
    const size_t nGluingLastDims = params.dstStrides[std::max(static_cast<int>(indexes.back() - 1), 0)];
    const bool vLastDim = indexes.back() < params.attrs.begin.size();
    indexes[indexes.size() - 1] = vLastDim ? indexes.back() : params.attrs.begin.size() - 1;
    indexes.push_back(params.attrs.begin.size() - 1);

    for (int idx = indexes.size() - 1; idx >= 0; idx -= 2) {
        if (indexes[idx - 1] < indexes[idx]) {
            for (size_t jdx = indexes[idx]; jdx > indexes[idx - 1]; --jdx) {
                params.dstBlockedDims[indexes[idx - 1]] *= params.dstBlockedDims[jdx];
                params.srcBlockedDims[indexes[idx - 1]] *= params.srcBlockedDims[jdx];
                params.dstStrides[indexes[idx - 1]] /= params.dstBlockedDims[jdx];
                params.srcStrides[indexes[idx - 1]] /= params.srcBlockedDims[jdx];

                params.attrs.begin[indexes[idx - 1]] *= params.dstBlockedDims[jdx];
            }
            const size_t beginShift = indexes[idx - 1] + 1;
            const size_t endShift = indexes[idx] + 1;

            params.dstBlockedDims.erase(params.dstBlockedDims.begin() + beginShift,
                                        params.dstBlockedDims.begin() + endShift);
            params.srcBlockedDims.erase(params.srcBlockedDims.begin() + beginShift,
                                        params.srcBlockedDims.begin() + endShift);
            params.dstStrides.erase(params.dstStrides.begin() + beginShift, params.dstStrides.begin() + endShift);
            params.srcStrides.erase(params.srcStrides.begin() + beginShift, params.srcStrides.begin() + endShift);

            params.attrs.begin.erase(params.attrs.begin.begin() + beginShift, params.attrs.begin.begin() + endShift);
            params.attrs.stride.erase(params.attrs.stride.begin() + beginShift, params.attrs.stride.begin() + endShift);
        }
    }

    workAmount = params.dstBlockedDims[0] * params.dstStrides[0] / nGluingLastDims;
    lastDstDim = nGluingLastDims * params.attrs.dataSize;
    params.nDimsForWork = params.dstBlockedDims.size() - static_cast<size_t>(vLastDim);

    if (params.nDimsForWork == 1 && realNDims > 2) {
        const size_t realSrcDim = srcBlockedDimsBefore[secondDim.first];
        const size_t realDstDim = dstBlockedDimsBefore[secondDim.first];

        params.dstStrides.insert(params.dstStrides.begin() + 1, params.dstStrides[0] / realDstDim);
        params.srcStrides.insert(params.srcStrides.begin() + 1, params.srcStrides[0] / realSrcDim);

        for (size_t idx = secondDim.first + 1; idx < secondDim.second; idx++) {
            params.attrs.begin[1] /= dstBlockedDimsBefore[idx];
        }

        if (params.dstBlockedDims[0] < m_threads_num) {
            params.dstBlockedDims[1] /= realDstDim;
            params.srcBlockedDims[1] /= realSrcDim;
            params.dstBlockedDims.insert(params.dstBlockedDims.begin() + 1, realDstDim);
            params.srcBlockedDims.insert(params.srcBlockedDims.begin() + 1, realSrcDim);
        }

        if (params.dstBlockedDims.size() > 2) {
            lastDstDim /= dstBlockedDimsBefore[secondDim.first];
        }
    }

    // some parameter calculations for common execution
    params.isOptimized = params.nDimsForWork == 1 && params.dstBlockedDims.size() > 1;
    if (params.isOptimized) {
        if (params.dstBlockedDims.size() == 2) {
            params.dstBlockedDims[1] = 1;
        }

        workAmount = params.dstBlockedDims[0] * params.dstBlockedDims[1];
        srcShift = (params.attrs.begin[0] * params.srcStrides[0] + params.attrs.begin[1] * params.srcStrides[1]) *
                   params.attrs.dataSize;
    } else {
        srcShift = params.attrs.stride.back() == 1 && params.attrs.stride.size() > 1
                       ? params.attrs.begin[params.nDimsForWork] * params.srcStrides[params.nDimsForWork] *
                             params.attrs.dataSize
                       : 0;
    }
}

static inline size_t parallel_init(size_t start, size_t nDims, const VectorDims& dims, VectorDims& indexes) {
    for (int j = nDims - 1; j >= 0; j--) {
        indexes[j] = start % dims[j];
        start = start / dims[j];
    }
    return start;
}

void StridedSlice::StridedSliceCommonExecutor::indicesCalculation() {
    // indices calculation before execution for the best performance
    srcIndices.resize(workAmount, 0);
    dstIndices.resize(workAmount, 0);

    // should choose more optimal thread count
    nThreads = m_threads_num > workAmount ? workAmount : m_threads_num;

    if (params.isOptimized) {
        indicesCalculationForOptimized();
        return;
    }

    auto getSrcIdx = [&](const VectorDims& indexes) {
        size_t srcIdx = 0;
        for (size_t i = 0; i < params.nDimsForWork; ++i) {
            srcIdx += (params.attrs.begin[i] + indexes[i] * params.attrs.stride[i]) * params.srcStrides[i];
        }
        return srcIdx * params.attrs.dataSize;
    };

    parallel_nt(nThreads, [&](const int ithr, const int nthr) {
        size_t start = 0, end = 0;
        VectorDims coords(params.nDimsForWork, 0);
        splitter(workAmount, nthr, ithr, start, end);
        parallel_init(start, params.nDimsForWork, params.dstBlockedDims, coords);

        size_t srcIdx = getSrcIdx(coords);
        for (size_t j = start; j < end; ++j) {
            dstIndices[j] = j * lastDstDim;
            srcIndices[j] = srcIdx;

            bool out = false;
            for (int k = params.nDimsForWork - 1; k >= 0; k--) {
                coords[k]++;
                if (coords[k] < params.dstBlockedDims[k]) {
                    srcIdx += params.attrs.stride[k] * params.srcStrides[k] * params.attrs.dataSize;
                    break;
                }

                coords[k] = 0;
                out = true;
            }

            if (out) {
                srcIdx = getSrcIdx(coords);
            }
        }
    });
}

void StridedSlice::StridedSliceCommonExecutor::indicesCalculationForOptimized() {
    const size_t dstIdx0 = params.dstStrides[0] * params.attrs.dataSize;
    const size_t dstIdx1 = params.dstStrides[1] * params.attrs.dataSize;
    const size_t srcIdx0 = params.attrs.stride[0] * params.srcStrides[0] * params.attrs.dataSize;
    const size_t srcIdx1 = params.attrs.stride[1] * params.srcStrides[1] * params.attrs.dataSize;

    for (size_t i0 = 0; i0 < params.dstBlockedDims[0]; i0++) {
        const size_t idx = i0 * params.dstBlockedDims[1];

        dstIndices[idx] = i0 * dstIdx0;
        srcIndices[idx] = i0 * srcIdx0;

        for (size_t i1 = 1; i1 < params.dstBlockedDims[1]; i1++) {
            dstIndices[idx + i1] = dstIndices[idx] + i1 * dstIdx1;
            srcIndices[idx + i1] = srcIndices[idx] + i1 * srcIdx1;
        }
    }
}

void StridedSlice::StridedSliceCommonExecutor::execStridedSlice(const std::vector<MemoryCPtr>& srcMemory,
                                                                const std::vector<MemoryCPtr>& dstMemory) {
    const auto* srcData = srcMemory[0]->getDataAs<const uint8_t>();
    auto* dstData = dstMemory[0]->getDataAs<uint8_t>();
    const uint8_t* srcShiftedData = srcData + srcShift;
    parallel_nt(nThreads, [&](const int ithr, const int nthr) {
        size_t start = 0, end = 0;
        splitter(workAmount, nthr, ithr, start, end);

        for (size_t iwork = start; iwork < end; ++iwork) {
            cpu_memcpy(&dstData[dstIndices[iwork]], &srcShiftedData[srcIndices[iwork]], lastDstDim);
        }
    });
}

void StridedSlice::StridedSliceCommonExecutor::execSliceScatter(const std::vector<MemoryCPtr>& srcMemory,
                                                                const std::vector<MemoryCPtr>& dstMemory) {
    const auto* srcData = srcMemory[0]->getDataAs<const uint8_t>();
    const auto* srcUpdates = srcMemory[1]->getDataAs<const uint8_t>();
    auto* dstData = dstMemory[0]->getDataAs<uint8_t>();
    cpu_parallel_memcpy(dstData, srcData, srcMemory[0]->getSize());
    if (srcMemory[1]->getSize() == 0) {
        // Updates are empty - do not apply
        return;
    }
    uint8_t* dstShiftedData = dstData + srcShift;
    parallel_nt(nThreads, [&](const int ithr, const int nthr) {
        size_t start = 0, end = 0;
        splitter(workAmount, nthr, ithr, start, end);

        for (size_t iwork = start; iwork < end; ++iwork) {
            cpu_memcpy(&dstShiftedData[srcIndices[iwork]], &srcUpdates[dstIndices[iwork]], lastDstDim);
        }
    });
}

void StridedSlice::StridedSliceCommonExecutor::exec(const std::vector<MemoryCPtr>& srcMemory,
                                                    const std::vector<MemoryCPtr>& dstMemory) {
    if (params.attrs.isSliceScatterOp) {
        execSliceScatter(srcMemory, dstMemory);
    } else {
        execStridedSlice(srcMemory, dstMemory);
    }
}

}  // namespace ov::intel_cpu::node
