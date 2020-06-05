// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_scatter_update_node.h"
#include "desc_iterator.hpp"
#include "mkldnn_quantize_node.h"
#include "mkldnn_depthwise_node.h"
#include "mkldnn_activation_node.h"
#include <ie_layers.h>
#include <mkldnn.hpp>
#include <string>
#include <vector>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include <ie_layers_internal.hpp>
#include "ie_parallel.hpp"
#include <algorithm>

#include "jit_generator.hpp"
#include "jit_uni_eltwise.hpp"
#include "jit_uni_depthwise.hpp"
#include "jit_uni_quantization.hpp"
#include "common/simple_copy.h"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace mkldnn::impl;
using namespace mkldnn::impl::cpu;
using namespace mkldnn::impl::utils;
using namespace Xbyak;


MKLDNNScatterUpdateNode::MKLDNNScatterUpdateNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(layer, eng, cache) {}

void MKLDNNScatterUpdateNode::getSupportedDescriptors() {
    if (!descs.empty())
        return;

    if ((getParentEdges().size() != 3) && (getParentEdges().size() != 4))
        THROW_IE_EXCEPTION << "Incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        THROW_IE_EXCEPTION << "Incorrect number of output edges for layer " << getName();

    auto indicesPre = getCnnLayer()->insData[INDICES_ID].lock()->getPrecision();
    if (indicesPre != Precision::I32 && indicesPre != Precision::I64) {
        THROW_IE_EXCEPTION << "Incorrect indices precision for layer " << getName() << ". Only I32 or I64 is supported.";
    }

    Type scatterUpdateType = getType();
    if (scatterUpdateType == ScatterUpdate) scatterUpdateMode = ScatterUpdateMode::ScatterUpdate;
    else if (scatterUpdateType == ScatterElementsUpdate) scatterUpdateMode = ScatterUpdateMode::ScatterElementsUpdate;
    else if (scatterUpdateType == ScatterNDUpdate) scatterUpdateMode = ScatterUpdateMode::ScatterNDUpdate;
    else
        THROW_IE_EXCEPTION << " Unsupported ScatterUpdate type for layer " << getName();

    if (scatterUpdateMode != ScatterUpdateMode::ScatterNDUpdate) {
        auto *layer = getCnnLayer().get();
        axis = layer->GetParamAsInt("axis");
    }
}

void MKLDNNScatterUpdateNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    auto isOneOf = [&](InferenceEngine::Precision precision, std::vector<InferenceEngine::Precision> precisions) {
        for (auto p : precisions) {
            if (p == precision) {
                return true;
            }
        }
        return false;
    };

    Precision indicesPrecision = getCnnLayer()->insData[INDICES_ID].lock()->getPrecision();
    if (!isOneOf(indicesPrecision, {Precision::I32, Precision::I64})) {
        THROW_IE_EXCEPTION << "Unsupported indices precision for layer " << getName() << ". Only I32 and I64 are supported.";
    }

    Precision inputPrecision = getCnnLayer()->insData[DATA_ID].lock()->getPrecision();
    Precision updatePrecision = getCnnLayer()->insData[UPDATE_ID].lock()->getPrecision();
    Precision outputPrecision = getCnnLayer()->outData[0]->getPrecision();
    if ((inputPrecision != updatePrecision || inputPrecision != outputPrecision)) {
        THROW_IE_EXCEPTION << "Input precision, output precision and update precision are not the same for layer " << getName();
    }
    if (!isOneOf(inputPrecision, {Precision::FP32, Precision::FP16, Precision::BF16, Precision::I64, Precision::U64, Precision::I32,
        Precision::I16, Precision::U16, Precision::I8, Precision::U8})) {
        THROW_IE_EXCEPTION << "Unsupported input data precision for layer " << getName();
    }

    auto inputType = MKLDNNExtensionUtils::IEPrecisionToDataType(inputPrecision);
    auto indicesType = MKLDNNExtensionUtils::IEPrecisionToDataType(indicesPrecision);
    auto outputType = MKLDNNExtensionUtils::IEPrecisionToDataType(outputPrecision);

    inputPrec = inputPrecision;
    indicesPrec = indicesPrecision;
    outputPrec = outputPrecision;
    inputSize = MKLDNNExtensionUtils::sizeOfDataType(inputType);
    indicesSize = MKLDNNExtensionUtils::sizeOfDataType(indicesType);
    outputSize = MKLDNNExtensionUtils::sizeOfDataType(outputType);

    InferenceEngine::LayerConfig config;
    config.dynBatchSupport = false;
    config.inConfs.resize(3);
    config.outConfs.resize(1);
    config.inConfs[DATA_ID].constant = false;
    config.inConfs[INDICES_ID].constant = false;
    config.inConfs[UPDATE_ID].constant = false;
    config.outConfs[0].constant = false;
    config.inConfs[DATA_ID].inPlace = -1;
    config.inConfs[INDICES_ID].inPlace = -1;
    config.inConfs[UPDATE_ID].inPlace = -1;
    config.outConfs[0].inPlace = -1;

    // indices and update format can be different with input?
    auto pushDesc = [&](memory::format inFormat, memory::format outFormat) {
        config.inConfs[DATA_ID].desc = MKLDNNMemoryDesc(getParentEdgeAt(DATA_ID)->getDims(), inputType, inFormat);
        config.inConfs[INDICES_ID].desc = MKLDNNMemoryDesc(getParentEdgeAt(INDICES_ID)->getDims(), indicesType, inFormat);
        config.inConfs[UPDATE_ID].desc = MKLDNNMemoryDesc(getParentEdgeAt(UPDATE_ID)->getDims(), inputType, inFormat);
        config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputType, outFormat);
        supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown, outFormat});
    };

    pushDesc(MKLDNNMemory::GetPlainFormat(memory::dims(getParentEdgeAt(DATA_ID)->getDims().ndims())),
             MKLDNNMemory::GetPlainFormat(memory::dims(getChildEdgeAt(0)->getDims().ndims())));
}

void MKLDNNScatterUpdateNode::createPrimitive() {
    auto &dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto &srcMemPtr = getParentEdgeAt(DATA_ID)->getMemoryPtr();
    auto &indicesMemPtr = getParentEdgeAt(INDICES_ID)->getMemoryPtr();
    auto &updateMemPtr = getParentEdgeAt(UPDATE_ID)->getMemoryPtr();

    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Destination memory didn't allocate.";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Input memory didn't allocate.";
    if (!indicesMemPtr || !indicesMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Indices memory didn't allocate.";
    if (!updateMemPtr || !updateMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Update memory didn't allocate.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor is not set.";
}

void MKLDNNScatterUpdateNode::execute(mkldnn::stream strm) {
    auto &dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto &srcMemPtr = getParentEdgeAt(DATA_ID)->getMemoryPtr();
    auto &indicesMemPtr = getParentEdgeAt(INDICES_ID)->getMemoryPtr();
    auto &updateMemPtr = getParentEdgeAt(UPDATE_ID)->getMemoryPtr();

    uint8_t *dstPtr = reinterpret_cast<uint8_t*>(dstMemPtr->GetData()) +
            dstMemPtr->GetDescriptor().data.layout_desc.blocking.offset_padding * outputSize;
    uint8_t *srcPtr = reinterpret_cast<uint8_t*>(srcMemPtr->GetData()) +
            srcMemPtr->GetDescriptor().data.layout_desc.blocking.offset_padding * inputSize;
    uint8_t *indicesPtr = reinterpret_cast<uint8_t*>(indicesMemPtr->GetData()) +
            indicesMemPtr->GetDescriptor().data.layout_desc.blocking.offset_padding * indicesSize;
    uint8_t *updatePtr = reinterpret_cast<uint8_t*>(updateMemPtr->GetData()) +
            updateMemPtr->GetDescriptor().data.layout_desc.blocking.offset_padding * inputSize;

    if (inputSize == 1) {
        auto dstData = reinterpret_cast<PrecisionTrait<Precision::U8>::value_type *>(dstPtr);
        auto srcData = reinterpret_cast<PrecisionTrait<Precision::U8>::value_type *>(srcPtr);
        auto update = reinterpret_cast<PrecisionTrait<Precision::U8>::value_type *>(updatePtr);
        if (indicesSize == 4) {
            auto indices = reinterpret_cast<int32_t *>(indicesPtr);
            scatterUpdate<PrecisionTrait<Precision::U8>::value_type, int32_t>(srcData, indices, update, axis, dstData, scatterUpdateMode);
        } else if (indicesSize == 8) {
            auto indices = reinterpret_cast<int64_t *>(indicesPtr);
            scatterUpdate<PrecisionTrait<Precision::U8>::value_type, int64_t>(srcData, indices, update, axis, dstData, scatterUpdateMode);
        } else {
            THROW_IE_EXCEPTION << "Unsupported indices precision for layer " << getName() << ". Only I32 and I64 are supported.";
        }
    } else if (inputSize == 2) {
        auto dstData = reinterpret_cast<PrecisionTrait<Precision::U16>::value_type *>(dstPtr);
        auto srcData = reinterpret_cast<PrecisionTrait<Precision::U16>::value_type *>(srcPtr);
        auto update = reinterpret_cast<PrecisionTrait<Precision::U16>::value_type *>(updatePtr);
        if (indicesSize == 4) {
            auto indices = reinterpret_cast<int32_t *>(indicesPtr);
            scatterUpdate<PrecisionTrait<Precision::U16>::value_type, int32_t>(srcData, indices, update, axis, dstData, scatterUpdateMode);
        } else if (indicesSize == 8) {
            auto indices = reinterpret_cast<int64_t *>(indicesPtr);
            scatterUpdate<PrecisionTrait<Precision::U16>::value_type, int64_t>(srcData, indices, update, axis, dstData, scatterUpdateMode);
        } else {
            THROW_IE_EXCEPTION << "Unsupported indices precision for layer " << getName() << ". Only I32 and I64 are supported.";
        }
    } else if (inputSize == 4) {
        auto dstData = reinterpret_cast<PrecisionTrait<Precision::I32>::value_type *>(dstPtr);
        auto srcData = reinterpret_cast<PrecisionTrait<Precision::I32>::value_type *>(srcPtr);
        auto update = reinterpret_cast<PrecisionTrait<Precision::I32>::value_type *>(updatePtr);
        if (indicesSize == 4) {
            auto indices = reinterpret_cast<int32_t *>(indicesPtr);
            scatterUpdate<PrecisionTrait<Precision::I32>::value_type, int32_t>(srcData, indices, update, axis, dstData, scatterUpdateMode);
        } else if (indicesSize == 8) {
            auto indices = reinterpret_cast<int64_t *>(indicesPtr);
            scatterUpdate<PrecisionTrait<Precision::I32>::value_type, int64_t>(srcData, indices, update, axis, dstData, scatterUpdateMode);
        } else {
            THROW_IE_EXCEPTION << "Unsupported indices precision for layer " << getName() << ". Only I32 and I64 are supported.";
        }
    } else if (inputSize == 8) {
        auto dstData = reinterpret_cast<PrecisionTrait<Precision::U64>::value_type *>(dstPtr);
        auto srcData = reinterpret_cast<PrecisionTrait<Precision::U64>::value_type *>(srcPtr);
        auto update = reinterpret_cast<PrecisionTrait<Precision::U64>::value_type *>(updatePtr);
        if (indicesSize == 4) {
            auto indices = reinterpret_cast<int32_t *>(indicesPtr);
            scatterUpdate<PrecisionTrait<Precision::U64>::value_type, int32_t>(srcData, indices, update, axis, dstData, scatterUpdateMode);
        } else if (indicesSize == 8) {
            auto indices = reinterpret_cast<int64_t *>(indicesPtr);
            scatterUpdate<PrecisionTrait<Precision::U64>::value_type, int64_t>(srcData, indices, update, axis, dstData, scatterUpdateMode);
        } else {
            THROW_IE_EXCEPTION << "Unsupported indices precision for layer " << getName() << ". Only I32 and I64 are supported.";
        }
    } else {
        THROW_IE_EXCEPTION << "Unsupported input data precision for layer " << getName();
    }
}

// shape5D: n    c   d    h    w
// block5D: cdhw dhw hw   w    1
// index  : 0    1   2    3    4
std::vector<size_t> getShape5D(const SizeVector& shape) {
    std::vector<size_t> shape5D(5, 1);
    for (int i = 0; i < shape.size(); i++) {
        shape5D[i] = shape[i];
    }
    return shape5D;
}

std::vector<size_t> getBlock5D(const std::vector<size_t>& shape5d) {
    std::vector<size_t> block5D(5, 1);
    for (int i = 3; i < 0; i--) {
        block5D[i] *= shape5d[i+1] * block5D[i+1];
    }
    return block5D;
}

// diff granularity level
template <typename data_t, typename index_t>
void MKLDNNScatterUpdateNode::scatterUpdate(data_t *srcData, index_t *indices, data_t *update, int axis, data_t *dstData, ScatterUpdateMode mode) {
    SizeVector srcDataDim = getParentEdgeAt(0)->getDesc().getDims();
    SizeVector indicesDim = getParentEdgeAt(1)->getDesc().getDims();
    SizeVector updateDim = getParentEdgeAt(2)->getDesc().getDims();
    SizeVector dstDataDim = getChildEdgeAt(0)->getDesc().getDims();

    size_t srcRank = srcDataDim.size();
    size_t dstRank = dstDataDim.size();
    size_t indicesRank = indicesDim.size();
    size_t updateRank = updateDim.size();

    // common check
    if (srcRank != dstRank) {
        THROW_IE_EXCEPTION << "Rank of output tensor should be the same as input tensor for layer " << getName();
    } else {
        for (size_t r = 0; r < srcRank; r++) {
            if (srcDataDim[r] != dstDataDim[r]) {
                THROW_IE_EXCEPTION << "Shape of output tensor should be the same as input tensor for layer " << getName()
                                   << ". The input shape is " << srcDataDim[r] << ", while output shape is " << dstDataDim[r]
                                   << "for" << r << "th dimension";
            }
        }
    }
    if (mode != ScatterUpdateMode::ScatterNDUpdate) {
        // Axis can be in range [-r, r - 1] where r is the rank of input data
        if (axis >= srcRank || axis < (srcRank * -1)) {
            THROW_IE_EXCEPTION << "Value of axis can be in range [-r, r - 1] where r is the rank of input data for layer " << getName();
        }
        axis = axis < 0 ? (axis + srcRank) : axis;
    }
    std::vector<size_t> shape5D = getShape5D(srcDataDim);
    std::vector<size_t> block5D = getBlock5D(shape5D);
    size_t idxLength = 1;
    for (size_t ri = 0; ri < indicesRank; ri++) {
        idxLength *= indicesDim[ri];
    }

    // if src and dst is the same pointer, no need copy value
    if (srcData != dstData) {
        std::memcpy(dstData, srcData, shape5D[0] * block5D[0] * sizeof(data_t));
    }

    if (mode == ScatterUpdateMode::ScatterUpdate) {
        // For the data tensor of shape [d_0, d_1, ..., d_n],
        // and indices tensor of shape [i_0, i_1, ..., i_k].
        // Updates tensor shape should be [d_0, d_1, ... d_(axis - 1), i_0, i_1, ..., i_k, d_(axis + 1), ..., d_n].
        if (updateRank != (srcRank + indicesRank -1)) {
            THROW_IE_EXCEPTION << "Rank of input, indices and update is not matched for layer " << getName()
                               << "with type" << getType();
        }

        SizeVector expectUpdateShape = {};
        for (size_t rs = 0; rs < srcRank; rs++) {
            if (rs != axis) {
                expectUpdateShape.push_back(srcDataDim[rs]);
            } else {
                for (size_t ri = 0; ri < indicesRank; ri++) {
                    expectUpdateShape.push_back(indicesDim[ri]);
                }
            }
        }
        for (size_t ru = 0; ru < updateRank; ru++) {
            if (updateDim[ru] != expectUpdateShape[ru]) {
                THROW_IE_EXCEPTION << "Shape of update tensor is not matched for input and indices for layer " << getName()
                                   << "with type" << getType();
            }
        }

        switch (axis) {
            case 0:
                parallel_for(idxLength, [&](size_t idx) {
                    data_t *dstEntry = dstData + indices[idx] * block5D[axis];
                    data_t *updateEntry = update + idx * block5D[axis];
                    std::memcpy(dstEntry, updateEntry, block5D[axis] * sizeof(data_t));
                });
                break;
            case 1:
                parallel_for2d(shape5D[0], idxLength, [&](size_t n, size_t idx) {
                    data_t *dstEntry = dstData + n * block5D[0] + indices[idx] * block5D[axis];
                    data_t *updateEntry = update + n * block5D[0] + idx * block5D[axis];
                    std::memcpy(dstEntry, updateEntry, block5D[axis] * sizeof(data_t));
                });
                break;
            case 2:
                parallel_for3d(shape5D[0], shape5D[1], idxLength, [&](size_t n, size_t c, size_t idx) {
                    data_t *dstEntry = dstData + n * block5D[0] + c * block5D[1] + indices[idx] * block5D[axis];
                    data_t *updateEntry = update + n * block5D[0] + c * block5D[1] + idx * block5D[axis];
                    std::memcpy(dstEntry, updateEntry, block5D[axis] * sizeof(data_t));
                });
                break;
            case 3:
                parallel_for4d(shape5D[0], shape5D[1], shape5D[2], idxLength, [&](size_t n, size_t c, size_t d, size_t idx) {
                    data_t *dstEntry = dstData + n * block5D[0] + c * block5D[1] + d * block5D[2] + indices[idx] * block5D[axis];
                    data_t *updateEntry = update + n * block5D[0] + c * block5D[1] + d * block5D[2] + idx * block5D[axis];
                    std::memcpy(dstEntry, updateEntry, block5D[axis] * sizeof(data_t));
                });
                break;
            case 4:
                parallel_for5d(shape5D[0], shape5D[1], shape5D[2], shape5D[3], idxLength, [&](size_t n, size_t c, size_t d, size_t h, size_t idx) {
                    data_t *dstEntry = dstData + n * block5D[0] + c * block5D[1] + d * block5D[2] + h * block5D[3] + indices[idx] * block5D[axis];
                    data_t *updateEntry = update + n * block5D[0] + c * block5D[1] + d * block5D[2] + h * block5D[3] + idx * block5D[axis];
                    std::memcpy(dstEntry, updateEntry, block5D[axis] * sizeof(data_t));
                });
                break;
            default:
                THROW_IE_EXCEPTION << "Axis should be less than 5 for layer " << getName()
                                   << "with type" << getType();
        }
    } else if (mode == ScatterUpdateMode::ScatterNDUpdate) {
        // indices is a (q-1)-dimension k-tuple, k is indices.shape[-1], q is ranke of indicies.
        // updates is a (q-1)-dimensional tensor of replacement-slice-values
        size_t k = indicesDim[indicesRank - 1];
        if (k > srcRank) {
            THROW_IE_EXCEPTION << "The last dimension of indices should not larger than the rank of input data for layer " << getName()
                               << "with type" << getType();
        }

        SizeVector expectUpdateShape = {};
        size_t tupleRank = indicesRank - 1;
        for (size_t ri = 0; ri < tupleRank; ri++) {
            expectUpdateShape.push_back(indicesDim[ri]);
        }
        for (size_t rd = k; rd < srcRank; rd++) {
            expectUpdateShape.push_back(srcDataDim[rd]);
        }
        if (expectUpdateShape.size() != updateRank) {
            THROW_IE_EXCEPTION << "Rank of update tensor is not matched for input and indices for layer " << getName()
                                   << "with type" << getType();
        }
        for (size_t ru = 0; ru < updateRank; ru++) {
            if (updateDim[ru] != expectUpdateShape[ru]) {
                THROW_IE_EXCEPTION << "Shape of update tensor is not matched for input and indices for layer " << getName()
                                   << "with type" << getType();
            }
        }

        size_t idxTupleLength = 1;
        for (size_t ri = 0; ri < indicesRank - 1; ri++) {
            idxTupleLength *= indicesDim[ri];
        }

        switch (k) {
            case 1:
                // tuple of 1 element
                parallel_for(idxTupleLength, [&](size_t tupleIdx) {
                    data_t *dstEntry = dstData + indices[tupleIdx * k] * block5D[0];
                    data_t *updateEntry = update + tupleIdx * block5D[0];
                    std::memcpy(dstEntry, updateEntry, block5D[0] * sizeof(data_t));
                });
                break;
            case 2:
                // tuples of 2 elements
                parallel_for(idxTupleLength, [&](size_t tupleIdx) {
                    data_t *dstEntry = dstData + indices[tupleIdx * k] * block5D[0] + indices[tupleIdx * k + 1] * block5D[1];
                    data_t *updateEntry = update + tupleIdx * block5D[1];
                    std::memcpy(dstEntry, updateEntry, block5D[1] * sizeof(data_t));
                });
                break;
            case 3:
                // tuples of 3 elements
                parallel_for(idxTupleLength, [&](size_t tupleIdx) {
                    data_t *dstEntry = dstData + indices[tupleIdx * k] * block5D[0] + indices[tupleIdx * k + 1] * block5D[1] +
                                        indices[tupleIdx * k + 2] * block5D[2];
                    data_t *updateEntry = update + tupleIdx * block5D[2];
                    std::memcpy(dstEntry, updateEntry, block5D[2] * sizeof(data_t));
                });
                break;
            case 4:
                // tuples of 4 elements
                parallel_for(idxTupleLength, [&](size_t tupleIdx) {
                    data_t *dstEntry = dstData + indices[tupleIdx * k] * block5D[0] + indices[tupleIdx * k + 1] * block5D[1] +
                                        indices[tupleIdx * k + 2] * block5D[2] + indices[tupleIdx * k + 3] * block5D[3];
                    data_t *updateEntry = update + tupleIdx * block5D[3];
                    std::memcpy(dstEntry, updateEntry, block5D[3] * sizeof(data_t));
                });
                break;
            case 5:
                // tuples of 5 elements
                parallel_for(idxTupleLength, [&](size_t tupleIdx) {
                    data_t *dstEntry = dstData + indices[tupleIdx * k] * block5D[0] + indices[tupleIdx * k + 1] * block5D[1] +
                            indices[tupleIdx * k + 2] * block5D[2] + indices[tupleIdx * k + 3] * block5D[3] + indices[tupleIdx * k + 4] * block5D[4];
                    data_t *updateEntry = update + tupleIdx * block5D[4];
                    std::memcpy(dstEntry, updateEntry, block5D[4] * sizeof(data_t));
                });
                break;
            default:
                THROW_IE_EXCEPTION << "Indices.shape[-1] should be less than 5 for layer " << getName()
                                   << "with type" << getType();
        }
    } else if (mode == ScatterUpdateMode::ScatterElementsUpdate) {
        // output[indices[i][j][k]][j][k] = updates[i][j][k] if axis = 0,
        // output[i][indices[i][j][k]][k] = updates[i][j][k] if axis = 1,
        // output[i][j][indices[i][j][k]] = updates[i][j][k] if axis = 2
        if (srcRank != indicesRank || srcRank != updateRank) {
            THROW_IE_EXCEPTION << "Rank of input, indices and update should be the same for layer " << getName()
                               << "with type" << getType();
        }
        for (size_t ri = 0; ri < indicesRank; ri++) {
            if (indicesDim[ri] != updateDim[ri]) {
                THROW_IE_EXCEPTION << "Shape of indices and update should be the same for layer " << getName()
                                   << "with type" << getType();
            }
        }
        std::vector<size_t> shape5DUpdate = getShape5D(updateDim);
        std::vector<size_t> block5DUpdate = getBlock5D(shape5DUpdate);
        parallel_for5d(shape5DUpdate[0], shape5DUpdate[1], shape5DUpdate[2], shape5DUpdate[3], shape5DUpdate[4],
            [&](size_t n, size_t c, size_t d, size_t h, size_t w) {
            size_t updatePosition = n * block5DUpdate[0] + c * block5DUpdate[1] + d * block5DUpdate[2] + h * block5DUpdate[3] + w * block5DUpdate[4];
            size_t dstPosition = 0;
            switch (axis) {
                case 0:
                    dstPosition = indices[updatePosition] * block5D[0] + c * block5D[1] + d * block5D[2] + h * block5D[3] + w * block5D[4];
                    break;
                case 1:
                    dstPosition = n * block5D[0] + indices[updatePosition] * block5D[1] + d * block5D[2] + h * block5D[3] + w * block5D[4];
                    break;
                case 2:
                    dstPosition = n * block5D[0] + c * block5D[1] + indices[updatePosition] * block5D[2] + h * block5D[3] + w * block5D[4];
                    break;
                case 3:
                    dstPosition = n * block5D[0] + c * block5D[1] + d * block5D[2] + indices[updatePosition] * block5D[3] + w * block5D[4];
                    break;
                case 4:
                    dstPosition = n * block5D[0] + c * block5D[1] + d * block5D[2] + h * block5D[3] + indices[updatePosition] * block5D[4];
                    break;
                default:
                    THROW_IE_EXCEPTION << "Axis should be less than 5 for layer " << getName()
                                       << "with type" << getType();
            }
            dstData[dstPosition] = update[updatePosition];
        });
    } else {
        THROW_IE_EXCEPTION << "unsupported ScatterUpdate mode for layer " << getName();
    }
}

bool MKLDNNScatterUpdateNode::created() const {
    return getType() == ScatterUpdate || getType() == ScatterElementsUpdate || getType() == ScatterNDUpdate;
}

REG_MKLDNN_PRIM_FOR(MKLDNNScatterUpdateNode, ScatterUpdate);
REG_MKLDNN_PRIM_FOR(MKLDNNScatterUpdateNode, ScatterElementsUpdate);
REG_MKLDNN_PRIM_FOR(MKLDNNScatterUpdateNode, ScatterNDUpdate);