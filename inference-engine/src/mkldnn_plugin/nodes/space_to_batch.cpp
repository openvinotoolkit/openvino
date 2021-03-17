// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"
#include "ie_parallel.hpp"

#include <cmath>
#include <string>
#include <vector>
#include <set>
#include <cassert>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class SpaceToBatchImpl: public ExtLayerBase {
public:
    explicit SpaceToBatchImpl(const CNNLayer* layer) {
        try {
            auto spaceToBatchLayer = dynamic_cast<const SpaceToBatchLayer*>(layer);
            if (!spaceToBatchLayer)
                IE_THROW() << "SpaceToBatch layer with name '" << layer->name << "' isn't instance of SpaceToBatchLayer class";

            if (spaceToBatchLayer->insData.size() != 4)
                IE_THROW() << "SpaceToBatch layer with name '" << spaceToBatchLayer->name << "' has incorrect number of input edges";

            if (spaceToBatchLayer->outData.size() != 1)
                IE_THROW() << "SpaceToBatch layer with name '" << spaceToBatchLayer->name << "' has incorrect number of output edges";

            auto data = spaceToBatchLayer->insData[0].lock();
            if (!data)
                THROW_IE_EXCEPTION << "SpaceToBatch layer with name '" << spaceToBatchLayer->name << "' has nullable input data";

            inDims = data->getTensorDesc().getDims();
            if (inDims.size() < 4)
                IE_THROW() << "SpaceToBatch layer with name '" << spaceToBatchLayer->name << "' doesn't support dimensions with rank less than 4";

            if (inDims.size() > 5)
                IE_THROW() << "SpaceToBatch layer with name '" << spaceToBatchLayer->name << "' doesn't support dimensions with rank greater than 5";

            outDims = spaceToBatchLayer->outData[0]->getTensorDesc().getDims();
            if (inDims.size() != outDims.size())
                IE_THROW() << "SpaceToBatch layer with name '" << spaceToBatchLayer->name << "' has incorrect number of input/output dimensions";

            const auto precision = data->getTensorDesc().getPrecision();
            const std::set<size_t> supported_precision_sizes = {1, 2, 4, 8};
            if (supported_precision_sizes.find(precision.size()) == supported_precision_sizes.end())
                IE_THROW() << "SpaceToBatch layer with name '" << spaceToBatchLayer->name << "' has unsupported precision: " << precision.name();

            blockShape = spaceToBatchLayer->_block_shape;
            padsBegin = spaceToBatchLayer->_pads_begin;

            auto createConfig = [&](Layout layout) {
                LayerConfig config;
                // TODO: remove Const layers
                for (int i = 0; i < spaceToBatchLayer->insData.size(); i++) {
                    auto inData = spaceToBatchLayer->insData[i].lock();
                    if (!inData)
                        IE_THROW() << "SpaceToBatch layer with name '" << spaceToBatchLayer->name << "' has nullable input data";
                    DataConfig inConfig;
                    if (i == 0)
                        inConfig.desc = TensorDesc(precision, inData->getTensorDesc().getDims(), layout);
                    else
                        inConfig.desc = TensorDesc(inData->getPrecision(), inData->getTensorDesc().getDims(), inData->getTensorDesc().getLayout());
                    config.inConfs.push_back(inConfig);
                }

                DataConfig outConfig;
                outConfig.desc = TensorDesc(precision, outDims, layout);
                config.outConfs.push_back(outConfig);

                config.dynBatchSupport = false;
                confs.push_back(config);
            };

            createConfig(inDims.size() == 4 ? NHWC : NDHWC);
            createConfig(TensorDesc::getLayoutByDims(inDims));

            std::vector<std::pair<ConfLayout, ConfLayout> > blockConfs {
                    {ConfLayout::BLK16, ConfLayout::BLK16},
                    {ConfLayout::BLK8, ConfLayout::BLK8}
            };
            for (auto conf : blockConfs) {
                addConfig(layer, {DataConfigurator(conf.first, precision),
                                  DataConfigurator(ConfLayout::ANY, spaceToBatchLayer->insData[1].lock()->getPrecision()),
                                  DataConfigurator(ConfLayout::ANY, spaceToBatchLayer->insData[2].lock()->getPrecision()),
                                  DataConfigurator(ConfLayout::ANY, spaceToBatchLayer->insData[3].lock()->getPrecision())},
                          {DataConfigurator(conf.second, precision)});
            }
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
        switch (inputs[0]->getTensorDesc().getPrecision().size()) {
            case 1: spaceToBatchKernel<PrecisionTrait<Precision::U8>::value_type> (inputs, outputs); break;
            case 2: spaceToBatchKernel<PrecisionTrait<Precision::U16>::value_type>(inputs, outputs); break;
            case 4: spaceToBatchKernel<PrecisionTrait<Precision::I32>::value_type>(inputs, outputs); break;
            case 8: spaceToBatchKernel<PrecisionTrait<Precision::U64>::value_type>(inputs, outputs); break;
            default: {
                if (resp) {
                    std::string errorMsg = "SpaceToBatch layer with name does not support precision '"
                                           + std::string(inputs[0]->getTensorDesc().getPrecision().name()) + "'";
                    errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);

                    return GENERAL_ERROR;
                }
            }
        }

        return OK;
    }

private:
    std::vector<size_t> getShape5D(const SizeVector& shape) {
        std::vector<size_t> shape5D(5, 1);
        for (int i = 0; i < shape.size(); i++) {
            shape5D[i] = shape[i];
        }
        return shape5D;
    }
    template<typename T>
    void spaceToBatchKernel(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs) noexcept {
        const T* srcData = inputs[0]->cbuffer().as<const T *>() + inputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        T* dstData = outputs[0]->buffer().as<T *>() + outputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        const size_t dimsSize = inDims.size();
        const auto layout = inputs[0]->getTensorDesc().getLayout();

        const auto inShape5D  = getShape5D(outDims);
        const auto outShape5D = getShape5D(inDims);

        const size_t inSpatialStep = inShape5D[2] * inShape5D[3] * inShape5D[4];
        const size_t inBatchStep   = inShape5D[1] * inSpatialStep;

        const size_t outSpatialStep = outShape5D[2] * outShape5D[3] * outShape5D[4];
        const size_t outBatchStep   = outShape5D[1] * outSpatialStep;

        const int64_t padFront = padsBegin[dimsSize - 3];
        const int64_t padTop   = padsBegin[dimsSize - 2];
        const int64_t padLeft  = padsBegin[dimsSize - 1];

        const size_t ID = dimsSize == 5 ? inShape5D[dimsSize - 3] : 1lu;
        const size_t OD = dimsSize == 5 ? outShape5D[dimsSize - 3] : 1lu;

        if (layout == NHWC || layout == NDHWC) {
            parallel_for(inShape5D[0], [&](size_t i0) {
                int64_t bIdx = i0 / outShape5D[0];
                const size_t srcIdx1 = (i0 - (bIdx * outShape5D[0])) * outBatchStep;
                const size_t dstIdx1 = i0 * inBatchStep;
                const int64_t owAdd = bIdx % blockShape[dimsSize - 1] - padLeft;
                bIdx /= blockShape[dimsSize - 1];
                const int64_t ohAdd = bIdx % blockShape[dimsSize - 2] - padTop;
                bIdx /= blockShape[dimsSize - 2];
                const int64_t odAdd = layout == NDHWC ? bIdx % blockShape[dimsSize - 3] - padFront : 0;
                bIdx = layout == NDHWC ? bIdx/blockShape[dimsSize - 3] : bIdx;
                const int64_t ocAdd = bIdx % blockShape[1] - padsBegin[1];
                const size_t i1Begin = (blockShape[1] - 1 - ocAdd) / blockShape[1];
                const size_t i1End = (outShape5D[1] - 1 - ocAdd) / blockShape[1] + 1;
                const size_t i2Begin = layout == NDHWC ? (blockShape[dimsSize - 3] - 1 - odAdd) / blockShape[dimsSize - 3] : 0lu;
                const size_t i2End   = layout == NDHWC ? (outShape5D[dimsSize - 3] - 1 - odAdd) / blockShape[dimsSize - 3] + 1 : 1lu;
                const size_t i3Begin = (blockShape[dimsSize - 2] - 1 - ohAdd) / blockShape[dimsSize - 2];
                const size_t i3End = (outShape5D[dimsSize - 2] - 1 - ohAdd) / blockShape[dimsSize - 2] + 1;
                const size_t i4Begin = (blockShape[dimsSize - 1] - 1 - owAdd) / blockShape[dimsSize - 1];
                const size_t i4End = (outShape5D[dimsSize - 1] - 1 - owAdd) / blockShape[dimsSize - 1] + 1;
                for (size_t i = 0; i < ID; ++i) {
                    const size_t end = inShape5D[dimsSize - 2] * inShape5D[dimsSize - 1] * inShape5D[1];
                    const size_t dstIdx2 = dstIdx1 + i * end;
                    std::fill(dstData + dstIdx2, dstData + dstIdx2 + end, T(0));
                }
                for (size_t i2 = i2Begin; i2 < i2End; ++i2) {
                    const size_t tmpOd = i2 * blockShape[dimsSize - 3] + odAdd;
                    const size_t dstIdx2 = dstIdx1 + i2 * inShape5D[dimsSize - 2] * inShape5D[dimsSize - 1] * inShape5D[1];
                    const size_t srcIdx2 = srcIdx1 + tmpOd * outShape5D[dimsSize - 2] * outShape5D[dimsSize - 1] * outShape5D[1];
                    for (size_t i3 = i3Begin; i3 < i3End; ++i3) {
                        const size_t tmpOh = i3 * blockShape[dimsSize - 2] + ohAdd;
                        const size_t dstIdx3 = dstIdx2 + i3 * inShape5D[dimsSize - 1] * inShape5D[1];
                        const size_t srcIdx3 = srcIdx2 + tmpOh * outShape5D[dimsSize - 1] * outShape5D[1];
                        for (size_t i4 = i4Begin; i4 < i4End; ++i4) {
                            const size_t tmpOw = i4 * blockShape[dimsSize - 1] + owAdd;
                            const size_t dstIdx4 = dstIdx3 + i4 * inShape5D[1];
                            const size_t srcIdx4 = srcIdx3 + tmpOw * outShape5D[1];
                            for (size_t i1 = i1Begin; i1 < i1End; ++i1) {
                                const size_t tmpOc = i1 * blockShape[1] + ocAdd;
                                const size_t dstIdx5 = dstIdx4 + i1;
                                const size_t srcIdx5 = srcIdx4 + tmpOc;
                                dstData[dstIdx5] = srcData[srcIdx5];
                            }
                        }
                    }
                }
            });
        } else if (layout == NCHW || layout == NCDHW) {
            parallel_for(inShape5D[0], [&](size_t i0) {
                int64_t bIdx = i0 / outShape5D[0];
                const size_t srcIdx0 = (i0 - (bIdx * outShape5D[0])) * outBatchStep;
                const size_t dstIdx0 = i0 * inBatchStep;
                const int64_t owAdd = bIdx % blockShape[dimsSize - 1] - padLeft;
                bIdx /= blockShape[dimsSize - 1];
                const int64_t ohAdd = bIdx % blockShape[dimsSize - 2] - padTop;
                bIdx /= blockShape[dimsSize - 2];
                const int64_t odAdd = layout == NCDHW ? bIdx % blockShape[dimsSize - 3] - padFront : 0;
                bIdx = layout == NCDHW ? bIdx/blockShape[dimsSize - 3] : bIdx;
                const int64_t ocAdd = bIdx % blockShape[1] - padsBegin[1];
                const size_t i1Begin = (blockShape[1] - 1 - ocAdd) / blockShape[1];
                const size_t i1End = (outShape5D[1] - 1 - ocAdd) / blockShape[1] + 1;
                const size_t i2Begin = layout == NCDHW ? (blockShape[dimsSize - 3] - 1 - odAdd) / blockShape[dimsSize - 3] : 0lu;
                const size_t i2End = layout == NCDHW ? (outShape5D[dimsSize - 3] - 1 - odAdd) / blockShape[dimsSize - 3] + 1 : 1lu;
                const size_t i3Begin = (blockShape[dimsSize - 2] - 1 - ohAdd) / blockShape[dimsSize - 2];
                const size_t i3End = (outShape5D[dimsSize - 2] - 1 - ohAdd) / blockShape[dimsSize - 2] + 1;
                const size_t i4Begin = (blockShape[dimsSize - 1] - 1 - owAdd) / blockShape[dimsSize - 1];
                const size_t i4End = (outShape5D[dimsSize - 1] - 1 - owAdd) / blockShape[dimsSize - 1] + 1;
                for (size_t i = 0; i < inShape5D[1]; ++i) {
                    const size_t end = inShape5D[2] * inShape5D[3] * inShape5D[4];
                    const size_t dstIdx = dstIdx0 + i * end;
                    std::fill(dstData + dstIdx, dstData + dstIdx + end, T(0));
                }
                for (size_t i1 = i1Begin; i1 < i1End; ++i1) {
                    const int64_t tmpOc = i1 * blockShape[1] + ocAdd;
                    const size_t srcIdx1 = srcIdx0 + tmpOc * outSpatialStep;
                    const size_t dstIdx1 = dstIdx0 + i1 * inSpatialStep;
                    for (size_t i2 = i2Begin; i2 < i2End; ++i2) {
                        const int64_t tmpOd = i2 * blockShape[dimsSize - 3] + odAdd;
                        const size_t dstIdx2 = dstIdx1 + i2 * inShape5D[dimsSize - 2] * inShape5D[dimsSize - 1];
                        const size_t srcIdx2 = srcIdx1 + tmpOd * outShape5D[dimsSize - 2] * outShape5D[dimsSize - 1];
                        for (size_t i3 = i3Begin; i3 < i3End; ++i3) {
                            const int64_t tmpOh = i3 * blockShape[dimsSize - 2] + ohAdd;
                            const size_t dstIdx3 = dstIdx2 + i3 * inShape5D[dimsSize - 1];
                            const size_t srcIdx3 = srcIdx2 + tmpOh * outShape5D[dimsSize - 1];
                            for (size_t i4 = i4Begin; i4 < i4End; ++i4) {
                                const int64_t tmpOw = i4 * blockShape[dimsSize - 1] + owAdd;
                                const size_t dstIdx4 = dstIdx3 + i4;
                                const size_t srcIdx4 = srcIdx3 + tmpOw;
                                dstData[dstIdx4] = srcData[srcIdx4];
                            }
                        }
                    }
                }
            });
        } else {  // nC[d]hw16c, nC[d]hw8c
            const size_t blockSize  = inputs[0]->getTensorDesc().getBlockingDesc().getBlockDims().back();
            const size_t blockCountInput = outputs[0]->getTensorDesc().getBlockingDesc().getBlockDims()[1];
            const size_t blockCountOutput = inputs[0]->getTensorDesc().getBlockingDesc().getBlockDims()[1];

            parallel_for(inShape5D[0], [&](size_t i0) {
                int64_t bIdx = i0 / outShape5D[0];
                const size_t srcIdx0 = (i0 - (bIdx * outShape5D[0])) * blockCountOutput * blockSize * OD * outShape5D[dimsSize - 2] * outShape5D[dimsSize - 1];
                const size_t dstIdx0 = i0 * blockCountInput * blockSize * ID * inShape5D[dimsSize - 2] * inShape5D[dimsSize - 1];
                const int64_t owAdd = bIdx % blockShape[dimsSize - 1] - padLeft;
                bIdx /= blockShape[dimsSize - 1];
                const int64_t ohAdd = bIdx % blockShape[dimsSize - 2] - padTop;
                bIdx /= blockShape[dimsSize - 2];
                const int64_t odAdd = dimsSize == 5 ? bIdx % blockShape[dimsSize - 3] - padFront : 0lu;
                bIdx = dimsSize == 5 ? bIdx / blockShape[dimsSize - 3] : bIdx;
                const int64_t ocAdd = bIdx % blockShape[1] - padsBegin[1];
                const size_t i1End = ((outShape5D[1] - 1 - ocAdd) / blockShape[1]) / blockSize + 1;
                const size_t i2Begin = dimsSize == 5 ? (blockShape[dimsSize - 3] - 1 - odAdd) / blockShape[dimsSize - 3] : 0lu;
                const size_t i2End = dimsSize == 5 ? (outShape5D[dimsSize - 3] - 1 - odAdd) / blockShape[dimsSize - 3] + 1 : 1lu;
                const size_t i3Begin = (blockShape[dimsSize - 2] - 1 - ohAdd) / blockShape[dimsSize - 2];
                const size_t i3End = (outShape5D[dimsSize - 2] - 1 - ohAdd) / blockShape[dimsSize - 2] + 1;
                const size_t i4Begin = (blockShape[dimsSize - 1] - 1 - owAdd) / blockShape[dimsSize - 1];
                const size_t i4End = (outShape5D[dimsSize - 1] - 1 - owAdd) / blockShape[dimsSize - 1] + 1;
                for (size_t i = 0; i < blockCountInput; ++i) {
                    const size_t end = ID * inShape5D[dimsSize - 2] * inShape5D[dimsSize - 1] * blockSize;
                    const size_t dstIdx = dstIdx0 + i * end;
                    std::fill(dstData + dstIdx, dstData + dstIdx + end, T(0));
                }
                for (size_t i1 = 0; i1 < i1End; ++i1) {
                    const size_t srcIdx1 = srcIdx0 + (i1 * blockShape[1]) * OD * outShape5D[dimsSize - 2] * outShape5D[dimsSize - 1] * blockSize;
                    const size_t dstIdx1 = dstIdx0 + i1 * ID * inShape5D[dimsSize - 2] * inShape5D[dimsSize - 1] * blockSize;
                    const size_t startI5 = (blockShape[1] - 1 - ocAdd) / blockShape[1];
                    const size_t i5BeginConst = i1 == 0 ? startI5 : 0lu;
                    size_t i5EndConst = (blockSize * blockCountOutput - 1 - ocAdd) / blockShape[1] - (i1 * blockSize) + 1;
                    i5EndConst = i5EndConst > blockSize ? blockSize : i5EndConst;
                    const size_t tempIt = (i5BeginConst * blockShape[1] + ocAdd) / blockSize;
                    size_t itEnd = ((i5EndConst - 1) * blockShape[1] + ocAdd) / blockSize + 1;
                    itEnd = itEnd > i5EndConst ? 1 : itEnd;
                    for (size_t i2 = i2Begin; i2 < i2End; ++i2) {
                        const int64_t tmpOd = i2 * blockShape[dimsSize - 3] + odAdd;
                        const size_t srcIdx2 = srcIdx1 + tmpOd * outShape5D[dimsSize - 2] * outShape5D[dimsSize - 1] * blockSize;
                        const size_t dstIdx2 = dstIdx1 + i2 * inShape5D[dimsSize - 2] * inShape5D[dimsSize - 1] * blockSize;
                        for (size_t i3 = i3Begin; i3 < i3End; ++i3) {
                            const int64_t tmpOh = i3 * blockShape[dimsSize - 2] + ohAdd;
                            const size_t srcIdx3 = srcIdx2 + tmpOh * outShape5D[dimsSize - 1] * blockSize;
                            const size_t dstIdx3 = dstIdx2 + i3 * inShape5D[dimsSize - 1] * blockSize;
                            for (size_t i4 = i4Begin; i4 < i4End; ++i4) {
                                const int64_t tmpOw = i4 * blockShape[dimsSize - 1] + owAdd;
                                const size_t srcIdx4 = srcIdx3 + tmpOw * blockSize;
                                const size_t dstIdx4 = dstIdx3 + i4 * blockSize;
                                for (size_t it = 0; it < itEnd; ++it) {
                                    size_t i5Begin = it == 0 ? i5BeginConst : (it * blockSize - 1 - ocAdd) / blockShape[1] + 1;
                                    const size_t i5End = it == itEnd - 1 ? i5EndConst : ((it + 1) * blockSize - 1 - ocAdd) / blockShape[1] + 1;
                                    for (size_t i5 = i5Begin; i5 < startI5; ++i5, ++i5Begin) {
                                        const int64_t tmpOc = i5 * blockShape[1] + ocAdd;
                                        const size_t srcIdx5 = srcIdx4 + (tempIt) * OD * outShape5D[dimsSize - 2] * outShape5D[dimsSize - 1] * blockSize
                                                                       + (tmpOc - tempIt * blockSize);
                                        const size_t dstIdx5 = dstIdx4 + i5;
                                        dstData[dstIdx5] = srcData[srcIdx5];
                                    }
                                    for (size_t i5 = i5Begin; i5 < i5End; ++i5) {
                                        const int64_t tmpOc = i5 * blockShape[1] + ocAdd;
                                        const size_t srcIdx5 = srcIdx4 + it * OD * outShape5D[dimsSize - 2] * outShape5D[dimsSize - 1] * blockSize
                                                                       + (tmpOc - it * blockSize);
                                        const size_t dstIdx5 = dstIdx4 + i5;
                                        dstData[dstIdx5] = srcData[srcIdx5];
                                    }
                                }
                            }
                        }
                    }
                }
            });
        }
    }

    SizeVector inDims;
    SizeVector outDims;
    std::vector<size_t> blockShape;
    std::vector<size_t> padsBegin;
};

REG_FACTORY_FOR(SpaceToBatchImpl, SpaceToBatch);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine

