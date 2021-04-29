// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"
#include "ie_parallel.hpp"

#include <cmath>
#include <string>
#include <vector>
#include <set>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class BatchToSpaceImpl: public ExtLayerBase {
public:
    explicit BatchToSpaceImpl(const CNNLayer *layer) {
        try {
            const auto batchToSpaceLayer = dynamic_cast<const BatchToSpaceLayer*>(layer);
            if (!batchToSpaceLayer)
                IE_THROW() << "BatchToSpace layer with name '" << layer->name << "' isn't instance of BatchToSpaceLayer class";

            if (batchToSpaceLayer->insData.size() != 4)
                IE_THROW() << "BatchToSpace layer with name '" << batchToSpaceLayer->name << "' has incorrect number of input edges";

            if (batchToSpaceLayer->outData.size() != 1)
                IE_THROW() << "BatchToSpace layer with name '" << batchToSpaceLayer->name << "' has incorrect number of output edges";

            auto data = batchToSpaceLayer->insData[0].lock();
            if (!data)
                IE_THROW() << "BatchToSpace layer with name '" << batchToSpaceLayer->name << "' has nullable input data";

            inDims = data->getTensorDesc().getDims();
            if (inDims.size() < 4)
                IE_THROW() << "BatchToSpace layer with name '" << batchToSpaceLayer->name << "' doesn't support dimensions with rank less than 4";

            if (inDims.size() > 5)
                IE_THROW() << "BatchToSpace layer with name '" << batchToSpaceLayer->name << "' doesn't support dimensions with rank greater than 5";

            outDims = batchToSpaceLayer->outData[0]->getTensorDesc().getDims();
            if (inDims.size() != outDims.size())
                IE_THROW() << "BatchToSpace layer with name '" << batchToSpaceLayer->name << "' has incorrect number of input/output dimensions";

            const auto precision = data->getTensorDesc().getPrecision();
            const std::set<size_t> supported_precision_sizes = {1, 2, 4, 8};
            if (supported_precision_sizes.find(precision.size()) == supported_precision_sizes.end())
                IE_THROW() << "BatchToSpace layer with name '" << batchToSpaceLayer->name << "' has unsupported precision: " << precision.name();

            blockShapeIn = batchToSpaceLayer->_block_shape;
            cropsBeginIn = batchToSpaceLayer->_crops_begin;

            auto createConfig = [&](Layout layout) {
                LayerConfig config;
                // TODO: remove Const layers
                for (int i = 0; i < batchToSpaceLayer->insData.size(); i++) {
                    auto inData = batchToSpaceLayer->insData[i].lock();
                    if (!inData)
                        IE_THROW() << "BatchToSpace layer with name '" << batchToSpaceLayer->name << "' has nullable input data";
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

            std::vector<std::pair<ConfLayout, ConfLayout>>  blockConfs { };
            if (inDims[1] % 8 == 0)  blockConfs.push_back({ConfLayout::BLK8, ConfLayout::BLK8});
            if (inDims[1] % 16 == 0) blockConfs.push_back({ConfLayout::BLK16, ConfLayout::BLK16});
            for (auto conf : blockConfs) {
                addConfig(layer, {DataConfigurator(conf.first, precision),
                                  DataConfigurator(ConfLayout::PLN, batchToSpaceLayer->insData[1].lock()->getPrecision()),
                                  DataConfigurator(ConfLayout::PLN, batchToSpaceLayer->insData[2].lock()->getPrecision()),
                                  DataConfigurator(ConfLayout::PLN, batchToSpaceLayer->insData[3].lock()->getPrecision())},
                          {DataConfigurator(conf.second, precision)});
            }
        } catch (InferenceEngine::Exception &ex) {
            errorMsg = ex.what();
        }
    }
    StatusCode execute(std::vector<Blob::Ptr> &inputs, std::vector<Blob::Ptr> &outputs, ResponseDesc *resp) noexcept override {
        switch (inputs[0]->getTensorDesc().getPrecision().size()) {
            case 1: batchToSpaceKernel<PrecisionTrait<Precision::U8>::value_type> (inputs, outputs);  break;
            case 2: batchToSpaceKernel<PrecisionTrait<Precision::U16>::value_type>(inputs, outputs); break;
            case 4: batchToSpaceKernel<PrecisionTrait<Precision::I32>::value_type>(inputs, outputs); break;
            default: {
                if (resp) {
                    std::string errorMsg = "BatchToSpace layer does not support precision '"
                                           + std::string(inputs[0]->getTensorDesc().getPrecision().name()) + "'";
                    errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
                    return GENERAL_ERROR;
                }
            }
        }
        return OK;
    }

private:
    std::vector<size_t> getShape5D(const SizeVector &shape) {
        std::vector<size_t> shape5D(5, 1);
        for (int i = 0; i < 2; i++) {
            shape5D[i] = shape[i];
            shape5D[4 - i] = shape[shape.size() - 1 - i];
        }
        shape5D[2] = shape.size() == 5 ? shape[2] : shape5D[2];
        return shape5D;
    }

    template<typename T>
    void batchToSpaceKernel(std::vector<Blob::Ptr> &inputs, std::vector<Blob::Ptr> &outputs) noexcept {
        const T *srcData = inputs[0]->cbuffer().as<const T *>() + inputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        T *dstData = outputs[0]->buffer().as<T *>() + outputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        const auto layout = inputs[0]->getTensorDesc().getLayout();
        const bool blocked = layout != NCHW && layout != NCDHW && layout != NHWC && layout != NDHWC;
        const auto dimsSize = inDims.size();

        auto inShape5D  = getShape5D(inDims);
        auto outShape5D = getShape5D(outDims);
        auto blockShape = getShape5D(blockShapeIn);

        if (layout == NHWC || layout == NDHWC) {
            inShape5D.push_back(inShape5D[1]);
            inShape5D.erase(inShape5D.begin() + 1);
            outShape5D.push_back(outShape5D[1]);
            outShape5D.erase(outShape5D.begin() + 1);
            blockShape.push_back(blockShape[1]);
            blockShape.erase(blockShape.begin() + 1);
        }

        const size_t blockSize = blocked ? outputs[0]->getTensorDesc().getBlockingDesc().getBlockDims().back() : 1lu;
        const size_t blockCountInput = inputs[0]->getTensorDesc().getBlockingDesc().getBlockDims()[1];
        const size_t blockCountOutput = outputs[0]->getTensorDesc().getBlockingDesc().getBlockDims()[1];
        const auto blockRemainder = inShape5D[1] % blockSize;
        const auto lastBlock = blockRemainder == 0 ? blockSize : blockRemainder;

        const size_t inSpatialStep = inShape5D[2] * inShape5D[3] * inShape5D[4];
        const size_t inBatchStep = (blocked ? blockSize * blockCountInput : inShape5D[1]) * inSpatialStep;

        const size_t outSpatialStep = outShape5D[2] * outShape5D[3] * outShape5D[4];
        const size_t outBatchStep = (blocked ? blockSize * blockCountOutput : outShape5D[1]) * outSpatialStep;

        size_t channels = (inShape5D[1] / blockSize);
        channels = channels == 0 ? 1 : channels;
        const size_t workAmount = inShape5D[0] * channels;

        parallel_nt(0, [&](const int ithr, const int nthr) {
            size_t start(0lu), end(0lu);
            splitter(workAmount, nthr, ithr, start, end);
            std::vector<size_t> indxStart(2, 0);
            std::vector<size_t> indxEnd(2, 0);
            parallel_it_init(start, indxStart[0], inShape5D[0], indxStart[1], channels);
            parallel_it_init((end - 1), indxEnd[0], inShape5D[0], indxEnd[1], channels);
            std::vector<int64_t> oAdd(5, 1);
            std::vector<size_t> begin(5, 0);
            std::vector<size_t> finish(5, 1);
            for (size_t i0 = indxStart[0]; i0 < indxEnd[0] + 1; ++i0) {
                int64_t bIdx = i0 / outShape5D[0];
                const size_t srcIdx0 = i0 * inBatchStep;
                const size_t dstIdx0 = (i0 - (bIdx * outShape5D[0])) * outBatchStep;
                oAdd[4] = bIdx % blockShapeIn[dimsSize - 1] - cropsBeginIn[dimsSize - 1];
                bIdx /= blockShapeIn[dimsSize - 1];
                oAdd[3] = bIdx % blockShapeIn[dimsSize - 2] - cropsBeginIn[dimsSize - 2];
                bIdx /= blockShapeIn[dimsSize - 2];
                oAdd[2] = dimsSize == 5 ? bIdx % blockShapeIn[2] - cropsBeginIn[2] : 0lu;
                bIdx = dimsSize == 5 ? bIdx / blockShapeIn[2] : bIdx;
                oAdd[1] = bIdx % blockShapeIn[1] - cropsBeginIn[1];
                if (layout == NHWC || layout == NDHWC) {
                    oAdd.push_back(oAdd[1]);
                    oAdd.erase(oAdd.begin() + 1);
                }
                begin[1] = (blockShape[1] - 1 - oAdd[1]) / blockShape[1] / blockSize;
                finish[1] = (outShape5D[1] - 1 - oAdd[1]) / blockShape[1] / blockSize;
                begin[2] = (blockShape[2] - 1 - oAdd[2]) / blockShape[2];
                finish[2] = (outShape5D[2] - 1 - oAdd[2]) / blockShape[2];
                begin[3] = (blockShape[3] - 1 - oAdd[3]) / blockShape[3];
                finish[3] = (outShape5D[3] - 1 - oAdd[3]) / blockShape[3];
                begin[4] = (blockShape[4] - 1 - oAdd[4]) / blockShape[4];
                finish[4] = (outShape5D[4] - 1 - oAdd[4]) / blockShape[4];
                const int64_t addTmpOC = blocked ? 0lu : oAdd[1];
                const int64_t addTmpOc = blocked ? oAdd[1] : 0lu;
                indxStart[1] = begin[1] > indxStart[1] ? begin[1] : indxStart[1];
                const size_t lastI1 = i0 == indxEnd[0] ? (indxEnd[1] > finish[1] ? finish[1] : indxEnd[1]) : finish[1];
                for (; indxStart[1] < lastI1 + 1; ++indxStart[1]) {
                    const size_t block = indxStart[1] == finish[1] ? lastBlock : blockSize;
                    const int64_t tmpOC = indxStart[1] * blockShape[1] + addTmpOC;
                    const size_t srcIdx1 = srcIdx0 + indxStart[1] * inSpatialStep * blockSize;
                    const size_t dstIdx1 = dstIdx0 + tmpOC * outSpatialStep * blockSize;
                    const size_t itEnd = blocked ? ((block - 1) * blockShape[1] + oAdd[1]) / blockSize : 0lu;
                    for (size_t i2 = begin[2]; i2 < finish[2] + 1; ++i2) {
                        const int64_t tmpOd = i2 * blockShape[2] + oAdd[2];
                        const size_t srcIdx2 = srcIdx1 + i2 * inShape5D[3] * inShape5D[4] * blockSize;
                        const size_t dstIdx2 = dstIdx1 + tmpOd * outShape5D[3] * outShape5D[4] * blockSize;
                        for (size_t i3 = begin[3]; i3 < finish[3] + 1; ++i3) {
                            const int64_t tmpOh = i3 * blockShape[3] + oAdd[3];
                            const size_t srcIdx3 = srcIdx2 + i3 * inShape5D[4] * blockSize;
                            const size_t dstIdx3 = dstIdx2 + tmpOh * outShape5D[4] * blockSize;
                            for (size_t i4 = begin[4]; i4 < finish[4] + 1; ++i4) {
                                const int64_t tmpOw = i4 * blockShape[4] + oAdd[4];
                                const size_t srcIdx4 = srcIdx3 + i4 * blockSize;
                                const size_t dstIdx4 = dstIdx3 + tmpOw * blockSize;
                                for (size_t it = 0; it < itEnd + 1; ++it) {
                                    const size_t i5Begin = it == 0 ? 0 : (it * blockSize - 1 - oAdd[1]) / blockShape[1] + 1;
                                    const size_t i5End = it == itEnd ? (block - 1) : ((it + 1) * blockSize - 1 - oAdd[1]) / blockShape[1];
                                    for (size_t i5 = i5Begin; i5 < i5End + 1; ++i5) {
                                        const int64_t tmpOc = i5 * blockShape[1] + addTmpOc;
                                        const size_t srcIdx5 = srcIdx4 + i5;
                                        const size_t dstIdx5 = dstIdx4 + it * outSpatialStep * blockSize + (tmpOc - it * blockSize);
                                        dstData[dstIdx5] = srcData[srcIdx5];
                                    }
                                }
                            }
                        }
                    }
                }
                indxStart[1] = 0lu;
            }
        });
    }

    SizeVector inDims;
    SizeVector outDims;
    std::vector<size_t> blockShapeIn;
    std::vector<size_t> cropsBeginIn;
};

REG_FACTORY_FOR(BatchToSpaceImpl, BatchToSpace);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
