// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"

#include <cmath>
#include <string>
#include <vector>
#include <cassert>
#include <set>
#include "ie_parallel.hpp"

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

            inDims = batchToSpaceLayer->insData[0].lock()->getTensorDesc().getDims();
            if (inDims.size() < 4)
                IE_THROW() << "BatchToSpace layer with name '" << batchToSpaceLayer->name << "' doesn't support dimensions with rank less than 4";

            if (inDims.size() > 5)
                IE_THROW() << "BatchToSpace layer with name '" << batchToSpaceLayer->name << "' doesn't support dimensions with rank greater than 5";

            outDims = batchToSpaceLayer->outData[0]->getTensorDesc().getDims();
            if (inDims.size() != outDims.size())
                IE_THROW() << "BatchToSpace layer with name '" << batchToSpaceLayer->name << "' has incorrect number of input/output dimensions";

            if (inDims[1] != outDims[1])
                IE_THROW() << "BatchToSpace layer with name '" << batchToSpaceLayer->name << "' has different IN and OUT channels number";

            const auto precision = batchToSpaceLayer->insData[0].lock()->getTensorDesc().getPrecision();
            const std::set<size_t> supported_precision_sizes = {1, 2, 4, 8};
            if (supported_precision_sizes.find(precision.size()) == supported_precision_sizes.end())
                IE_THROW() << "BatchToSpace layer with name '" << batchToSpaceLayer->name << "' has unsupported precision: " << precision.name();

            _block_shape = batchToSpaceLayer->_block_shape;
            _crops_begin = batchToSpaceLayer->_crops_begin;

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
            createConfig(TensorDesc::getLayoutByDims((inDims)));
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }
    StatusCode execute(std::vector<Blob::Ptr> &inputs, std::vector<Blob::Ptr> &outputs, ResponseDesc *resp) noexcept override {
        switch (inputs[0]->getTensorDesc().getPrecision().size()) {
            case 1: batchToSpaceKernel<PrecisionTrait<Precision::U8>::value_type> (inputs, outputs);  break;
            case 2: batchToSpaceKernel<PrecisionTrait<Precision::U16>::value_type>(inputs, outputs); break;
            case 4: batchToSpaceKernel<PrecisionTrait<Precision::I32>::value_type>(inputs, outputs); break;
            case 8: batchToSpaceKernel<PrecisionTrait<Precision::U64>::value_type>(inputs, outputs); break;
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
        for (int i = 0; i < shape.size(); i++) {
            shape5D[i] = shape[i];
        }
        return shape5D;
    }
    template<typename T>
    void batchToSpaceKernel(std::vector<Blob::Ptr> &inputs, std::vector<Blob::Ptr> &outputs) noexcept {
        const T *src_data = inputs[0]->cbuffer().as<const T *>() + inputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        T *dst_data = outputs[0]->buffer().as<T *>() + outputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        const size_t dimsSize = inDims.size();
        const auto layout = inputs[0]->getTensorDesc().getLayout();

        auto inShape5D  = getShape5D(inDims);
        auto outShape5D = getShape5D(outDims);

        size_t inSpatialStep = inShape5D[2] * inShape5D[3] * inShape5D[4];
        size_t inBatchStep = inShape5D[1] * inSpatialStep;

        size_t outSpatialStep = outShape5D[2] * outShape5D[3] * outShape5D[4];
        size_t outBatchStep = outShape5D[1] * outSpatialStep;

        size_t cropFront = _crops_begin[dimsSize - 3];
        size_t cropTop   = _crops_begin[dimsSize - 2];
        size_t cropLeft  = _crops_begin[dimsSize - 1];

        if (layout == NHWC || layout == NDHWC) {
            parallel_for(inShape5D[0], [&](size_t i0) {
                int64_t bIdx = i0 / outShape5D[0];
                size_t srcIdx1 = i0 * inBatchStep;
                size_t dstIdx1 = (i0 - (bIdx * outShape5D[0])) * outBatchStep;
                int64_t owAdd = bIdx % _block_shape[dimsSize - 1] - cropLeft;
                bIdx /= _block_shape[dimsSize - 1];
                int64_t ohAdd = (layout == NDHWC ? bIdx % _block_shape[dimsSize - 2] : bIdx) - cropTop;
                bIdx /= _block_shape[dimsSize - 2];
                int64_t odAdd = layout == NDHWC ? bIdx % _block_shape[dimsSize - 3] - cropFront : 0lu;
                size_t i2Begin = layout == NDHWC ? (_block_shape[dimsSize - 3] - 1 - odAdd) / _block_shape[dimsSize - 3] : 0lu;
                size_t i2End = layout == NDHWC ? (outShape5D[dimsSize - 3] - 1 - odAdd) / _block_shape[dimsSize - 3] + 1 : 1lu;
                for (size_t i2 = i2Begin; i2 < i2End; ++i2) {
                    int64_t tmpOd = i2 * _block_shape[dimsSize - 3] + odAdd;
                    size_t srcIdx2 = srcIdx1 + i2 * inShape5D[dimsSize - 2] * inShape5D[dimsSize - 1] * inShape5D[1];
                    size_t dstIdx2 = dstIdx1 + (tmpOd) * outShape5D[dimsSize - 2] * outShape5D[dimsSize - 1] * outShape5D[1];
                    int64_t i3Begin = (_block_shape[dimsSize - 2] - 1 - ohAdd) / _block_shape[dimsSize - 2];
                    int64_t i3End = (outShape5D[dimsSize - 2] - 1 - ohAdd) / _block_shape[dimsSize - 2] + 1;
                    for (size_t i3 = i3Begin; i3 < i3End; ++i3) {
                        int64_t tmpOh = i3 * _block_shape[dimsSize - 2] + ohAdd;
                        size_t srcIdx3 = srcIdx2 + i3 * inShape5D[dimsSize - 1] * inShape5D[1];
                        size_t dstIdx3 = dstIdx2 + tmpOh * outShape5D[dimsSize - 1] * outShape5D[1];
                        int64_t i4Begin = (_block_shape[dimsSize - 1] - 1 - owAdd) / _block_shape[dimsSize - 1];
                        int64_t i4End = (outShape5D[dimsSize - 1] - 1 - owAdd) / _block_shape[dimsSize - 1] + 1;
                        for (size_t i4 = i4Begin; i4 < i4End; ++i4) {
                            int64_t tmpOw = i4 * _block_shape[dimsSize - 1]  + owAdd;
                            size_t srcIdx4 = srcIdx3 + i4 * inShape5D[1];
                            size_t dstIdx4 = dstIdx3 + tmpOw * outShape5D[1];
                            for (size_t i1 = 0; i1 < inShape5D[1]; ++i1) {
                                size_t srcIdx5 = srcIdx4 + i1;
                                size_t dstIdx5 = dstIdx4 + i1;
                                dst_data[dstIdx5] = src_data[srcIdx5];
                            }
                        }
                    }
                }
            });
        } else {
            parallel_for2d(inShape5D[0], inShape5D[1], [&](size_t i0, size_t i1) {
                int64_t bIdx = i0 / outShape5D[0];
                size_t srcIdx1 = i0 * inBatchStep + i1 * inSpatialStep;
                size_t dstIdx1 = (i0 - (bIdx * outShape5D[0])) * outBatchStep + i1 * outSpatialStep;
                int64_t owAdd = bIdx % _block_shape[dimsSize - 1] - cropLeft;
                bIdx /= _block_shape[dimsSize - 1];
                int64_t ohAdd = (layout == NCDHW ? bIdx % _block_shape[dimsSize - 2] : bIdx) - cropTop;
                bIdx /= _block_shape[dimsSize - 2];
                int64_t odAdd = layout == NCDHW ? bIdx % _block_shape[dimsSize - 3] - cropFront : 0;
                size_t i2Begin = layout == NCDHW ? (_block_shape[dimsSize - 3] - 1 - odAdd) / _block_shape[dimsSize - 3] : 0lu;
                size_t i2End = layout == NCDHW ? (outShape5D[dimsSize - 3] - 1 - odAdd) / _block_shape[dimsSize - 3] + 1 : 1lu;
                for (size_t i2 = i2Begin; i2 < i2End; ++i2) {
                    size_t tmpOd = i2 * _block_shape[dimsSize - 3];
                    size_t srcIdx2 = srcIdx1 + i2 * inShape5D[dimsSize - 2] * inShape5D[dimsSize - 1];
                    size_t dstIdx2 = dstIdx1 + (tmpOd + odAdd) * outShape5D[dimsSize - 2] * outShape5D[dimsSize - 1];
                    int64_t i3Begin = (_block_shape[dimsSize - 2] - 1 - ohAdd) / _block_shape[dimsSize - 2];
                    int64_t i3End = (outShape5D[dimsSize - 2] - 1 - ohAdd) / _block_shape[dimsSize - 2] + 1;
                    for (size_t i3 = i3Begin; i3 < i3End; ++i3) {
                        size_t tmpOh = i3 * _block_shape[dimsSize - 2];
                        size_t srcIdx3 = srcIdx2 + i3 * inShape5D[dimsSize - 1];
                        size_t dstIdx3 = dstIdx2 + (tmpOh + ohAdd) * outShape5D[dimsSize - 1];
                        int64_t i4Begin = (_block_shape[dimsSize - 1] - 1 - owAdd) / _block_shape[dimsSize - 1];
                        int64_t i4End = (outShape5D[dimsSize - 1] - 1 - owAdd) / _block_shape[dimsSize - 1] + 1;
                        for (size_t i4 = i4Begin; i4 < i4End; ++i4) {
                            size_t tmpOw = i4 * _block_shape[dimsSize - 1];
                            size_t srcIdx4 = srcIdx3 + i4;
                            size_t dstIdx4 = dstIdx3 + tmpOw + owAdd;
                            dst_data[dstIdx4] = src_data[srcIdx4];
                        }
                    }
                }
            });
        }
    }
    SizeVector inDims;
    SizeVector outDims;
    std::vector<size_t> _block_shape;
    std::vector<size_t> _crops_begin;
};

REG_FACTORY_FOR(BatchToSpaceImpl, BatchToSpace);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine

