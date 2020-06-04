// Copyright (C) 2018-2020 Intel Corporation
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

class DepthToSpaceImpl: public ExtLayerBase {
    enum class DepthToSpaceMode {
        BLOCKS_FIRST,
        DEPTH_FIRST
    };

public:
    explicit DepthToSpaceImpl(const CNNLayer* layer) {
        try {
            if (layer->insData.empty() || layer->outData.empty())
                THROW_IE_EXCEPTION << "DepthToSpace layer with name '" << layer->name << "' has incorrect number of input/output edges";

            inDims = layer->insData[0].lock()->getTensorDesc().getDims();
            if (inDims.size() < 3)
                THROW_IE_EXCEPTION << "DepthToSpace layer with name '" << layer->name << "' has incorrect number of input dimensions";

            if (inDims.size() > 5)
                THROW_IE_EXCEPTION << "DepthToSpace layer with name '" << layer->name << "' doesn't support dimensions with rank greater than 5";

            SizeVector outDims = layer->outData[0]->getTensorDesc().getDims();
            if (inDims.size() != outDims.size())
                THROW_IE_EXCEPTION << "DepthToSpace layer with name '" << layer->name << "' has incorrect number of input/output dimensions";

            std::string modeString = layer->GetParamAsString("mode");
            if (modeString == "blocks_first") {
                mode = DepthToSpaceMode::BLOCKS_FIRST;
            } else if (modeString == "depth_first") {
                mode = DepthToSpaceMode::DEPTH_FIRST;
            } else {
                THROW_IE_EXCEPTION << "DepthToSpace layer with name '" << layer->name << "' doesn't support mode: " << modeString;
            }

            blockSize = layer->GetParamAsUInt("block_size", 1);
            if (blockSize == 0)
                THROW_IE_EXCEPTION << layer->name << " Incorrect blockSize parameter is zero!";

            size_t numSpatialDims = inDims.size() - 2;
            blockStep = static_cast<size_t>(std::pow(blockSize, numSpatialDims));
            if (inDims[1] % blockStep)
                THROW_IE_EXCEPTION << "DepthToSpace layer with name '" << layer->name <<
                    "' has block_size parameter which is incompatible with input tensor channels dimension size";

            if (inDims[1] / blockStep != outDims[1])
                THROW_IE_EXCEPTION << "DepthToSpace layer with name '" << layer->name << " has incompatible input/output channels";

            for (int i = 0; i < numSpatialDims; i++) {
                if (inDims[i + 2] * blockSize != outDims[i + 2])
                    THROW_IE_EXCEPTION << "DepthToSpace layer with name '" << layer->name << " has incompatible spatial dims";
            }

            auto computePrc = layer->insData[0].lock()->getTensorDesc().getPrecision();
            const std::set<size_t> supported_precision_sizes = {1, 2, 4, 8};
            if (supported_precision_sizes.find(computePrc.size()) == supported_precision_sizes.end())
                THROW_IE_EXCEPTION << "DepthToSpace layer with name '" << layer->name << " doesn't support precision: " << computePrc.name();


            if (inDims.size() == 4 || inDims.size() == 5) {
                LayerConfig config;
                DataConfig inConfig;
                inConfig.desc = TensorDesc(computePrc, inDims, inDims.size() == 4 ? NHWC : NDHWC);
                config.inConfs.push_back(inConfig);

                DataConfig outConfig;
                outConfig.desc = TensorDesc(computePrc, outDims, outDims.size() == 4 ? NHWC : NDHWC);
                config.outConfs.push_back(outConfig);

                config.dynBatchSupport = false;
                confs.push_back(config);
            }

            LayerConfig config;
            DataConfig inConfig;
            inConfig.desc = TensorDesc(computePrc, inDims, InferenceEngine::TensorDesc::getLayoutByDims(inDims));
            config.inConfs.push_back(inConfig);

            DataConfig outConfig;
            outConfig.desc = TensorDesc(computePrc, outDims, InferenceEngine::TensorDesc::getLayoutByDims(outDims));
            config.outConfs.push_back(outConfig);

            config.dynBatchSupport = false;
            confs.push_back(config);
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
        switch (inputs[0]->getTensorDesc().getPrecision().size()) {
            case 1: depthToSpaceKernel<PrecisionTrait<Precision::U8>::value_type>(inputs, outputs); break;
            case 2: depthToSpaceKernel<PrecisionTrait<Precision::U16>::value_type>(inputs, outputs); break;
            case 4: depthToSpaceKernel<PrecisionTrait<Precision::I32>::value_type>(inputs, outputs); break;
            case 8: depthToSpaceKernel<PrecisionTrait<Precision::U64>::value_type>(inputs, outputs); break;
            default: {
                if (resp) {
                    std::string errorMsg = "DepthToSpace layer with name does not support precision '"
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

    std::vector<size_t> getBlock3D(const SizeVector& shape) {
        std::vector<size_t> block3D(3, 1);
        for (int i = 0; i < shape.size() - 2; i++) {
            block3D[i] = blockSize;
        }
        return block3D;
    }

    template<typename T>
    void depthToSpaceKernel(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs) {
        const T *src_data = inputs[0]->cbuffer().as<const T *>() + inputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        T* dst_data = outputs[0]->buffer().as<T *>() + outputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        auto shape5D = getShape5D(inDims);
        auto block3D = getBlock3D(inDims);

        size_t spatialStep = shape5D[2] * shape5D[3] * shape5D[4];
        size_t batchStep = shape5D[1] * spatialStep;

        size_t srcChannels = shape5D[1];
        size_t dstChannels = srcChannels / blockStep;

        size_t blockShift = mode == DepthToSpaceMode::BLOCKS_FIRST ? (dstChannels) : 1;
        size_t channelShift = mode == DepthToSpaceMode::BLOCKS_FIRST ? 1 : blockStep;

        if (inputs[0]->getTensorDesc().getLayout() == NHWC || inputs[0]->getTensorDesc().getLayout() == NDHWC) {
            parallel_for2d(shape5D[0], shape5D[2], [&](size_t i0, size_t i2) {
                size_t srcIdx1 = i0 * batchStep;
                size_t dstIdx1 = i0 * batchStep;
                for (size_t b2 = 0; b2 < block3D[0]; b2++) {
                    size_t srcIdx2 = srcIdx1 + i2 * shape5D[3] * shape5D[4] * srcChannels + b2 * block3D[1] * block3D[2] * blockShift;
                    size_t dstIdx2 = dstIdx1 + (i2 * block3D[0] + b2) * shape5D[3] * block3D[1] * shape5D[4] * block3D[2] * dstChannels;
                    for (size_t i3 = 0; i3 < shape5D[3]; i3++) {
                        for (size_t b3 = 0; b3 < block3D[1]; b3++) {
                            size_t srcIdx3 = srcIdx2 + i3 * shape5D[4] * srcChannels + b3 * block3D[2] * blockShift;
                            size_t dstIdx3 = dstIdx2 + (i3 * block3D[1] + b3) * shape5D[4] * block3D[2] * dstChannels;
                            for (size_t i4 = 0; i4 < shape5D[4]; i4++) {
                                for (size_t b4 = 0; b4 < block3D[2]; b4++) {
                                    size_t srcIdx4 = srcIdx3 + i4 * srcChannels + b4 * blockShift;
                                    size_t dstIdx4 = dstIdx3 + (i4 * block3D[2] + b4) * dstChannels;
                                    for (size_t i1 = 0; i1 < dstChannels; i1++) {
                                        size_t srcIdx5 = srcIdx4 + i1 * channelShift;
                                        size_t dstIdx5 = dstIdx4 + i1;
                                        dst_data[dstIdx5] = src_data[srcIdx5];
                                    }
                                }
                            }
                        }
                    }
                }
            });
        } else {
            parallel_for2d(shape5D[0], dstChannels, [&](size_t i0, size_t i1) {
                size_t srcIdx1 = i0 * batchStep + i1 * channelShift * spatialStep;
                size_t dstIdx1 = i0 * batchStep + i1 * blockStep * spatialStep;
                for (size_t i2 = 0; i2 < shape5D[2]; i2++) {
                    for (size_t b2 = 0; b2 < block3D[0]; b2++) {
                        size_t srcIdx2 = srcIdx1 + i2 * shape5D[3] * shape5D[4] + b2 * block3D[1] * block3D[2] * blockShift * spatialStep;
                        size_t dstIdx2 = dstIdx1 + (i2 * block3D[0] + b2) * shape5D[3] * block3D[1] * shape5D[4] * block3D[2];
                        for (size_t i3 = 0; i3 < shape5D[3]; i3++) {
                            for (size_t b3 = 0; b3 < block3D[1]; b3++) {
                                size_t srcIdx3 = srcIdx2 + i3 * shape5D[4] + b3 * block3D[2] * blockShift * spatialStep;
                                size_t dstIdx3 = dstIdx2 + (i3 * block3D[1] + b3) * shape5D[4] * block3D[2];
                                for (size_t i4 = 0; i4 < shape5D[4]; i4++) {
                                    for (size_t b4 = 0; b4 < block3D[2]; b4++) {
                                        size_t srcIdx4 = srcIdx3 + i4 + b4 * blockShift * spatialStep;
                                        size_t dstIdx4 = dstIdx3 + i4 * block3D[2] + b4;
                                        dst_data[dstIdx4] = src_data[srcIdx4];
                                    }
                                }
                            }
                        }
                    }
                }
            });
        }
    }

    DepthToSpaceMode mode;
    SizeVector inDims;
    size_t blockSize;
    size_t blockStep;
};

REG_FACTORY_FOR(DepthToSpaceImpl, DepthToSpace);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
