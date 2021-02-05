// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"

#include <string>
#include <vector>
#include "ie_precision.hpp"
#include "common/cpu_convert.h"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class ConvertImpl: public ExtLayerBase {
public:
    explicit ConvertImpl(const CNNLayer* layer) {
        try {
            logPrefix = "Convert layer with name '" + layer->name + "' ";
            if (layer->insData.size() != 1 || layer->outData.size() != 1)
                THROW_IE_EXCEPTION << logPrefix << "has incorrect number of input/output edges";

            precision = layer->GetParamAsString("precision");

            LayerConfig config;
            DataConfig dataIn;
            const SizeVector& ins_dims = layer->insData[0].lock()->getTensorDesc().getDims();
            dataIn.desc = TensorDesc(layer->insData[0].lock()->getTensorDesc().getPrecision(), ins_dims,
                                     layer->insData[0].lock()->getTensorDesc().getLayout());
            config.inConfs.push_back(dataIn);

            DataConfig dataConfigOut;
            const SizeVector& out_dims = layer->outData[0]->getTensorDesc().getDims();
            dataConfigOut.desc = TensorDesc(layer->outData[0]->getTensorDesc().getPrecision(), out_dims,
                                            layer->outData[0]->getTensorDesc().getLayout());
            config.outConfs.push_back(dataConfigOut);
            config.dynBatchSupport = false;
            confs.push_back(config);
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
        try {
            void *srcPtr = inputs[0]->cbuffer().as<void *>();
            void *dstPtr = outputs[0]->buffer().as<void *>();
            if (inputs[0]->size() != outputs[0]->size())
                THROW_IE_EXCEPTION << logPrefix << "has input and output buffers with different sizes";
            cpu_convert(srcPtr, dstPtr, inputs[0]->getTensorDesc().getPrecision(), outputs[0]->getTensorDesc().getPrecision(), outputs[0]->size());
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
            if (resp)
                errorMsg.copy(resp->msg, sizeof(resp->msg)-1);
            return GENERAL_ERROR;
        } catch(...) {
            return GENERAL_ERROR;
        }
        return OK;
    }

private:
    std::string precision;
    std::string logPrefix;
};

REG_FACTORY_FOR(ConvertImpl, Convert);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
