// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "list.hpp"
#include "base.hpp"

#include <cmath>
#include <string>
#include <vector>
#include "ie_parallel.hpp"
#include "ie_precision.hpp"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class ConvertImpl: public ExtLayerBase {
    template<typename src_d, typename dst_d>
    void exec_cast(const Blob::CPtr& inputs, Blob::Ptr& outputs) {
        const src_d *src_data = inputs->cbuffer().as<src_d *>() +
                                inputs->getTensorDesc().getBlockingDesc().getOffsetPadding();
        dst_d* dst_data = outputs->buffer().as<dst_d *>() +
                          outputs->getTensorDesc().getBlockingDesc().getOffsetPadding();
        if (inputs->size() != outputs->size())
            THROW_IE_EXCEPTION << "Input and output buffers have different sizes!";
        parallel_for(inputs->size(), [&](size_t i) {
            dst_data[i] = static_cast<dst_d>(src_data[i]);
        });
    }

public:
    explicit ConvertImpl(const CNNLayer* layer) {
        try {
            if (layer->insData.size() != 1 || layer->outData.empty())
                THROW_IE_EXCEPTION << "Incorrect number of input/output edges!";

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

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        try {
            auto compare = getPrecisionMask(inputs[0]->getTensorDesc().getPrecision(), outputs[0]->getTensorDesc().getPrecision());
            switch (compare) {
                case getPrecisionMask(Precision::U8, Precision::FP32):
                    exec_cast<PrecisionTrait<Precision::U8>::value_type, PrecisionTrait<Precision::FP32>::value_type>(inputs[0], outputs[0]);
                    break;
                case getPrecisionMask(Precision::FP32, Precision::U8):
                    exec_cast<PrecisionTrait<Precision::FP32>::value_type, PrecisionTrait<Precision::U8>::value_type>(inputs[0], outputs[0]);
                    break;
                case getPrecisionMask(Precision::I16, Precision::FP32):
                    exec_cast<PrecisionTrait<Precision::I16>::value_type, PrecisionTrait<Precision::FP32>::value_type>(inputs[0], outputs[0]);
                    break;
                case getPrecisionMask(Precision::FP32, Precision::I16):
                    exec_cast<PrecisionTrait<Precision::FP32>::value_type, PrecisionTrait<Precision::I16>::value_type>(inputs[0], outputs[0]);
                    break;
                case getPrecisionMask(Precision::U16, Precision::FP32):
                    exec_cast<PrecisionTrait<Precision::U16>::value_type, PrecisionTrait<Precision::FP32>::value_type>(inputs[0], outputs[0]);
                    break;
                case getPrecisionMask(Precision::FP32, Precision::U16):
                    exec_cast<PrecisionTrait<Precision::FP32>::value_type, PrecisionTrait<Precision::U16>::value_type>(inputs[0], outputs[0]);
                    break;
                case getPrecisionMask(Precision::I32, Precision::I32):
                    exec_cast<PrecisionTrait<Precision::I32>::value_type, PrecisionTrait<Precision::I32>::value_type>(inputs[0], outputs[0]);
                    break;
                case getPrecisionMask(Precision::I64, Precision::I64):
                    exec_cast<PrecisionTrait<Precision::I64>::value_type, PrecisionTrait<Precision::I64>::value_type>(inputs[0], outputs[0]);
                    break;
                case getPrecisionMask(Precision::FP32, Precision::FP32):
                    exec_cast<PrecisionTrait<Precision::FP32>::value_type, PrecisionTrait<Precision::FP32>::value_type>(inputs[0], outputs[0]);
                    break;
                case getPrecisionMask(Precision::I32, Precision::I64):
                    exec_cast<PrecisionTrait<Precision::I32>::value_type, PrecisionTrait<Precision::I64>::value_type>(inputs[0], outputs[0]);
                    break;
                case getPrecisionMask(Precision::I32, Precision::FP32):
                    exec_cast<PrecisionTrait<Precision::I32>::value_type, PrecisionTrait<Precision::FP32>::value_type>(inputs[0], outputs[0]);
                    break;
                case getPrecisionMask(Precision::FP32, Precision::I32):
                    exec_cast<PrecisionTrait<Precision::FP32>::value_type, PrecisionTrait<Precision::I32>::value_type>(inputs[0], outputs[0]);
                    break;
                case getPrecisionMask(Precision::FP32, Precision::I64):
                    exec_cast<PrecisionTrait<Precision::FP32>::value_type, PrecisionTrait<Precision::I64>::value_type>(inputs[0], outputs[0]);
                    break;
                case getPrecisionMask(Precision::U8, Precision::I32):
                    exec_cast<PrecisionTrait<Precision::U8>::value_type, PrecisionTrait<Precision::I32>::value_type>(inputs[0], outputs[0]);
                    break;
                default:
                    std::string errorMsg = "Unsupported precisions!";
                    if (resp) {
                        errorMsg.copy(resp->msg, sizeof(resp->msg)-1);
                    }
                    THROW_IE_EXCEPTION << errorMsg;
            }
        } catch(...) {
            return GENERAL_ERROR;
        }
        return OK;
    }

private:
    std::string precision;
};

REG_FACTORY_FOR(ImplFactory<ConvertImpl>, Convert);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
