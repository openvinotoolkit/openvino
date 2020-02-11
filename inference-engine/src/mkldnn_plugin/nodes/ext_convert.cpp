// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ext_list.hpp"
#include "ext_base.hpp"

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

            addConfig(layer, {{ConfLayout::PLN, false, 0}}, {{ConfLayout::PLN, false, 0}});
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        try {
            auto compare = getPrecisionMask(inputs[0]->getTensorDesc().getPrecision(), outputs[0]->getTensorDesc().getPrecision());
            switch (compare) {
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
                default:
                    THROW_IE_EXCEPTION << "Unsupported precisions!";
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
