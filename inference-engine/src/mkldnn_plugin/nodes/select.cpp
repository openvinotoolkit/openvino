// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "list.hpp"
#include "base.hpp"

#include <cmath>
#include <string>
#include <vector>
#include <cassert>
#include <algorithm>
#include "ie_parallel.hpp"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class SelectImpl: public ExtLayerBase {
    enum {condition, then_, else_, numOfInputs};

public:
    explicit SelectImpl(const CNNLayer* layer) {
        try {
            if (numOfInputs != layer->insData.size() || 1 != layer->outData.size()) {
                THROW_IE_EXCEPTION << layer->name << " Incorrect number of input/output edges!";
            }

            auto conditionPrecision = layer->insData[condition].lock()->getTensorDesc().getPrecision();

            if (Precision::I32 != conditionPrecision
                && Precision::FP32 != conditionPrecision
                && Precision::U8 != conditionPrecision) {
                THROW_IE_EXCEPTION << layer->name << " Incorrect condition tensor precision: " << conditionPrecision
                << ". Should be I32, U8 or FP32";
            }

            addConfig(layer, {{ConfLayout::PLN, false},
                              {ConfLayout::PLN, false},
                              {ConfLayout::PLN, false}},
                             {{ConfLayout::PLN, false}});
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    template <typename COND_T, typename DATA_T>
    void execute_impl(std::vector<Blob::Ptr>& inputs, Blob::Ptr& output) noexcept {
        auto *conditionData = inputs[condition]->cbuffer().as<const COND_T*>();
        auto *thenData = inputs[then_]->cbuffer().as<const DATA_T*>();
        auto *elseData = inputs[else_]->cbuffer().as<const DATA_T*>();

        auto *dstData = output->cbuffer().as<DATA_T *>();
        enum {N, C, H, W, Dims};
        int dim[Dims] = {1, 1, 1, 1};
        int cdim[Dims] = {1, 1, 1, 1};

        SizeVector dims = inputs[then_]->getTensorDesc().getDims();
        std::copy(std::begin(dims), std::end(dims), std::begin(dim) + (Dims - dims.size()));

        SizeVector cDims = inputs[condition]->getTensorDesc().getDims();
        std::copy(std::begin(cDims), std::end(cDims), std::begin(cdim) + (Dims - cDims.size()));

        parallel_for3d(dim[N], dim[H], dim[W], [&](int b, int h, int w) {
            for (int c = 0; c < dim[C]; c++) {
                dstData[b*dim[C]*dim[H]*dim[W] + c*dim[H]*dim[W] + h*dim[W] + w]
                        = conditionData[(b % cdim[N])*cdim[C]*cdim[H]*cdim[W] +
                                        (c % cdim[C])*cdim[H]*cdim[W] +
                                        (h % cdim[H])*cdim[W] +
                                        (w % cdim[W])]
                          ?      thenData[b*dim[C]*dim[H]*dim[W] + c*dim[H]*dim[W] + h*dim[W] + w]
                          :      elseData[b*dim[C]*dim[H]*dim[W] + c*dim[H]*dim[W] + h*dim[W] + w];
            }
        });
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
        auto &outputData = outputs[0];

        auto cond_precision = inputs[condition]->getTensorDesc().getPrecision();
        auto data_precision = inputs[then_]->getTensorDesc().getPrecision();

        auto compare = getPrecisionMask(cond_precision, data_precision);
        switch (compare) {
            /* 64 bit data type */
            case getPrecisionMask(Precision::I32, Precision::I64):
                execute_impl<int32_t, int64_t>(inputs, outputData);
                break;
            case getPrecisionMask(Precision::U8, Precision::I64):
                execute_impl<uint8_t, int64_t>(inputs, outputData);
                break;
            case getPrecisionMask(Precision::I32, Precision::U64):
                execute_impl<int32_t, uint64_t>(inputs, outputData);
                break;
            case getPrecisionMask(Precision::U8, Precision::U64):
                execute_impl<uint8_t , uint64_t>(inputs, outputData);
                break;

            /* 32 bit data type */
            case getPrecisionMask(Precision::I32, Precision::FP32):
            case getPrecisionMask(Precision::I32, Precision::I32):
                execute_impl<int32_t , int32_t>(inputs, outputData);
                break;
            case getPrecisionMask(Precision::U8, Precision::FP32):
            case getPrecisionMask(Precision::U8, Precision::I32):
                execute_impl<uint8_t , int32_t>(inputs, outputData);
                break;

            /* 16 bit data type */
            case getPrecisionMask(Precision::I32, Precision::FP16):
            case getPrecisionMask(Precision::I32, Precision::Q78):
            case getPrecisionMask(Precision::I32, Precision::I16):
            case getPrecisionMask(Precision::I32, Precision::U16):
                execute_impl<int32_t , int16_t>(inputs, outputData);
                break;
            case getPrecisionMask(Precision::U8, Precision::FP16):
            case getPrecisionMask(Precision::U8, Precision::Q78):
            case getPrecisionMask(Precision::U8, Precision::I16):
            case getPrecisionMask(Precision::U8, Precision::U16):
                execute_impl<uint8_t , int16_t>(inputs, outputData);
                break;

            /* 8 bit data type */
            case getPrecisionMask(Precision::I32, Precision::I8):
            case getPrecisionMask(Precision::I32, Precision::U8):
                execute_impl<int32_t , int8_t>(inputs, outputData);
                break;
            case getPrecisionMask(Precision::U8, Precision::I8):
            case getPrecisionMask(Precision::U8, Precision::U8):
                execute_impl<uint8_t , int8_t>(inputs, outputData);
                break;

            default:
                if (resp) {
                    std::string errorMsg = "Incorrect Reduce layer type";
                    errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
                }
                return GENERAL_ERROR;
        }


        return OK;
    }
};


REG_FACTORY_FOR(ImplFactory<SelectImpl>, Select);
}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
