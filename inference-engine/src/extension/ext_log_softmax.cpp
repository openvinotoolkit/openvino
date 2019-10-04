// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ext_list.hpp"
#include "ext_base.hpp"

#include <cmath>
#include <limits>
#include <cfloat>
#include <string>
#include <vector>
#include <cassert>
#include "ie_parallel.hpp"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class LogSoftmaxImpl: public ExtLayerBase {
public:
    explicit LogSoftmaxImpl(const CNNLayer* layer) {
        try {
            if (layer->insData.empty() || layer->outData.empty())
                THROW_IE_EXCEPTION << layer->name << " Incorrect number of input/output edges!";

            if (layer->insData.size() != 1)
                THROW_IE_EXCEPTION << layer->name << " Incorrect number of input edges!";

            if (layer->insData[0].lock()->getTensorDesc().getPrecision() != Precision::FP32)
                THROW_IE_EXCEPTION << layer->name << " Incorrect input data tensor precision. Only FP32 is supported!";

            SizeVector dims = layer->insData[0].lock()->getTensorDesc().getDims();
            if (!dims.size())
                dims = SizeVector(1, 1);
            int axis = layer->GetParamAsInt("axis", -1);
            if (axis < 0)
                axis += dims.size();

            if (dims.size() < static_cast<size_t>(1 + axis))
                THROW_IE_EXCEPTION << layer->name << " Incorrect input parameters dimensions and axis number!";

            int j;
            for (j = dims.size() - 1; j >= 0; j--) {
                if (dims[j] != 1) break;
            }
            if (j == axis) is_last_dim = true;

            for (int i = 0; i < axis; i++)
                axis_step *= dims[i];
            reduced_axis_size = dims[axis];
            for (size_t i = (axis + 1); i < dims.size(); i++)
                reduced_axis_stride *= dims[i];

            addConfig(layer, { { ConfLayout::PLN, false, 0 } }, { { ConfLayout::PLN, false, 0 } });
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
        const float *src_data = inputs[0]->cbuffer().as<float *>() +
            inputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        float* dst_data = outputs[0]->cbuffer().as<float *>() +
            outputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        if (is_last_dim) {
            parallel_for(axis_step, [&](size_t i) {
                float reduce_prod = 0.0f;
                const float *src_dataPtr = &src_data[i * reduced_axis_size];
                for (size_t j = 0; j < reduced_axis_size; ++j)
                    reduce_prod += expf(src_dataPtr[j]);
                reduce_prod = logf(reduce_prod);
                float *dst_dataPtr = reinterpret_cast<float*>(&dst_data[i * reduced_axis_size]);
                for (size_t j = 0; j < reduced_axis_size; ++j)
                    dst_dataPtr[j] = src_dataPtr[j] - reduce_prod;
            });
        } else {
            parallel_for2d(axis_step, reduced_axis_stride, [&](size_t k, size_t i) {
                float reduce_prod = 0.0f;
                const float *src_dataPtr = &src_data[k * reduced_axis_stride * reduced_axis_size + i];
                for (size_t j = 0; j < reduced_axis_size; ++j) {
                    reduce_prod += expf((*src_dataPtr));
                    src_dataPtr += reduced_axis_stride;
                }

                reduce_prod = logf(reduce_prod);
                src_dataPtr = &src_data[k * reduced_axis_stride * reduced_axis_size + i];
                float *dst_dataPtr = reinterpret_cast<float*>(&dst_data[k * reduced_axis_stride * reduced_axis_size + i]);
                for (size_t j = 0; j < reduced_axis_size; ++j) {
                    (*dst_dataPtr) = (*src_dataPtr) - reduce_prod;
                    src_dataPtr += reduced_axis_stride;
                    dst_dataPtr += reduced_axis_stride;
                }
            });
        }

        return OK;
    }

private:
    size_t reduced_axis_size;
    size_t reduced_axis_stride = 1;
    size_t axis_step = 1;
    bool is_last_dim = false;
};

REG_FACTORY_FOR(ImplFactory<LogSoftmaxImpl>, LogSoftmax);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
