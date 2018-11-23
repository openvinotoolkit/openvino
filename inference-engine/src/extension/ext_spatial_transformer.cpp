// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "ext_list.hpp"
#include "ext_base.hpp"

#include "matrixmult.h"

#include <algorithm>
#include <vector>
#include <cmath>
#include <map>
#include <string>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class SpatialTransformerImpl: public ExtLayerBase {
public:
    explicit SpatialTransformerImpl(const CNNLayer* layer) {
        try {
            if (layer->insData.size() != 2 || layer->outData.empty())
                THROW_IE_EXCEPTION << "Incorrect number of input/output edges!";

            addConfig(layer, {DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN)}, {DataConfigurator(ConfLayout::PLN)});
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        std::vector<size_t> real_dims = inputs[0]->getTensorDesc().getDims();
        size_t data_size = inputs[0]->size();

        const auto *src_data = inputs[0]->cbuffer().as<const float *>();
        auto *theta = inputs[1]->buffer().as<float *>();
        auto *dst_data = outputs[0]->buffer().as<float *>();

        auto N = real_dims[0];
        auto C = real_dims[1];
        auto output_H_ = real_dims[2];
        auto output_W_ = real_dims[3];

        // Prepare input and output grid
        std::vector<float> input_grid_data(N * output_H_ * output_W_ * 2);
        std::vector<float> output_grid_data(3 * output_H_ * output_W_);
        for (int i = 0; i < output_H_ * output_W_; ++i) {
            output_grid_data[3 * i] = (i / output_W_) * 1.0 / output_H_ * 2 - 1;
            output_grid_data[3 * i + 1] = (i % output_W_) * 1.0 / output_W_ * 2 - 1;
            output_grid_data[3 * i + 2] = 1;
        }

        // Actually execute
        for (int i = 0; i < N; ++i) {
            auto coordinates = input_grid_data.begin() + (output_H_ * output_W_ * 2) * i;

            auto M_size = output_H_ * output_W_;
            auto N_size = 2;
            auto K_size = 3;

            matrixMult(&output_grid_data[0], theta + 6 * i, &(*coordinates), M_size, N_size, K_size, true);

            int row_idx;
            float px, py;

            for (int j = 0; j < C; ++j) {
                for (int s = 0; s < output_H_; ++s) {
                    for (int t = 0; t < output_W_; ++t) {
                        row_idx = output_W_ * s + t;

                        px = coordinates[row_idx * 2];
                        py = coordinates[row_idx * 2 + 1];

                        size_t dst_offset = ((i * C + j) * output_H_ + s) * output_W_ + t;
                        size_t src_offset = ((i * C + j) * output_H_ + 0) * output_W_ + 0;
                        dst_data[dst_offset] = transform_forward_cpu(src_data + src_offset, px, py);
                    }
                }
            }
        }
        return OK;
    }

private:
    float transform_forward_cpu(const float *pic, float px, float py) {
        int H = 24;
        int W = 94;

        float res = 0.0f;
        float x = (px + 1) / 2 * H;
        float y = (py + 1) / 2 * W;

        int m, n;
        float w;

        m = std::floor(x);
        n = std::floor(y);
        w = 0;
        if (m >= 0 && m < H && n >= 0 && n < W) {
            w = std::max<float>(0.0f, 1 - std::abs(x - m)) * std::max<float>(0.0f, 1 - std::abs(y - n));
            res += w * pic[m * W + n];
        }

        m = std::floor(x) + 1;
        n = std::floor(y);
        w = 0;
        if (m >= 0 && m < H && n >= 0 && n < W) {
            w = std::max<float>(0.0f, 1 - std::abs(x - m)) * std::max<float>(0.0f, 1 - std::abs(y - n));
            res += w * pic[m * W + n];
        }

        m = std::floor(x);
        n = std::floor(y) + 1;
        w = 0;
        if (m >= 0 && m < H && n >= 0 && n < W) {
            w = std::max<float>(0.0f, 1 - std::abs(x - m)) * std::max<float>(0.0f, 1 - std::abs(y - n));
            res += w * pic[m * W + n];
        }

        m = std::floor(x) + 1;
        n = std::floor(y) + 1;
        w = 0;
        if (m >= 0 && m < H && n >= 0 && n < W) {
            w = std::max<float>(0.0f, 1 - std::abs(x - m)) * std::max<float>(0.0f, 1 - std::abs(y - n));
            res += w * pic[m * W + n];
        }

        return res;
    }
};

class SpatialTransformerShapeInfer : public IShapeInferImpl {
public:
    StatusCode inferShapes(const std::vector<SizeVector>& inShapes,
                           const std::map<std::string, std::string>& params,
                           const std::map<std::string, Blob::Ptr>& blobs,
                           std::vector<SizeVector>& outShapes,
                           ResponseDesc* resp) noexcept override {
        outShapes.push_back(inShapes[0]);
        return InferenceEngine::OK;
    }
};

REG_FACTORY_FOR(ImplFactory<SpatialTransformerImpl>, SpatialTransformer);
REG_SHAPE_INFER_FOR_TYPE(SpatialTransformerShapeInfer, SpatialTransformer);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
