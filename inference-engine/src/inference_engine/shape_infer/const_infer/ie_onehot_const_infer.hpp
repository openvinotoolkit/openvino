// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_blob.h>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <ie_layers.h>
#include "ie_const_infer_impl.hpp"

namespace InferenceEngine {
namespace ShapeInfer {

/**
 * @brief Implementation of Const inference for OneHot layer
 */
class OneHotConstInfer : public ConstInferImpl {
public:
    explicit OneHotConstInfer(const std::string& type) : ConstInferImpl(type) {}

    void inferImpl(const std::vector<Blob::CPtr>& inData,
                   const std::map<std::string, std::string>& params,
                   const std::map<std::string, Blob::Ptr>& blobs,
                   std::vector<Blob::Ptr>& outData) override {
        OneHotLayer layer(LayerParams {});
        layer.params = params;
        layer.type = _type;
        _validator->parseParams(&layer);
        _validator->checkParams(&layer);
        auto src_dims = inData[0]->getTensorDesc().getDims();

        const auto *src_data = inData[0]->cbuffer().as<const float *>();
        auto *dst_data = outData[0]->buffer().as<float *>();
        std::size_t prefix_size = 1;
        auto input_dims = inData[0]->getTensorDesc().getDims();

        std::size_t actual_axis = (layer.axis == -1) ? src_dims.size() : layer.axis;
        for (size_t i = 0; i < actual_axis; ++i)
            prefix_size *= input_dims[i];

        std::size_t suffix_size = inData[0]->size() / prefix_size;

        std::size_t dst_offset = 0;
        for (std::size_t prefix_idx = 0; prefix_idx < prefix_size; ++prefix_idx) {
            for (std::size_t depth_idx = 0; depth_idx < layer.depth; ++depth_idx) {
                for (std::size_t suffix_idx = 0; suffix_idx < suffix_size; suffix_idx++) {
                    auto src_index = prefix_idx * suffix_size + suffix_idx;
                    std::size_t v = static_cast<std::size_t>(src_data[src_index]);
                    dst_data[dst_offset++] = (v == depth_idx) ? layer.on_value : layer.off_value;
                }
            }
        }
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
