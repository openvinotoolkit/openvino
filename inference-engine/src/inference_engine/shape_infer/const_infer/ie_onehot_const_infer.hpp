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
#include "precision_utils.h"

namespace InferenceEngine {
namespace ShapeInfer {

/**
 * @brief Implementation of Const inference for OneHot layer
 */
class OneHotConstInfer : public ConstInferImpl {
public:
    explicit OneHotConstInfer(const std::string& type) : ConstInferImpl(type) {}

    template <typename T>
    void inferImplBody(const std::vector<Blob::CPtr>& inData,
                       const std::map<std::string, std::string>& params,
                       std::vector<Blob::Ptr>& outData) {
        OneHotLayer layer(LayerParams {});
        layer.params = params;
        layer.type = _type;
        _validator->parseParams(&layer);
        _validator->checkParams(&layer);
        auto src_dims = inData[0]->getTensorDesc().getDims();

        const auto *src_data = inData[0]->cbuffer().as<const T*>();
        auto *dst_data = outData[0]->buffer().as<T*>();
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
                    auto v = static_cast<std::size_t>(src_data[src_index]);
                    dst_data[dst_offset++] = (v == depth_idx) ? layer.on_value : layer.off_value;
                }
            }
        }
    }

    void inferImplBody_fp16(const std::vector<Blob::CPtr>& inData,
                       const std::map<std::string, std::string>& params,
                       std::vector<Blob::Ptr>& outData) {
        OneHotLayer layer(LayerParams {});
        layer.params = params;
        layer.type = _type;
        _validator->parseParams(&layer);
        _validator->checkParams(&layer);
        auto src_dims = inData[0]->getTensorDesc().getDims();

        const auto *src_data = inData[0]->cbuffer().as<const int16_t *>();
        auto *dst_data = outData[0]->buffer().as<int16_t *>();
        std::size_t prefix_size = 1;
        auto input_dims = inData[0]->getTensorDesc().getDims();

        std::size_t actual_axis = (layer.axis == -1) ? src_dims.size() : layer.axis;
        for (size_t i = 0; i < actual_axis; ++i)
            prefix_size *= input_dims[i];

        std::size_t suffix_size = inData[0]->size() / prefix_size;

        int16_t val_on = PrecisionUtils::f32tof16(layer.on_value);
        int16_t val_off = PrecisionUtils::f32tof16(layer.off_value);

        std::size_t dst_offset = 0;
        for (std::size_t prefix_idx = 0; prefix_idx < prefix_size; ++prefix_idx) {
            for (std::size_t depth_idx = 0; depth_idx < layer.depth; ++depth_idx) {
                for (std::size_t suffix_idx = 0; suffix_idx < suffix_size; suffix_idx++) {
                    auto src_index = prefix_idx * suffix_size + suffix_idx;
                    auto v = static_cast<std::size_t>(src_data[src_index]);
                    dst_data[dst_offset++] = (v == depth_idx) ? val_on : val_off;
                }
            }
        }
    }

    void inferImpl(const std::vector<Blob::CPtr>& inData,
                   const std::map<std::string, std::string>& params,
                   const std::map<std::string, Blob::Ptr>& blobs,
                   std::vector<Blob::Ptr>& outData) override {
        auto inputBlob = inData.front();
        Precision precision = inputBlob->getTensorDesc().getPrecision();
        switch (precision) {
            case Precision::FP32: inferImplBody<PrecisionTrait<Precision::FP32>::value_type>(inData, params, outData); break;
            case Precision::FP16: inferImplBody_fp16(inData, params, outData); break;
            case Precision::Q78: inferImplBody<PrecisionTrait<Precision::Q78>::value_type>(inData, params, outData); break;
            case Precision::I16: inferImplBody<PrecisionTrait<Precision::I16>::value_type>(inData, params, outData); break;
            case Precision::U8: inferImplBody<PrecisionTrait<Precision::U8>::value_type>(inData, params, outData); break;
            case Precision::I8: inferImplBody<PrecisionTrait<Precision::I8>::value_type>(inData, params, outData); break;
            case Precision::U16: inferImplBody<PrecisionTrait<Precision::U16>::value_type>(inData, params, outData); break;
            case Precision::I32: inferImplBody<PrecisionTrait<Precision::I32>::value_type>(inData, params, outData); break;
            case Precision::I64: inferImplBody<PrecisionTrait<Precision::I64>::value_type>(inData, params, outData); break;
            default: THROW_IE_EXCEPTION << "OneHot const inference: Unsupported precision " << precision.name();
        }
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
