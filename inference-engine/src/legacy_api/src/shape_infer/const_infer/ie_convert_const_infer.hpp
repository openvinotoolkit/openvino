// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_blob.h>
#include <legacy/ie_layers.h>
#include <ie_memcpy.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ie_const_infer_impl.hpp"
#include "ie_parallel.hpp"
#include "ie_precision.hpp"

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Implementation of Const inference for Tile layer
 */
class ConvertConstInfer : public ConstInferImpl {
    template <typename src_d, typename dst_d>
    void exec_cast(const Blob::CPtr& inData, Blob::Ptr& outData) {
        const src_d* src_data =
            inData->cbuffer().as<src_d*>() + inData->getTensorDesc().getBlockingDesc().getOffsetPadding();
        dst_d* dst_data =
            outData->buffer().as<dst_d*>() + outData->getTensorDesc().getBlockingDesc().getOffsetPadding();
        if (inData->size() != outData->size())
            THROW_IE_EXCEPTION << " Convert constant inference error: Input and output buffers have different sizes! "
                                  "Input buffer size = `"
                               << inData->size() << "` output buffer size = `" << outData->size() << "`";
        parallel_for(inData->size(), [&](size_t i) {
            dst_data[i] = static_cast<dst_d>(src_data[i]);
        });
    }

    template<typename dst_d>
    void exec_from_fp16_cast(const Blob::CPtr &inData, Blob::Ptr &outData) {
        const ie_fp16 *src_data =
                inData->cbuffer().as<ie_fp16 *>() + inData->getTensorDesc().getBlockingDesc().getOffsetPadding();
        dst_d *dst_data =
                outData->buffer().as<dst_d *>() + outData->getTensorDesc().getBlockingDesc().getOffsetPadding();
        if (inData->size() != outData->size())
            THROW_IE_EXCEPTION << " Convert constant inference error: Input and output buffers have different sizes! "
                                  "Input buffer size = `"
                               << inData->size() << "` output buffer size = `" << outData->size() << "`";
        parallel_for(inData->size(), [&](size_t i) {
            dst_data[i] = static_cast<dst_d>(PrecisionUtils::f16tof32(src_data[i]));
        });
    }

    template<typename src_d>
    void exec_to_fp16_cast(const Blob::CPtr &inData, Blob::Ptr &outData) {
        const src_d* src_data =
                inData->cbuffer().as<src_d*>() + inData->getTensorDesc().getBlockingDesc().getOffsetPadding();
        ie_fp16* dst_data =
                outData->buffer().as<ie_fp16*>() + outData->getTensorDesc().getBlockingDesc().getOffsetPadding();
        if (inData->size() != outData->size())
            THROW_IE_EXCEPTION << " Convert constant inference error: Input and output buffers have different sizes! "
                                  "Input buffer size = `"
                               << inData->size() << "` output buffer size = `" << outData->size() << "`";
        parallel_for(inData->size(), [&](size_t i) {
            dst_data[i] = PrecisionUtils::f32tof16(static_cast<float>(src_data[i]));
        });
    }

public:
    explicit ConvertConstInfer(const std::string& type): ConstInferImpl(type) {}

    void inferImpl(const std::vector<Blob::CPtr>& inData, const std::map<std::string, std::string>& params,
                   const std::map<std::string, Blob::Ptr>& blobs, std::vector<Blob::Ptr>& outData) override {
        LayerParams lp {};
        ConcatLayer layer(lp);
        layer.params = params;
        _validator->parseParams(&layer);
        if (inData.size() != 1)
            THROW_IE_EXCEPTION << " Convert constant inference error: incorrect number of inputs! Expected 1, got "
                               << inData.size();
        if (outData.size() != 1)
            THROW_IE_EXCEPTION << " Convert constant inference error: incorrect number of outputs! Expected 1, got "
                               << outData.size();
        if (layer.params["precision"] != outData[0]->getTensorDesc().getPrecision().name())
            THROW_IE_EXCEPTION << " Convert constant inference error: layer `precision` parameter and actual output "
                                  "data precision mismatch! "
                                  "`precision`=\""
                               << layer.params["precision"] << "\", "
                               << "`output_data_precision`=\"" << outData[0]->getTensorDesc().getPrecision() << "\"";

        auto compare =
            getPrecisionMask(inData[0]->getTensorDesc().getPrecision(), outData[0]->getTensorDesc().getPrecision());
        switch (compare) {
        case getPrecisionMask(Precision::I32, Precision::I32):
            exec_cast<PrecisionTrait<Precision::I32>::value_type, PrecisionTrait<Precision::I32>::value_type>(
                inData[0], outData[0]);
            break;
        case getPrecisionMask(Precision::I64, Precision::I64):
            exec_cast<PrecisionTrait<Precision::I64>::value_type, PrecisionTrait<Precision::I64>::value_type>(
                inData[0], outData[0]);
            break;
        case getPrecisionMask(Precision::U64, Precision::U64):
            exec_cast<PrecisionTrait<Precision::U64>::value_type, PrecisionTrait<Precision::U64>::value_type>(
                inData[0], outData[0]);
            break;
        case getPrecisionMask(Precision::FP32, Precision::FP32):
            exec_cast<PrecisionTrait<Precision::FP32>::value_type, PrecisionTrait<Precision::FP32>::value_type>(
                inData[0], outData[0]);
            break;
        case getPrecisionMask(Precision::I32, Precision::I64):
            exec_cast<PrecisionTrait<Precision::I32>::value_type, PrecisionTrait<Precision::I64>::value_type>(
                inData[0], outData[0]);
            break;
        case getPrecisionMask(Precision::I32, Precision::U64):
            exec_cast<PrecisionTrait<Precision::I32>::value_type, PrecisionTrait<Precision::U64>::value_type>(
                inData[0], outData[0]);
            break;
        case getPrecisionMask(Precision::I32, Precision::FP32):
            exec_cast<PrecisionTrait<Precision::I32>::value_type, PrecisionTrait<Precision::FP32>::value_type>(
                inData[0], outData[0]);
            break;
        case getPrecisionMask(Precision::FP32, Precision::I32):
            exec_cast<PrecisionTrait<Precision::FP32>::value_type, PrecisionTrait<Precision::I32>::value_type>(
                inData[0], outData[0]);
            break;
        case getPrecisionMask(Precision::FP32, Precision::I64):
            exec_cast<PrecisionTrait<Precision::FP32>::value_type, PrecisionTrait<Precision::I64>::value_type>(
                inData[0], outData[0]);
            break;
        case getPrecisionMask(Precision::FP32, Precision::U64):
            exec_cast<PrecisionTrait<Precision::FP32>::value_type, PrecisionTrait<Precision::U64>::value_type>(
                inData[0], outData[0]);
            break;
        case getPrecisionMask(Precision::FP32, Precision::U8):
            exec_cast<PrecisionTrait<Precision::FP32>::value_type, PrecisionTrait<Precision::U8>::value_type>(
                    inData[0], outData[0]);
            break;
        case getPrecisionMask(Precision::FP32, Precision::BOOL):
            exec_cast<PrecisionTrait<Precision::FP32>::value_type, PrecisionTrait<Precision::BOOL>::value_type>(
                    inData[0], outData[0]);
            break;
        case getPrecisionMask(Precision::BOOL, Precision::BOOL):
            exec_cast<PrecisionTrait<Precision::BOOL>::value_type, PrecisionTrait<Precision::BOOL>::value_type>(
                    inData[0], outData[0]);
            break;
        case getPrecisionMask(Precision::FP16, Precision::FP32):
            exec_from_fp16_cast<PrecisionTrait<Precision::FP32>::value_type>(inData[0], outData[0]);
            break;
        case getPrecisionMask(Precision::FP16, Precision::I32):
            exec_from_fp16_cast<PrecisionTrait<Precision::I32>::value_type>(inData[0], outData[0]);
            break;
        case getPrecisionMask(Precision::FP16, Precision::I64):
            exec_from_fp16_cast<PrecisionTrait<Precision::I64>::value_type>(inData[0], outData[0]);
            break;
        case getPrecisionMask(Precision::FP16, Precision::U64):
            exec_from_fp16_cast<PrecisionTrait<Precision::U64>::value_type>(inData[0], outData[0]);
            break;
        case getPrecisionMask(Precision::FP16, Precision::U8):
            exec_from_fp16_cast<PrecisionTrait<Precision::U8>::value_type>(inData[0], outData[0]);
            break;
        case getPrecisionMask(Precision::FP16, Precision::BOOL):
            exec_from_fp16_cast<PrecisionTrait<Precision::BOOL>::value_type>(inData[0], outData[0]);
            break;
        case getPrecisionMask(Precision::FP32, Precision::FP16):
            exec_to_fp16_cast<PrecisionTrait<Precision::FP32>::value_type>(inData[0], outData[0]);
            break;
        default:
            THROW_IE_EXCEPTION << " Convert constant inference error: Unsupported precision configuration! "
                               << " Input precision: " << inData[0]->getTensorDesc().getPrecision()
                               << ", output precision: " << outData[0]->getTensorDesc().getPrecision();
        }
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
