// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include <map>

#include "common_layers_params.hpp"
#include "common_test_utils/data_utils.hpp"

namespace CommonTestUtils {

void getConvOutShape(const std::vector<size_t> &inShape,
                     const conv_common_params &params,
                     std::vector<size_t> &outShape) {
    outShape.resize(inShape.size(), 1lu);
    outShape[0] = inShape[0];
    outShape[1] = params.out_c;
    for (int i = 0; i < params.kernel.size() && i + 2 < outShape.size(); i++) {
        outShape[i + 2] =
                (inShape[i + 2] + params.pads_begin[i] + params.pads_end[i] -
                 (params.kernel[i] - 1lu) * params.dilation[i] - 1lu) /
                params.stride[i] + 1lu;
    }
}

std::map<std::string, std::string> convertConvParamToMap(const conv_common_params &params) {
    std::map<std::string, std::string> resMap;

    auto propVecToStr = [](const InferenceEngine::PropertyVector<unsigned int> &vec) -> std::string {
        std::string str = "";
        for (size_t i = 0; i < vec.size(); i++) {
            str += std::to_string(vec[i]);
            if (i < vec.size() - 1) str += ",";
        }
        return str;
    };
    resMap["dilations"] = propVecToStr(params.dilation);
    resMap["kernel"] = propVecToStr(params.kernel);
    resMap["strides"] = propVecToStr(params.stride);
    resMap["pads_begin"] = propVecToStr(params.pads_begin);
    resMap["pads_end"] = propVecToStr(params.pads_end);
    resMap["group"] = std::to_string(params.group);
    resMap["output"] = std::to_string(params.out_c);
    if (!params.auto_pad.empty())
        resMap["auto_pad"] = params.auto_pad;
    if (!params.quantization_level.empty())
        resMap["quantization_level"] = params.quantization_level;

    return resMap;
}

void getPoolOutShape(const std::vector<size_t> &inShape,
                     const pool_common_params &params,
                     std::vector<size_t> &outShape) {
    outShape.resize(inShape.size(), 1lu);
    outShape[0] = inShape[0];
    outShape[1] = inShape[1];
    for (int i = 0; i < params.kernel.size() && i + 2 < outShape.size(); i++) {
        outShape[i + 2] =
                (inShape[i + 2] + params.pads_begin[i] + params.pads_end[i] - params.kernel[i]) / params.stride[i] +
                1lu;
    }
}

std::map<std::string, std::string> convertPoolParamToMap(const pool_common_params &params) {
    std::map<std::string, std::string> resMap;

    auto propVecToStr = [](const InferenceEngine::PropertyVector<unsigned int> &vec) -> std::string {
        std::string str = "";
        for (size_t i = 0; i < vec.size(); i++) {
            str += std::to_string(vec[i]);
            if (i < vec.size() - 1) str += ",";
        }
        return str;
    };
    resMap["kernel"] = propVecToStr(params.kernel);
    resMap["strides"] = propVecToStr(params.stride);
    resMap["pads_begin"] = propVecToStr(params.pads_begin);
    resMap["pads_end"] = propVecToStr(params.pads_end);
    if (!params.auto_pad.empty())
        resMap["auto_pad"] = params.auto_pad;
    if (!params.rounding_type.empty())
        resMap["rounding_type"] = params.rounding_type;

    if (params.avg) {
        resMap["pool-method"] = "avg";
        resMap["exclude-pad"] = params.exclude_pad ? "true" : "false";
    } else {
        resMap["pool-method"] = "max";
    }

    return resMap;
}

size_t getConvWeightsSize(const std::vector<size_t> &inShape,
                          const conv_common_params &params,
                          const std::string &precision) {
    if (!params.with_weights)
        return 0lu;
    size_t weights_size = 0lu;
    if (params.group != 0lu && params.out_c != 0lu && params.kernel.size() != 0lu) {
        auto prc = InferenceEngine::Precision::FromStr(precision);

        weights_size = prc.size() * inShape[1] * params.out_c / params.group;
        for (size_t i = 0lu; i < params.kernel.size(); i++) {
            weights_size *= params.kernel[i];
        }
    }
    return weights_size;
}

size_t getConvBiasesSize(const conv_common_params &params,
                         const std::string &precision) {
    if (!params.with_bias)
        return 0lu;
    auto prc = InferenceEngine::Precision::FromStr(precision);
    return prc.size() * params.out_c;
}

InferenceEngine::Blob::Ptr getWeightsBlob(size_t sizeInBytes, const std::string &precision) {
    InferenceEngine::Blob::Ptr weights;
    if (precision.empty()) {
        /* Just keep U8 blob for weights */
        weights = InferenceEngine::make_shared_blob<uint8_t>(
                {InferenceEngine::Precision::U8, {sizeInBytes}, InferenceEngine::C});
    } else {
        /* Keep blob for weights in original precision */
        if (precision == "U8") {
            using dataType = InferenceEngine::PrecisionTrait<InferenceEngine::Precision::U8>::value_type;
            weights = InferenceEngine::make_shared_blob<dataType>(
                    {InferenceEngine::Precision::U8, {sizeInBytes}, InferenceEngine::C});
        } else if (precision == "FP32") {
            using dataType = InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type;
            weights = InferenceEngine::make_shared_blob<dataType>(
                    {InferenceEngine::Precision::FP32, {sizeInBytes / sizeof(dataType)}, InferenceEngine::C});
        } else if (precision == "FP16" || precision == "Q78") {
            using dataType = InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP16>::value_type;
            weights = InferenceEngine::make_shared_blob<dataType>(
                    {InferenceEngine::Precision::FP16, {sizeInBytes / sizeof(dataType)}, InferenceEngine::C});
        } else {
            IE_THROW() << "Precision " << precision << " is not covered by getWeightsBlob()";
        }
    }

    weights->allocate();
    CommonTestUtils::fill_data(weights->buffer().as<float *>(), weights->byteSize() / sizeof(float));

    return weights;
}

void get_common_dims(const InferenceEngine::Blob &blob,
                     int32_t &dimx,
                     int32_t &dimy,
                     int32_t &dimz) {
    std::vector<int32_t> dims(blob.getTensorDesc().getDims().begin(), blob.getTensorDesc().getDims().end());
    if (dims.size() == 2) {
        dimz = 1;
        dimy = dims[0];
        dimx = dims[1];
    } else if (dims.size() == 3) {
        dimx = dims[2];
        dimy = dims[1];
        dimz = dims[0];
    } else if (dims.size() == 4 && dims[0] == 1) {
        dimx = dims[3];
        dimy = dims[2];
        dimz = dims[1];
    }
}

void get_common_dims(const InferenceEngine::Blob &blob,
                     int32_t &dimx,
                     int32_t &dimy,
                     int32_t &dimz,
                     int32_t &dimn) {
    std::vector<int32_t> dims(blob.getTensorDesc().getDims().begin(), blob.getTensorDesc().getDims().end());
    dimn = 1;
    if (dims.size() == 2) {
        dimz = 1;
        dimy = dims[0];
        dimx = dims[1];
    } else if (dims.size() == 3) {
        dimx = dims[2];
        dimy = dims[1];
        dimz = dims[0];
    } else if (dims.size() == 4) {
        dimx = dims[3];
        dimy = dims[2];
        dimz = dims[1];

        if (dims[0] != 1) {
            dimn = dims[0];
        }
    }
}

}  // namespace CommonTestUtils
