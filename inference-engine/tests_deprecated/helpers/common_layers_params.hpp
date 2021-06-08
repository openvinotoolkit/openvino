// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <vector>

#include <debug.h>  // to allow putting vector into exception string stream
#include <legacy/ie_layers_property.hpp>
#include "ie_blob.h"

namespace CommonTestUtils {

struct conv_common_params {
    InferenceEngine::PropertyVector<unsigned int> stride;
    InferenceEngine::PropertyVector<unsigned int> kernel;
    InferenceEngine::PropertyVector<unsigned int> pads_begin;
    InferenceEngine::PropertyVector<unsigned int> pads_end;
    InferenceEngine::PropertyVector<unsigned int> dilation;
    std::string auto_pad;
    size_t group;
    size_t out_c;
    bool with_bias;
    bool with_weights;
    std::string quantization_level;
};

struct pool_common_params {
    InferenceEngine::PropertyVector<unsigned int> stride;
    InferenceEngine::PropertyVector<unsigned int> kernel;
    InferenceEngine::PropertyVector<unsigned int> pads_begin;
    InferenceEngine::PropertyVector<unsigned int> pads_end;
    std::string auto_pad;
    bool avg;
    bool exclude_pad;
    std::string rounding_type;
};

struct eltwise_common_params {
    std::string operation;
    std::vector<float> coeff;
};

struct def_conv_common_params : conv_common_params {
    explicit def_conv_common_params(conv_common_params convCommonParams) {
        this->stride = convCommonParams.stride;
        this->kernel = convCommonParams.kernel;
        this->pads_begin = convCommonParams.pads_begin;
        this->pads_end = convCommonParams.pads_end;
        this->dilation = convCommonParams.dilation;
        this->auto_pad = convCommonParams.auto_pad;
        this->group = convCommonParams.group;
        this->out_c = convCommonParams.out_c;
        this->with_bias = convCommonParams.with_bias;
    }

    size_t deformable_group;
};

void getConvOutShape(const std::vector<size_t> &inShape,
                     const conv_common_params &params,
                     std::vector<size_t> &outShape);

std::map<std::string, std::string> convertConvParamToMap(const conv_common_params &params);

void getPoolOutShape(const std::vector<size_t> &inShape,
                     const pool_common_params &params,
                     std::vector<size_t> &outShape);

std::map<std::string, std::string> convertPoolParamToMap(const pool_common_params &params);

size_t getConvWeightsSize(const std::vector<size_t> &inShape,
                          const conv_common_params &params,
                          const std::string &precision);

size_t getConvBiasesSize(const conv_common_params &params,
                         const std::string &precision);

InferenceEngine::Blob::Ptr getWeightsBlob(size_t sizeInBytes, const std::string &precision = "");

void get_common_dims(const InferenceEngine::Blob &blob,
                     int32_t &dimx,
                     int32_t &dimy,
                     int32_t &dimz);

void get_common_dims(const InferenceEngine::Blob &blob,
                     int32_t &dimx,
                     int32_t &dimy,
                     int32_t &dimz,
                     int32_t &dimn);


}  // namespace CommonTestUtils
