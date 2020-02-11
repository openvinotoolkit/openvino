// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <details/ie_exception.hpp>
#include <ie_layers_property.hpp>
#include <ie_blob.h>

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
};

struct eltwise_common_params {
    std::string operation;
    std::vector<float> coeff;
};

struct def_conv_common_params : conv_common_params {
    def_conv_common_params(conv_common_params convCommonParams, size_t deformable_group) :
            def_conv_common_params(convCommonParams) {
        this->deformable_group = deformable_group;
    }

    def_conv_common_params(conv_common_params convCommonParams) {
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

struct Statistic {
    std::vector<float> min;
    std::vector<float> max;

    bool empty() const {
        return min.empty() || max.empty();
    }

    std::string serialize_min() const {
        return serialize(min);
    }
    std::string serialize_max() const {
        return serialize(max);
    }

protected:
    std::string serialize(const std::vector<float>& in) const {
        if (in.empty())
            return "";
        std::string out = std::to_string(in[0lu]);
        for (size_t i = 1lu; i < in.size(); i++)
            out += ", " + std::to_string(in[i]);
        return out;
    }
};

void getConvOutShape(const std::vector<size_t>& inShape,
                    const conv_common_params& params,
                    std::vector<size_t>& outShape);

void getPoolOutShape(const std::vector<size_t>& inShape,
                    const pool_common_params& params,
                    std::vector<size_t>& outShape);

size_t getConvWeightsSize(const std::vector<size_t>& inShape,
                    const conv_common_params& params,
                    const std::string& precision);

size_t getConvBiasesSize(const conv_common_params& params,
                    const std::string& precision);

InferenceEngine::TBlob<uint8_t>::Ptr getWeightsBlob(size_t size,
                                    const std::string& precision);

InferenceEngine::TBlob<uint8_t>::Ptr getConvWeightsBlob(const std::vector<size_t>& inShape,
                    const conv_common_params& params);

void fillStatistic(Statistic& out, size_t size, float min, float max);
