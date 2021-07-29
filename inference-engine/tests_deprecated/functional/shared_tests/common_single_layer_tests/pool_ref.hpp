// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cfloat>
#include <ie_blob.h>
#include <gtest/gtest.h>
#include <legacy/ie_layers_internal.hpp>
#include "common_layers_params.hpp"

template<typename data_t>
void ref_pool_common(const std::vector<InferenceEngine::Blob::Ptr> srcs,
        InferenceEngine::Blob &dst,
        const CommonTestUtils::pool_common_params &p);

void Pool_parseParams(InferenceEngine::CNNLayer* layer);

template<typename data_t>
void common_ref_pool_wrap(const std::vector<InferenceEngine::Blob::Ptr> srcs, InferenceEngine::Blob::Ptr &dst,
                          const std::map<std::string, std::string> &params_map) {
    InferenceEngine::LayerParams lp{};
    InferenceEngine::PoolingLayer poolLayer(lp);
    auto data = std::make_shared<InferenceEngine::Data>("insData", srcs[0]->getTensorDesc());
    poolLayer.params = params_map;
    poolLayer.insData.push_back(data);
    Pool_parseParams(&poolLayer);

    CommonTestUtils::pool_common_params params;
    params.kernel = poolLayer._kernel;
    auto allPad = InferenceEngine::getPaddings(poolLayer);
    params.pads_begin = allPad.begin;
    params.pads_end = allPad.end;
    params.stride = poolLayer._stride;
    params.avg = poolLayer._type == InferenceEngine::PoolingLayer::PoolType::AVG;
    params.exclude_pad = poolLayer._exclude_pad;

    ref_pool_common<data_t>(srcs, *dst.get(), params);
}
