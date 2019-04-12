// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <string.h>
#include <ie_builders.hpp>

#include "builder_test.hpp"

using namespace testing;
using namespace InferenceEngine;

class ResampleLayerBuilderTest : public BuilderTestCommon {};

TEST_F(ResampleLayerBuilderTest, checkTypeParameter) {
    InferenceEngine::Builder::Layer ieLayer("Resample", "upsample");
    ieLayer.getParameters()["type"] = std::string("caffe.ResampleParameter.NEAREST");
    ieLayer.getParameters()["antialias"] = false;
    ieLayer.getParameters()["factor"] = 2.0f;
    ieLayer.getParameters()["width"] = 10;
    ieLayer.getParameters()["height"] = 10;

    ASSERT_EQ("Resample", ieLayer.getType());
    ASSERT_EQ("caffe.ResampleParameter.NEAREST", ieLayer.getParameters()["type"].as<std::string>());

    InferenceEngine::Builder::ResampleLayer resampleLayer("upsample");
    resampleLayer.setResampleType("caffe.ResampleParameter.NEAREST");
    resampleLayer.setAntialias(false);
    resampleLayer.setFactor(2);
    resampleLayer.setWidth(10);
    resampleLayer.setHeight(10);
    ASSERT_EQ("Resample", resampleLayer.getType());
    ASSERT_EQ("caffe.ResampleParameter.NEAREST", resampleLayer.getResampleType());
}