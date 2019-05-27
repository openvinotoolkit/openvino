// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <utility>

#include <utility>

// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>
#include <inference_engine/shape_infer/const_infer/ie_const_infer_holder.hpp>
#include "built_in_shape_infer_general_test.hpp"
#include "adult_test_utils.hpp"

namespace IE = InferenceEngine;

namespace ShapeInferTests {

class CommonTests : public ::testing::Test {
protected:
    ASITestBuilder assertThat();

protected:
    std::string type;
    InOutData inOutData;
};

class BasicTest
        : public CommonTests,
          public testing::WithParamInterface<std::tuple<LayerType, InOutDataParam>> {
protected:
    void SetUp() override;
};

class BlobTest
        : public CommonTests,
          public testing::WithParamInterface<std::tuple<LayerType, InOutDataParam, BlobsParam>> {
protected:
    void SetUp() override;

protected:
    FloatMap blobsParam;
};

class ParamsTest
        : public CommonTests,
          public testing::WithParamInterface<std::tuple<LayerType, InOutDataParam, MapParams>> {
protected:
    void SetUp() override;

protected:
    MapStrStr strParams;
};

class BasicAdultTest : public BasicTest {
};

class StridedSliceTest : public ParamsTest {
public:
    std::vector<IE::Precision> getPrecisions();
};

class FillTest : public BasicTest {
protected:
    std::vector<float> refGen(const InOutData& inOutData);
};

class RangeTest : public BasicTest {
protected:
    std::vector<float> refGen(const InOutData& inOutData);
};

}  // namespace ShapeInferTests
