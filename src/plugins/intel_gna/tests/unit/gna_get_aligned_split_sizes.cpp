// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <vector>
// to suppress deprecated definition errors
#define IMPLEMENT_INFERENCE_ENGINE_PLUGIN
#include "common/gna_target.hpp"
#include "layers/gna_split_layer.hpp"
#include "ngraph/opsets/opset9.hpp"

using namespace ov::intel_gna::limitations;
using namespace ov::intel_gna::target;

namespace {

using GetAlignedSplitSizesData = std::tuple<uint32_t,              // total size
                                            uint32_t,              // maximum split size
                                            uint32_t,              // alignment
                                            std::vector<uint32_t>  // expected sizes
                                            >;

const std::vector<GetAlignedSplitSizesData> data = {
    GetAlignedSplitSizesData{10, 100, 64, std::vector<uint32_t>{10}},
    GetAlignedSplitSizesData{1024, 100, 64, std::vector<uint32_t>(16, 64)},
    GetAlignedSplitSizesData{151, 100, 64, std::vector<uint32_t>{64, 64, 23}},
    GetAlignedSplitSizesData{151, 65, 32, std::vector<uint32_t>{64, 64, 23}},
    GetAlignedSplitSizesData{151, 33, 32, std::vector<uint32_t>{32, 32, 32, 32, 23}},
    GetAlignedSplitSizesData{151, 17, 16, std::vector<uint32_t>{16, 16, 16, 16, 16, 16, 16, 16, 16, 7}},
    GetAlignedSplitSizesData{151, 65, 1, std::vector<uint32_t>{65, 65, 21}},
    GetAlignedSplitSizesData{67000, 65528, 64, std::vector<uint32_t>{65472, 1528}},
    GetAlignedSplitSizesData{67000, 65528, 16, std::vector<uint32_t>{65520, 1480}}};

TEST(GetAlignedSplitSizesTest, testAlignedSplitSizes) {
    for (const auto& dataItem : data) {
        auto sizes =
            ov::intel_gna::GetAlignedSplitSizes(std::get<0>(dataItem), std::get<1>(dataItem), std::get<2>(dataItem));
        ASSERT_EQ(sizes, std::get<3>(dataItem));
    }
}

using VariadicSplitParameters = std::tuple<ov::Shape,             // input size
                                           uint32_t,              // axis
                                           std::vector<int32_t>,  // split lengths
                                           bool                   // supported
                                           >;

void RunVariadicSplitSupportedTest(DeviceVersion device_version, std::vector<VariadicSplitParameters> test_vectors) {
    ov::Shape input_shape;
    uint32_t axis;
    std::vector<int32_t> split_lengths;
    bool result;

    Limitations::init(device_version);
    for (const auto& item : test_vectors) {
        std::tie(input_shape, axis, split_lengths, result) = item;

        auto split = std::make_shared<ngraph::opset9::VariadicSplit>(
            std::make_shared<ngraph::opset9::Parameter>(ngraph::element::f32, input_shape),
            ngraph::opset9::Constant::create(ngraph::element::i64, ngraph::Shape({1}), {axis}),
            ngraph::opset9::Constant::create(ngraph::element::i64,
                                             ngraph::Shape({split_lengths.size()}),
                                             split_lengths));
        ASSERT_TRUE(Limitations::is_split_supported(split, false) == result);
    }
}

TEST(CheckSplitSupported, CheckVariadicSplitSupported_GNA3_5) {
    RunVariadicSplitSupportedTest(
        DeviceVersion::GNA3_5,
        {VariadicSplitParameters{ov::Shape{1024}, 0, std::vector<int32_t>{192, 192, 320, 320}, true},
         VariadicSplitParameters{ov::Shape{1, 1024}, 1, std::vector<int32_t>{640, 192, 192}, true},
         VariadicSplitParameters{ov::Shape{1024}, 0, std::vector<int32_t>{16, 1008}, false},
         VariadicSplitParameters{ov::Shape{1024}, 0, std::vector<int32_t>{500, 24, 500}, false},
         VariadicSplitParameters{ov::Shape{1, 1024}, 1, std::vector<int32_t>{700, 300, 24}, false}});
}

TEST(CheckSplitSupported, CheckVariadicSplitSupported_GNA3_6) {
    RunVariadicSplitSupportedTest(
        DeviceVersion::GNA3_6,
        {VariadicSplitParameters{ov::Shape{1024}, 0, std::vector<int32_t>{192, 192, 320, 320}, true},
         VariadicSplitParameters{ov::Shape{1, 1024}, 1, std::vector<int32_t>{640, 192, 192}, true},
         VariadicSplitParameters{ov::Shape{1024}, 0, std::vector<int32_t>{16, 1008}, true},
         VariadicSplitParameters{ov::Shape{1024}, 0, std::vector<int32_t>{500, 24, 500}, false},
         VariadicSplitParameters{ov::Shape{1, 1024}, 1, std::vector<int32_t>{700, 300, 24}, false}});
}

using SplitParameters = std::tuple<ov::Shape,  // input size
                                   uint32_t,   // axis
                                   uint32_t,   // num_splits
                                   bool        // supported
                                   >;

void RunSplitSupportedTest(DeviceVersion device_version, std::vector<SplitParameters> test_vectors) {
    ov::Shape input_shape;
    uint32_t axis;
    uint32_t num_splits;
    bool result;

    Limitations::init(device_version);
    for (const auto& item : test_vectors) {
        std::tie(input_shape, axis, num_splits, result) = item;
        auto split = std::make_shared<ngraph::opset9::Split>(
            std::make_shared<ngraph::opset9::Parameter>(ngraph::element::f32, input_shape),
            ngraph::opset9::Constant::create(ngraph::element::i64, ngraph::Shape({}), {axis}),
            num_splits);
        ASSERT_TRUE(Limitations::is_split_supported(split, false) == result);
    }
}

TEST(CheckSplitSupported, CheckSplitSupported_GNA3_5) {
    RunSplitSupportedTest(DeviceVersion::GNA3_5,
                          {
                              SplitParameters{ov::Shape{1024}, 0, 4, true},
                              SplitParameters{ov::Shape{1, 1024}, 1, 16, true},
                              SplitParameters{ov::Shape{1024}, 0, 64, false},
                              SplitParameters{ov::Shape{1, 1024}, 1, 256, false},
                          });
}

TEST(CheckSplitSupported, CheckSplitSupported_GNA3_6) {
    RunSplitSupportedTest(DeviceVersion::GNA3_6,
                          {
                              SplitParameters{ov::Shape{1024}, 0, 4, true},
                              SplitParameters{ov::Shape{1, 1024}, 1, 16, true},
                              SplitParameters{ov::Shape{1024}, 0, 64, true},
                              SplitParameters{ov::Shape{1, 1024}, 1, 256, false},
                          });
}
}  // namespace
