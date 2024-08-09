// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
typedef std::tuple<
        ov::op::util::InterpolateBase::InterpolateMode,          // InterpolateMode
        ov::op::util::InterpolateBase::ShapeCalcMode,            // ShapeCalculationMode
        ov::op::util::InterpolateBase::CoordinateTransformMode,  // CoordinateTransformMode
        ov::op::util::InterpolateBase::NearestMode,              // NearestMode
        bool,                                                    // AntiAlias
        std::vector<size_t>,                                     // PadBegin
        std::vector<size_t>,                                     // PadEnd
        double,                                                  // Cube coef
        std::vector<int64_t>,                                    // Axes
        std::vector<float>                                       // Scales
> InterpolateSpecificParams;

typedef std::tuple<
        InterpolateSpecificParams,
        ov::element::Type,                 // Model type
        std::vector<InputShape>,           // Input shapes
        ov::Shape,                         // Target shapes
        std::string,                       // Device name
        std::map<std::string, std::string> // Additional network configuration
> InterpolateLayerTestParams;

class InterpolateLayerTest : public testing::WithParamInterface<InterpolateLayerTestParams>,
                             virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<InterpolateLayerTestParams>& obj);

protected:
    void SetUp() override;
};

class Interpolate11LayerTest : public testing::WithParamInterface<InterpolateLayerTestParams>,
                             virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<InterpolateLayerTestParams>& obj);

protected:
    void SetUp() override;
};

class GPUInterpolateLayerTest : public InterpolateLayerTest {
protected:
    void SetUp() override {
        InterpolateLayerTest::SetUp();
        ov::test::InterpolateLayerTestParams params = GetParam();
        ov::test::InterpolateSpecificParams interpolate_params;
        ov::element::Type model_type;
        std::vector<ov::test::InputShape> shapes;
        ov::Shape target_shape;
        std::map<std::string, std::string> additional_config;
        std::tie(interpolate_params, model_type, shapes, target_shape, targetDevice, additional_config) = this->GetParam();
        // Some rounding float to integer types on GPU may differ from CPU, and as result,
        // the actual values may differ from reference ones on 1 when the float is very close to an integer,
        // e.g 6,0000023 calculated on CPU may be cast to 5 by OpenCL convert_uchar function.
        // That is why the threshold is set 1.f for integer types.
        if (targetDevice == "GPU" &&
                (model_type == ov::element::u8 || model_type == ov::element::i8)) {
            abs_threshold = 1.f;
        }
    }
};

class GPUInterpolate11LayerTest : public Interpolate11LayerTest {
protected:
    void SetUp() override {
        Interpolate11LayerTest::SetUp();
        ov::test::InterpolateLayerTestParams params = GetParam();
        ov::test::InterpolateSpecificParams interpolate_params;
        ov::element::Type model_type;
        std::vector<ov::test::InputShape> shapes;
        ov::Shape target_shape;
        std::map<std::string, std::string> additional_config;
        std::tie(interpolate_params, model_type, shapes, target_shape, targetDevice, additional_config) = this->GetParam();
        // Some rounding float to integer types on GPU may differ from CPU, and as result,
        // the actual values may differ from reference ones on 1 when the float is very close to an integer,
        // e.g 6,0000023 calculated on CPU may be cast to 5 by OpenCL convert_uchar function.
        // That is why the threshold is set 1.f for integer types.
        if (targetDevice == "GPU" &&
                (model_type == ov::element::u8 || model_type == ov::element::i8)) {
            abs_threshold = 1.f;
        }
    }
};
}  // namespace test
}  // namespace ov
