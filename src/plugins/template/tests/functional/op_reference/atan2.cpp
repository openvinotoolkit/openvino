// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/atan2.hpp"

#include <gtest/gtest.h>

#include <cmath>

#include "base_reference_test.hpp"

using namespace ov;
using namespace reference_tests;

namespace {

struct Atan2Params {
    template <class IT>
    Atan2Params(const PartialShape& y_shape,
                const PartialShape& x_shape,
                const element::Type& type,
                const std::vector<IT>& y_values,
                const std::vector<IT>& x_values,
                const std::vector<IT>& expected)
        : yShape(y_shape),
          xShape(x_shape),
          inType(type),
          outType(type),
          yData(CreateTensor(type, y_values)),
          xData(CreateTensor(type, x_values)),
          refData(CreateTensor(type, expected)) {}

    PartialShape yShape;
    PartialShape xShape;
    element::Type inType;
    element::Type outType;
    ov::Tensor yData;
    ov::Tensor xData;
    ov::Tensor refData;
};

class ReferenceAtan2LayerTest : public testing::TestWithParam<Atan2Params>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.yShape, params.xShape, params.inType);
        inputData = {params.yData, params.xData};
        refOutData = {params.refData};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<Atan2Params>& obj) {
        const auto& p = obj.param;
        std::ostringstream result;
        result << "yShape=" << p.yShape << "_xShape=" << p.xShape << "_type=" << p.inType;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const PartialShape& y_shape,
                                                  const PartialShape& x_shape,
                                                  const element::Type& type) {
        const auto y = std::make_shared<op::v0::Parameter>(type, y_shape);
        const auto x = std::make_shared<op::v0::Parameter>(type, x_shape);
        const auto atan2 = std::make_shared<op::v17::Atan2>(y, x);
        return std::make_shared<Model>(OutputVector{atan2}, ParameterVector{y, x});
    }
};

TEST_P(ReferenceAtan2LayerTest, Atan2WithHardcodedRefs) {
    Exec();
}

template <element::Type_t ET>
std::vector<Atan2Params> generateParams() {
    using T = typename element_type_traits<ET>::value_type;
    const float pi = static_cast<float>(std::atan(1.0) * 4);
    return {
        // Quadrant I: y>0, x>0
        Atan2Params(PartialShape{4},
                    PartialShape{4},
                    ET,
                    std::vector<T>{1.f, 1.f, 0.f, static_cast<T>(pi / 2)},
                    std::vector<T>{1.f, 0.f, 1.f, static_cast<T>(pi / 2)},
                    std::vector<T>{static_cast<T>(pi / 4),
                                   static_cast<T>(pi / 2),
                                   0.f,
                                   static_cast<T>(std::atan2(pi / 2, pi / 2))}),
        // Quadrant II: y>0, x<0 
        Atan2Params(PartialShape{2},
                    PartialShape{2},
                    ET,
                    std::vector<T>{1.f, 0.f},
                    std::vector<T>{-1.f, -1.f},
                    std::vector<T>{static_cast<T>(3 * pi / 4), static_cast<T>(pi)}),
        // Negative y
        Atan2Params(PartialShape{3},
                    PartialShape{3},
                    ET,
                    std::vector<T>{-1.f, -1.f, 0.f},
                    std::vector<T>{1.f, -1.f, 0.f},
                    std::vector<T>{static_cast<T>(-pi / 4), static_cast<T>(-3 * pi / 4), 0.f}),
        // Broadcast: scalar x
        Atan2Params(PartialShape{3},
                    PartialShape{1},
                    ET,
                    std::vector<T>{0.f, 1.f, -1.f},
                    std::vector<T>{1.f},
                    std::vector<T>{0.f, static_cast<T>(pi / 4), static_cast<T>(-pi / 4)}),
    };
}

std::vector<Atan2Params> generateCombinedParams() {
    const std::vector<std::vector<Atan2Params>> all{
        generateParams<element::Type_t::f32>(),
        generateParams<element::Type_t::f16>(),
        generateParams<element::Type_t::bf16>(),
    };
    std::vector<Atan2Params> combined;
    for (const auto& v : all)
        combined.insert(combined.end(), v.begin(), v.end());
    return combined;
}

INSTANTIATE_TEST_SUITE_P(smoke_Atan2_With_Hardcoded_Refs,
                         ReferenceAtan2LayerTest,
                         ::testing::ValuesIn(generateCombinedParams()),
                         ReferenceAtan2LayerTest::getTestCaseName);

}  // namespace
