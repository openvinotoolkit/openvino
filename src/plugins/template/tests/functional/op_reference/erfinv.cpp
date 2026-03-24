// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/erfinv.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct ErfInvParams {
    template <class IT>
    ErfInvParams(const ov::PartialShape& shape,
                 const ov::element::Type& iType,
                 const std::vector<IT>& iValues,
                 const std::vector<IT>& oValues)
        : pshape(shape),
          inType(iType),
          outType(iType),
          inputData(CreateTensor(iType, iValues)),
          refData(CreateTensor(iType, oValues)) {}

    ov::PartialShape pshape;
    ov::element::Type inType;
    ov::element::Type outType;
    ov::Tensor inputData;
    ov::Tensor refData;
};

class ReferenceErfInvLayerTest : public testing::TestWithParam<ErfInvParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape, params.inType);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<ErfInvParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.pshape << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const PartialShape& input_shape, const element::Type& input_type) {
        const auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        const auto erfinv = std::make_shared<op::v16::ErfInv>(in);
        return std::make_shared<ov::Model>(OutputVector{erfinv}, ParameterVector{in});
    }
};

TEST_P(ReferenceErfInvLayerTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<ErfInvParams> generateErfInvFloatParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    // Expected values computed from the Giles (2010) approximation at float32 precision.
    // The symmetric property erfinv(-x) = -erfinv(x) is verified across the full domain.
    std::vector<ErfInvParams> params{
        // Zero — must be exactly 0
        ErfInvParams(ov::PartialShape{1}, IN_ET, std::vector<T>{T(0.0f)}, std::vector<T>{T(0.0f)}),
        // Small-to-mid domain: ±0.1, ±0.3, ±0.5
        ErfInvParams(ov::PartialShape{6},
                     IN_ET,
                     std::vector<T>{T(0.1f), T(-0.1f), T(0.3f), T(-0.3f), T(0.5f), T(-0.5f)},
                     std::vector<T>{T(0.08885599f),
                                    T(-0.08885599f),
                                    T(0.27246271f),
                                    T(-0.27246271f),
                                    T(0.47693628f),
                                    T(-0.47693628f)}),
        // Mid-to-large domain: ±0.7, ±0.9
        ErfInvParams(ov::PartialShape{4},
                     IN_ET,
                     std::vector<T>{T(0.7f), T(-0.7f), T(0.9f), T(-0.9f)},
                     std::vector<T>{T(0.73286909f), T(-0.73286909f), T(1.16308725f), T(-1.16308725f)}),
        // Boundary: x=±1 → ±inf
        ErfInvParams(
            ov::PartialShape{2},
            IN_ET,
            std::vector<T>{T(1.0f), T(-1.0f)},
            std::vector<T>{T(std::numeric_limits<float>::infinity()), T(-std::numeric_limits<float>::infinity())}),
    };
    return params;
}

// f64 uses higher-precision expected values from the Giles double-precision path.
std::vector<ErfInvParams> generateErfInvF64Params() {
    std::vector<ErfInvParams> params{
        ErfInvParams(ov::PartialShape{7},
                     element::f64,
                     std::vector<double>{0.0, 0.1, -0.1, 0.5, -0.5, 0.9, -0.9},
                     std::vector<double>{0.0,
                                         0.08885599049425768,
                                         -0.08885599049425768,
                                         0.4769362762044699,
                                         -0.4769362762044699,
                                         1.1630871536766741,
                                         -1.1630871536766741}),
        ErfInvParams(
            ov::PartialShape{2},
            element::f64,
            std::vector<double>{1.0, -1.0},
            std::vector<double>{std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()}),
    };
    return params;
}

std::vector<ErfInvParams> generateErfInvCombinedParams() {
    const std::vector<std::vector<ErfInvParams>> typeParams{generateErfInvFloatParams<element::Type_t::f32>(),
                                                            generateErfInvFloatParams<element::Type_t::f16>(),
                                                            generateErfInvFloatParams<element::Type_t::bf16>(),
                                                            generateErfInvF64Params()};
    std::vector<ErfInvParams> combinedParams;
    for (const auto& p : typeParams) {
        combinedParams.insert(combinedParams.end(), p.begin(), p.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_ErfInv_With_Hardcoded_Refs,
                         ReferenceErfInvLayerTest,
                         testing::ValuesIn(generateErfInvCombinedParams()),
                         ReferenceErfInvLayerTest::getTestCaseName);

// Dedicated edge-case test: |x|>1 must produce NaN, x=±1 must produce ±inf.
class ReferenceErfInvEdgeCaseTest : public CommonReferenceTest {
public:
    void SetUp() {
        const auto in = std::make_shared<op::v0::Parameter>(element::f32, Shape{4});
        function =
            std::make_shared<ov::Model>(OutputVector{std::make_shared<op::v16::ErfInv>(in)}, ParameterVector{in});
        // inputs: two out-of-domain values, then the two boundary values
        inputData = {CreateTensor(element::f32, std::vector<float>{2.0f, -1.5f, 1.0f, -1.0f})};
    }

    void Validate() override {
        const auto* data = inferRequest.get_tensor(executableNetwork.output(0)).data<float>();
        EXPECT_TRUE(std::isnan(data[0])) << "erfinv(2) expected NaN, got " << data[0];
        EXPECT_TRUE(std::isnan(data[1])) << "erfinv(-1.5) expected NaN, got " << data[1];
        EXPECT_TRUE(std::isinf(data[2]) && data[2] > 0) << "erfinv(1) expected +inf, got " << data[2];
        EXPECT_TRUE(std::isinf(data[3]) && data[3] < 0) << "erfinv(-1) expected -inf, got " << data[3];
    }
};

TEST(ErfInvEdgeCases, OutOfDomainAndBoundary) {
    ReferenceErfInvEdgeCaseTest test;
    test.SetUp();
    test.Exec();
}

}  // namespace
