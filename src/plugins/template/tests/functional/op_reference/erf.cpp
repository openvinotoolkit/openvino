// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/erf.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

using namespace reference_tests;
using namespace ov;

struct ErfParams {
    template <class IT>
    ErfParams(const ov::Shape& shape, const ov::element::Type& iType, const std::vector<IT>& iValues)
        : pshape(shape),
          inType(iType),
          outType(iType),
          inputData(CreateTensor(iType, iValues)) {
        std::vector<IT> oValues;
        std::vector<double> output;
        for (auto element : iValues)
            output.push_back(static_cast<double>(element));

        std::transform(output.begin(), output.end(), output.begin(), [](double input) -> double {
            return std::erf(input);
        });

        if (std::is_integral<IT>()) {
            std::transform(output.begin(), output.end(), output.begin(), [](double input) -> double {
                return std::round(input);
            });
        }

        for (auto element : output)
            oValues.push_back(static_cast<IT>(element));
        refData = CreateTensor(pshape, outType, oValues);
    }
    ov::Shape pshape;
    ov::element::Type inType;
    ov::element::Type outType;
    ov::Tensor inputData;
    ov::Tensor refData;
};

class ReferenceErfLayerTest : public testing::TestWithParam<ErfParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        auto params = GetParam();
        function = CreateFunction(params.pshape, params.inType, params.outType);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<ErfParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.pshape << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const Shape& input_shape,
                                                 const element::Type& input_type,
                                                 const element::Type& expected_output_type) {
        const auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        const auto erf = std::make_shared<op::v0::Erf>(in);
        return std::make_shared<ov::Model>(NodeVector{erf}, ParameterVector{in});
    }
};

TEST_P(ReferenceErfLayerTest, CompareWithRefs) {
    Exec();
}

INSTANTIATE_TEST_SUITE_P(
    smoke_Erf_With_Hardcoded_Refs,
    ReferenceErfLayerTest,
    ::testing::Values(
        ErfParams(ov::Shape{2, 5},
                  ov::element::f32,
                  std::vector<float>{-INFINITY, -4.0f, -3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, INFINITY}),
        ErfParams(ov::Shape{2, 5},
                  ov::element::f16,
                  std::vector<float16>{-INFINITY, -4.0f, -3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, INFINITY}),
        ErfParams(ov::Shape{2, 3},
                  ov::element::i32,
                  std::vector<
                      int32_t>{std::numeric_limits<int32_t>::min(), -2, -1, 1, 2, std::numeric_limits<int32_t>::max()}),
        ErfParams(ov::Shape{2, 3},
                  ov::element::u32,
                  std::vector<uint32_t>{std::numeric_limits<uint32_t>::min(),
                                        0,
                                        1,
                                        2,
                                        3,
                                        std::numeric_limits<uint32_t>::max()}),
        ErfParams(ov::Shape{2, 3},
                  ov::element::i64,
                  std::vector<
                      int64_t>{std::numeric_limits<int64_t>::min(), -2, -1, 1, 2, std::numeric_limits<int64_t>::max()}),
        ErfParams(ov::Shape{2, 3},
                  ov::element::u64,
                  std::vector<uint64_t>{std::numeric_limits<uint64_t>::min(),
                                        0,
                                        1,
                                        2,
                                        3,
                                        std::numeric_limits<uint64_t>::max()})),
    ReferenceErfLayerTest::getTestCaseName);
