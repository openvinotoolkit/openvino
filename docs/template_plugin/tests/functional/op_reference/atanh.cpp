// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <limits>

#include "openvino/op/atanh.hpp"
#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

struct AtanhParams {
    template <class IT>
    AtanhParams(const ov::PartialShape& shape, const ov::element::Type& iType, const std::vector<IT>& iValues)
        : pshape(shape), inType(iType), outType(iType), inputData(CreateTensor(iType, iValues)) {
        std::vector<IT> oValues;
        std::vector<double> output;
        for (auto element : iValues)
            output.push_back(static_cast<double>(element));

        std::transform(output.begin(), output.end(), output.begin(), [](double input) -> double {
            return std::atanh(input);
        });

        if (std::is_integral<IT>()) {
            std::transform(output.begin(), output.end(), output.begin(), [](double input) -> double {
                return std::round(input);
            });
        }

        for (auto element : output)
            oValues.push_back(static_cast<IT>(element));
        refData = CreateTensor(outType, oValues);
    }
    ov::PartialShape pshape;
    ov::element::Type inType;
    ov::element::Type outType;
    ov::runtime::Tensor inputData;
    ov::runtime::Tensor refData;
};

class ReferenceAtanhLayerTest : public testing::TestWithParam<AtanhParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape, params.inType, params.outType);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<AtanhParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.pshape << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType;
        return result.str();
    }

private:
    static std::shared_ptr<Function> CreateFunction(const PartialShape& input_shape, const element::Type& input_type,
                                                    const element::Type& expected_output_type) {
        const auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        const auto atanh = std::make_shared<op::v3::Atanh>(in);
        return std::make_shared<ov::Function>(NodeVector {atanh}, ParameterVector {in});
    }
};

TEST_P(ReferenceAtanhLayerTest, CompareWithRefs) {
    Exec();
}

INSTANTIATE_TEST_SUITE_P(
    smoke_Atanh_With_Hardcoded_Refs, ReferenceAtanhLayerTest,
    ::testing::Values(AtanhParams(ov::PartialShape {2, 4}, ov::element::f32,
                                std::vector<float> {-INFINITY, -2.0f, -1.0f, -0.5f, 0.0f, 0.8f, 1.0f, INFINITY}),
                      AtanhParams(ov::PartialShape {2, 4}, ov::element::f16,
                                std::vector<float16> {-INFINITY, -2.0f, -1.0f, -0.5f, -0.0f, 0.8f, 1.0f, INFINITY}),
                      AtanhParams(ov::PartialShape {2, 3}, ov::element::i32,
                                std::vector<int32_t> {std::numeric_limits<int32_t>::min(), -2, -1, 1, 2, std::numeric_limits<int32_t>::max()}),
                      AtanhParams(ov::PartialShape {2, 3}, ov::element::u32,
                                std::vector<uint32_t> {std::numeric_limits<uint32_t>::min(), 0, 1, 2, 3, std::numeric_limits<uint32_t>::max()}),
                      AtanhParams(ov::PartialShape {2, 3}, ov::element::i64,
                                std::vector<int64_t> {std::numeric_limits<int64_t>::min(), -2, -1, 1, 2, std::numeric_limits<int64_t>::max()}),
                      AtanhParams(ov::PartialShape {2, 3}, ov::element::u64,
                                std::vector<uint64_t> {std::numeric_limits<uint64_t>::min(), 0, 1, 2, 3, std::numeric_limits<uint64_t>::max()})),
    ReferenceAtanhLayerTest::getTestCaseName);
