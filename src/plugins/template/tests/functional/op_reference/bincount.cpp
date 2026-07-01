// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/bincount.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/parameter.hpp"

namespace {
struct BincountParams {
    template <class TData>
    BincountParams(const ov::PartialShape& shape,
                   const ov::element::Type& dataType,
                   const std::vector<TData>& dataValues,
                   int64_t minlength,
                   const std::vector<int64_t>& oValues)
        : pshape(shape),
          dataType(dataType),
          weightsType(ov::element::dynamic),
          minlength(minlength),
          outType(ov::element::i64),
          inputData({reference_tests::CreateTensor(dataType, dataValues)}),
          refData(reference_tests::CreateTensor(ov::Shape{oValues.size()}, ov::element::i64, oValues)) {}

    template <class TData, class TWeight>
    BincountParams(const ov::PartialShape& shape,
                   const ov::element::Type& dataType,
                   const std::vector<TData>& dataValues,
                   const ov::element::Type& weightsType,
                   const std::vector<TWeight>& weightsValues,
                   int64_t minlength,
                   const std::vector<TWeight>& oValues)
        : pshape(shape),
          dataType(dataType),
          weightsType(weightsType),
          minlength(minlength),
          outType(weightsType),
          inputData({reference_tests::CreateTensor(dataType, dataValues),
                     reference_tests::CreateTensor(weightsType, weightsValues)}),
          refData(reference_tests::CreateTensor(ov::Shape{oValues.size()}, weightsType, oValues)) {}

    ov::PartialShape pshape;
    ov::element::Type dataType;
    ov::element::Type weightsType;
    int64_t minlength;
    ov::element::Type outType;
    std::vector<ov::Tensor> inputData;
    ov::Tensor refData;
};

class ReferenceBincountLayerTest : public testing::TestWithParam<BincountParams>,
                                    public reference_tests::CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape, params.dataType, params.weightsType, params.minlength);
        inputData = params.inputData;
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<BincountParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.pshape << "_";
        result << "dataType=" << param.dataType << "_";
        result << "minlength=" << param.minlength << "_";
        result << "weighted=" << (param.weightsType.is_dynamic() ? "false" : "true");
        return result.str();
    }

private:
    static std::shared_ptr<ov::Model> CreateFunction(const ov::PartialShape& input_shape,
                                                      const ov::element::Type& data_type,
                                                      const ov::element::Type& weights_type,
                                                      int64_t minlength) {
        const auto data = std::make_shared<ov::op::v0::Parameter>(data_type, input_shape);
        if (weights_type.is_dynamic()) {
            const auto bincount = std::make_shared<ov::op::v17::Bincount>(data, minlength);
            return std::make_shared<ov::Model>(ov::OutputVector{bincount}, ov::ParameterVector{data});
        }
        const auto weights = std::make_shared<ov::op::v0::Parameter>(weights_type, input_shape);
        const auto bincount = std::make_shared<ov::op::v17::Bincount>(data, weights, minlength);
        return std::make_shared<ov::Model>(ov::OutputVector{bincount}, ov::ParameterVector{data, weights});
    }
};

TEST_P(ReferenceBincountLayerTest, CompareWithRefs) {
    Exec();
}

std::vector<BincountParams> generateBincountUnweightedParams() {
    return {
        // Output size derived from max(data)+1; minlength=0.
        BincountParams(ov::PartialShape{5},
                       ov::element::i32,
                       std::vector<int32_t>{0, 1, 1, 3, 3},
                       0,
                       std::vector<int64_t>{1, 2, 0, 2}),
        // minlength larger than max(data)+1 pads the output with zeros.
        BincountParams(ov::PartialShape{3},
                       ov::element::i32,
                       std::vector<int32_t>{0, 1, 2},
                       6,
                       std::vector<int64_t>{1, 1, 1, 0, 0, 0}),
        // u8 data type support.
        BincountParams(ov::PartialShape{4},
                       ov::element::u8,
                       std::vector<uint8_t>{2, 2, 2, 0},
                       0,
                       std::vector<int64_t>{1, 0, 3}),
        // Empty input with minlength: all-zero histogram of size minlength.
        BincountParams(ov::PartialShape{0},
                       ov::element::i32,
                       std::vector<int32_t>{},
                       4,
                       std::vector<int64_t>{0, 0, 0, 0}),
    };
}

std::vector<BincountParams> generateBincountWeightedParams() {
    return {
        BincountParams(ov::PartialShape{4},
                       ov::element::i32,
                       std::vector<int32_t>{0, 1, 1, 2},
                       ov::element::f32,
                       std::vector<float>{1.5f, 2.0f, 3.0f, 4.0f},
                       0,
                       std::vector<float>{1.5f, 5.0f, 4.0f}),
    };
}

INSTANTIATE_TEST_SUITE_P(smoke_Bincount_Unweighted_With_Hardcoded_Refs,
                         ReferenceBincountLayerTest,
                         testing::ValuesIn(generateBincountUnweightedParams()),
                         ReferenceBincountLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Bincount_Weighted_With_Hardcoded_Refs,
                         ReferenceBincountLayerTest,
                         testing::ValuesIn(generateBincountWeightedParams()),
                         ReferenceBincountLayerTest::getTestCaseName);

}  // namespace
