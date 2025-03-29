// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/string_tensor_unpack.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

namespace {
struct StringTensorUnpackParams {
    StringTensorUnpackParams(const reference_tests::Tensor& dataTensor,
                             const reference_tests::Tensor& beginsTensor,
                             const reference_tests::Tensor& endsTensor,
                             const reference_tests::Tensor& symbolsTensor)
        : dataTensor(dataTensor),
          beginsTensor(beginsTensor),
          endsTensor(endsTensor),
          symbolsTensor(symbolsTensor) {}

    reference_tests::Tensor dataTensor;
    reference_tests::Tensor beginsTensor;
    reference_tests::Tensor endsTensor;
    reference_tests::Tensor symbolsTensor;
};

class ReferenceStringTensorUnpackV15LayerTest : public testing::TestWithParam<StringTensorUnpackParams>,
                                                public reference_tests::CommonReferenceTest {
protected:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params);
        inputData = {params.dataTensor.data};
        refOutData = {params.beginsTensor.data, params.endsTensor.data, params.symbolsTensor.data};
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<StringTensorUnpackParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "dShape=" << param.dataTensor.shape;
        return result.str();
    }

private:
    static std::shared_ptr<ov::Model> CreateFunction(const StringTensorUnpackParams& params) {
        const auto data = std::make_shared<ov::op::v0::Parameter>(params.dataTensor.type, params.dataTensor.shape);
        const auto string_tensor_unpack = std::make_shared<ov::op::v15::StringTensorUnpack>(data);
        return std::make_shared<ov::Model>(ov::OutputVector{string_tensor_unpack->outputs()},
                                           ov::ParameterVector{data},
                                           ov::op::util::VariableVector{});
    }
};

TEST_P(ReferenceStringTensorUnpackV15LayerTest, CompareWithRefs) {
    Exec();
}

std::vector<StringTensorUnpackParams> generateStringTensorUnpackParams() {
    using ov::element::Type_t;
    using reference_tests::Tensor;
    const std::vector<StringTensorUnpackParams> stringTensorUnpackParams{
        // simple 1D case
        StringTensorUnpackParams(
            Tensor({2}, ov::element::string, std::vector<std::string>{"Intel", "OpenVINO"}),
            Tensor({2}, ov::element::i32, std::vector<int32_t>{0, 5}),
            Tensor({2}, ov::element::i32, std::vector<int32_t>{5, 13}),
            Tensor({13},
                   ov::element::u8,
                   std::vector<uint8_t>{0x49, 0x6e, 0x74, 0x65, 0x6c, 0x4f, 0x70, 0x65, 0x6e, 0x56, 0x49, 0x4e, 0x4f})),
        // 2D strings with spaces and an empty string
        StringTensorUnpackParams(
            Tensor({1, 5},
                   ov::element::string,
                   std::vector<std::string>{"Intel Corp", "   ", "Open VINO", "", "Artificial Intelligence"}),
            Tensor({1, 5}, ov::element::i32, std::vector<int32_t>{0, 10, 13, 22, 22}),
            Tensor({1, 5}, ov::element::i32, std::vector<int32_t>{10, 13, 22, 22, 45}),
            Tensor({45}, ov::element::u8, std::vector<uint8_t>{0x49, 0x6e, 0x74, 0x65, 0x6c, 0x20, 0x43, 0x6f, 0x72,
                                                               0x70, 0x20, 0x20, 0x20, 0x4f, 0x70, 0x65, 0x6e, 0x20,
                                                               0x56, 0x49, 0x4e, 0x4f, 0x41, 0x72, 0x74, 0x69, 0x66,
                                                               0x69, 0x63, 0x69, 0x61, 0x6c, 0x20, 0x49, 0x6e, 0x74,
                                                               0x65, 0x6c, 0x6c, 0x69, 0x67, 0x65, 0x6e, 0x63, 0x65})),
        // strings with special characters
        StringTensorUnpackParams(
            Tensor({3}, ov::element::string, std::vector<std::string>{"In@tel", "Open#VINO", "A$I"}),
            Tensor({3}, ov::element::i32, std::vector<int32_t>{0, 6, 15}),
            Tensor({3}, ov::element::i32, std::vector<int32_t>{6, 15, 18}),
            Tensor({18},
                   ov::element::u8,
                   std::vector<uint8_t>{0x49,
                                        0x6e,
                                        0x40,
                                        0x74,
                                        0x65,
                                        0x6c,
                                        0x4f,
                                        0x70,
                                        0x65,
                                        0x6e,
                                        0x23,
                                        0x56,
                                        0x49,
                                        0x4e,
                                        0x4f,
                                        0x41,
                                        0x24,
                                        0x49})),
        // (2, 2, 2) data shape
        StringTensorUnpackParams(
            Tensor({2, 2, 2},
                   ov::element::string,
                   std::vector<
                       std::string>{"Intel", "OpenVINO", "AI", "Edge", "Computer", "Vision", "Neural", "Networks"}),
            Tensor({2, 2, 2}, ov::element::i32, std::vector<int32_t>{0, 5, 13, 15, 19, 27, 33, 39}),
            Tensor({2, 2, 2}, ov::element::i32, std::vector<int32_t>{5, 13, 15, 19, 27, 33, 39, 47}),
            Tensor({47},
                   ov::element::u8,
                   std::vector<uint8_t>{0x49, 0x6e, 0x74, 0x65, 0x6c, 0x4f, 0x70, 0x65, 0x6e, 0x56, 0x49, 0x4e,
                                        0x4f, 0x41, 0x49, 0x45, 0x64, 0x67, 0x65, 0x43, 0x6f, 0x6d, 0x70, 0x75,
                                        0x74, 0x65, 0x72, 0x56, 0x69, 0x73, 0x69, 0x6f, 0x6e, 0x4e, 0x65, 0x75,
                                        0x72, 0x61, 0x6c, 0x4e, 0x65, 0x74, 0x77, 0x6f, 0x72, 0x6b, 0x73})),
        // single, empty string
        StringTensorUnpackParams(Tensor({1, 1, 1, 1}, ov::element::string, std::vector<std::string>{""}),
                                 Tensor({1, 1, 1, 1}, ov::element::i32, std::vector<int32_t>{0}),
                                 Tensor({1, 1, 1, 1}, ov::element::i32, std::vector<int32_t>{0}),
                                 Tensor({0}, ov::element::u8, std::vector<uint8_t>{})),
        // empty data
        StringTensorUnpackParams(Tensor({0}, ov::element::string, std::vector<std::string>{}),
                                 Tensor({0}, ov::element::i32, std::vector<int32_t>{}),
                                 Tensor({0}, ov::element::i32, std::vector<int32_t>{}),
                                 Tensor({0}, ov::element::u8, std::vector<uint8_t>{})),
    };
    return stringTensorUnpackParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_StringTensorUnpack_With_Hardcoded_Refs,
                         ReferenceStringTensorUnpackV15LayerTest,
                         testing::ValuesIn(generateStringTensorUnpackParams()),
                         ReferenceStringTensorUnpackV15LayerTest::getTestCaseName);

}  // namespace
