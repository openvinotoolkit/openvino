// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/string_tensor_pack.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

namespace {
struct StringTensorPackParams {
    StringTensorPackParams(const reference_tests::Tensor& beginsTensor,
                           const reference_tests::Tensor& endsTensor,
                           const reference_tests::Tensor& symbolsTensor,
                           const reference_tests::Tensor& stringTensor)
        : beginsTensor(beginsTensor),
          endsTensor(endsTensor),
          symbolsTensor(symbolsTensor),
          stringTensor(stringTensor) {}

    reference_tests::Tensor beginsTensor;
    reference_tests::Tensor endsTensor;
    reference_tests::Tensor symbolsTensor;
    reference_tests::Tensor stringTensor;
};

class ReferenceStringTensorPackV15LayerTest : public testing::TestWithParam<StringTensorPackParams>,
                                              public reference_tests::CommonReferenceTest {
protected:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params);
        inputData = {params.beginsTensor.data, params.endsTensor.data, params.symbolsTensor.data};
        refOutData = {params.stringTensor.data};
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<StringTensorPackParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "indicesShape=" << param.beginsTensor.shape;
        result << "_indicesType=" << param.beginsTensor.type;
        result << "_symbolsShape=" << param.symbolsTensor.shape;
        return result.str();
    }

private:
    static std::shared_ptr<ov::Model> CreateFunction(const StringTensorPackParams& params) {
        const auto begins =
            std::make_shared<ov::op::v0::Parameter>(params.beginsTensor.type, params.beginsTensor.shape);
        const auto ends = std::make_shared<ov::op::v0::Parameter>(params.endsTensor.type, params.endsTensor.shape);
        const auto symbols =
            std::make_shared<ov::op::v0::Parameter>(params.symbolsTensor.type, params.symbolsTensor.shape);
        const auto string_tensor_pack = std::make_shared<ov::op::v15::StringTensorPack>(begins, ends, symbols);
        return std::make_shared<ov::Model>(ov::OutputVector{string_tensor_pack->outputs()},
                                           ov::ParameterVector{begins, ends, symbols},
                                           ov::op::util::VariableVector{});
    }
};

TEST_P(ReferenceStringTensorPackV15LayerTest, CompareWithRefs) {
    Exec();
}

template <ov::element::Type_t T_idx>
std::vector<StringTensorPackParams> generateStringTensorPackParams() {
    using ov::element::Type_t;
    using reference_tests::Tensor;
    using T_I = typename ov::element_type_traits<T_idx>::value_type;
    const std::vector<StringTensorPackParams> StringTensorPackParamsList{
        // simple 1D case
        StringTensorPackParams(
            Tensor({2}, T_idx, std::vector<T_I>{0, 5}),
            Tensor({2}, T_idx, std::vector<T_I>{5, 13}),
            Tensor({13},
                   ov::element::u8,
                   std::vector<uint8_t>{0x49, 0x6e, 0x74, 0x65, 0x6c, 0x4f, 0x70, 0x65, 0x6e, 0x56, 0x49, 0x4e, 0x4f}),
            Tensor({2}, ov::element::string, std::vector<std::string>{"Intel", "OpenVINO"})),
        // 2D strings with spaces and an empty string
        StringTensorPackParams(
            Tensor({1, 5}, T_idx, std::vector<T_I>{0, 10, 13, 22, 22}),
            Tensor({1, 5}, T_idx, std::vector<T_I>{10, 13, 22, 22, 45}),

            Tensor({45}, ov::element::u8, std::vector<uint8_t>{0x49, 0x6e, 0x74, 0x65, 0x6c, 0x20, 0x43, 0x6f, 0x72,
                                                               0x70, 0x20, 0x20, 0x20, 0x4f, 0x70, 0x65, 0x6e, 0x20,
                                                               0x56, 0x49, 0x4e, 0x4f, 0x41, 0x72, 0x74, 0x69, 0x66,
                                                               0x69, 0x63, 0x69, 0x61, 0x6c, 0x20, 0x49, 0x6e, 0x74,
                                                               0x65, 0x6c, 0x6c, 0x69, 0x67, 0x65, 0x6e, 0x63, 0x65}),
            Tensor({1, 5},
                   ov::element::string,
                   std::vector<std::string>{"Intel Corp", "   ", "Open VINO", "", "Artificial Intelligence"})),
        // strings with special characters
        StringTensorPackParams(
            Tensor({3}, T_idx, std::vector<T_I>{0, 6, 15}),
            Tensor({3}, T_idx, std::vector<T_I>{6, 15, 18}),
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
                                        0x49}),
            Tensor({3}, ov::element::string, std::vector<std::string>{"In@tel", "Open#VINO", "A$I"})),
        // (2, 2, 2) data shape
        StringTensorPackParams(
            Tensor({2, 2, 2}, T_idx, std::vector<T_I>{0, 5, 13, 15, 19, 27, 33, 39}),
            Tensor({2, 2, 2}, T_idx, std::vector<T_I>{5, 13, 15, 19, 27, 33, 39, 47}),
            Tensor({47},
                   ov::element::u8,
                   std::vector<uint8_t>{0x49, 0x6e, 0x74, 0x65, 0x6c, 0x4f, 0x70, 0x65, 0x6e, 0x56, 0x49, 0x4e,
                                        0x4f, 0x41, 0x49, 0x45, 0x64, 0x67, 0x65, 0x43, 0x6f, 0x6d, 0x70, 0x75,
                                        0x74, 0x65, 0x72, 0x56, 0x69, 0x73, 0x69, 0x6f, 0x6e, 0x4e, 0x65, 0x75,
                                        0x72, 0x61, 0x6c, 0x4e, 0x65, 0x74, 0x77, 0x6f, 0x72, 0x6b, 0x73}),
            Tensor({2, 2, 2},
                   ov::element::string,
                   std::vector<
                       std::string>{"Intel", "OpenVINO", "AI", "Edge", "Computer", "Vision", "Neural", "Networks"})),
        // single, empty string
        StringTensorPackParams(Tensor({1, 1, 1, 1}, T_idx, std::vector<T_I>{0}),
                               Tensor({1, 1, 1, 1}, T_idx, std::vector<T_I>{0}),
                               Tensor({0}, ov::element::u8, std::vector<uint8_t>{}),
                               Tensor({1, 1, 1, 1}, ov::element::string, std::vector<std::string>{""})),
        // empty data
        StringTensorPackParams(Tensor({0}, T_idx, std::vector<T_I>{}),
                               Tensor({0}, T_idx, std::vector<T_I>{}),
                               Tensor({0}, ov::element::u8, std::vector<uint8_t>{}),
                               Tensor({0}, ov::element::string, std::vector<std::string>{})),
        // skipped symbols
        StringTensorPackParams(
            Tensor({1, 2}, T_idx, std::vector<T_I>{0, 8}),
            Tensor({1, 2}, T_idx, std::vector<T_I>{3, 13}),
            Tensor({13},
                   ov::element::u8,
                   std::vector<uint8_t>{0x49, 0x6e, 0x74, 0x65, 0x6c, 0x4f, 0x70, 0x65, 0x6e, 0x56, 0x49, 0x4e, 0x4f}),
            Tensor({1, 2}, ov::element::string, std::vector<std::string>{"Int", "nVINO"})),
        // empty string at the end
        StringTensorPackParams(
            Tensor({1, 3}, T_idx, std::vector<T_I>{0, 5, 13}),
            Tensor({1, 3}, T_idx, std::vector<T_I>{5, 13, 13}),
            Tensor({13},
                   ov::element::u8,
                   std::vector<uint8_t>{0x49, 0x6e, 0x74, 0x65, 0x6c, 0x4f, 0x70, 0x65, 0x6e, 0x56, 0x49, 0x4e, 0x4f}),
            Tensor({1, 3}, ov::element::string, std::vector<std::string>{"Intel", "OpenVINO", ""})),
        // empty bytes input
        StringTensorPackParams(Tensor({1, 3}, T_idx, std::vector<T_I>{0, 0, 0}),
                               Tensor({1, 3}, T_idx, std::vector<T_I>{0, 0, 0}),
                               Tensor({0}, ov::element::u8, std::vector<uint8_t>{}),
                               Tensor({1, 3}, ov::element::string, std::vector<std::string>{"", "", ""})),
    };
    return StringTensorPackParamsList;
}

std::vector<StringTensorPackParams> generateStringTensorPackV15CombinedParams() {
    using ov::element::Type_t;
    const std::vector<std::vector<StringTensorPackParams>> stringTensorPackTypeParams{
        generateStringTensorPackParams<Type_t::i32>(),
        generateStringTensorPackParams<Type_t::i64>(),
    };

    std::vector<StringTensorPackParams> combinedParams;
    for (const auto& params : stringTensorPackTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_StringTensorPack_With_Hardcoded_Refs,
                         ReferenceStringTensorPackV15LayerTest,
                         testing::ValuesIn(generateStringTensorPackV15CombinedParams()),
                         ReferenceStringTensorPackV15LayerTest::getTestCaseName);

}  // namespace
