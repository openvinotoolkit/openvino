// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/select.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

struct SelectParams {
    template <class IT, class OT>
    SelectParams(const element::Type& data_type,
                 const op::AutoBroadcastSpec& broadcast,
                 const PartialShape& select_input_pshape,
                 const std::vector<char>& select_input,
                 const PartialShape& if_input_pshape,
                 const std::vector<IT>& if_input,
                 const PartialShape& else_input_pshape,
                 const std::vector<IT>& else_input,
                 const std::vector<OT>& expected_output)
        : data_type(data_type),
          broadcast(broadcast),
          select_input_pshape(select_input_pshape),
          select_input(CreateTensor(element::boolean, select_input)),
          if_input_pshape(if_input_pshape),
          if_input(CreateTensor(data_type, if_input)),
          else_input_pshape(else_input_pshape),
          else_input(CreateTensor(data_type, else_input)),
          expected_output(CreateTensor(data_type, expected_output)) {}

    element::Type data_type;
    op::AutoBroadcastSpec broadcast;
    PartialShape select_input_pshape;
    ov::Tensor select_input;
    PartialShape if_input_pshape;
    ov::Tensor if_input;
    PartialShape else_input_pshape;
    ov::Tensor else_input;
    ov::Tensor expected_output;
};

class ReferenceSelectLayerTest : public testing::TestWithParam<SelectParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        legacy_compare = true;
        auto params = GetParam();
        function = CreateFunction(params.data_type,
                                  params.broadcast,
                                  params.select_input_pshape,
                                  params.if_input_pshape,
                                  params.else_input_pshape);
        inputData = {params.select_input, params.if_input, params.else_input};
        refOutData = {params.expected_output};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<SelectParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "data_type=" << param.data_type << "_";
        result << "broadcast=" << param.broadcast.m_type << "_";
        result << "select_shape=" << param.select_input_pshape << "_";
        result << "if_shape=" << param.if_input_pshape << "_";
        result << "else_shape=" << param.else_input_pshape;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const element::Type& data_type,
                                                 const op::AutoBroadcastSpec& broadcast,
                                                 const PartialShape& select_pshape,
                                                 const PartialShape& if_pshape,
                                                 const PartialShape& else_pshape) {
        auto A = std::make_shared<op::v0::Parameter>(element::boolean, select_pshape);
        auto B = std::make_shared<op::v0::Parameter>(data_type, if_pshape);
        auto C = std::make_shared<op::v0::Parameter>(data_type, else_pshape);
        return std::make_shared<ov::Model>(std::make_shared<op::v1::Select>(A, B, C, broadcast),
                                           ParameterVector{A, B, C});
    }
};

TEST_P(ReferenceSelectLayerTest, CompareWithHardcodedRefs) {
    Exec();
}

INSTANTIATE_TEST_SUITE_P(smoke_Select_With_Hardcoded_Refs,
                         ReferenceSelectLayerTest,
                         ::testing::Values(
                             // fp32, no brodcasting
                             SelectParams(element::f32,                                // if/else/output data type
                                          op::AutoBroadcastType::NONE,                 // broadcasting type
                                          PartialShape{2, 2, 2},                       // select shape
                                          std::vector<char>{0, 1, 1, 0, 0, 1, 0, 1},   // select data
                                          PartialShape{2, 2, 2},                       // if shape
                                          std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8},  // if data
                                          PartialShape{2, 2, 2},                       // else shape
                                          std::vector<float>{11, 12, 13, 14, 15, 16, 17, 18},  // else data
                                          std::vector<float>{11, 2, 3, 14, 15, 6, 17, 8}),     // expected output data
                             // i32, no brodcasting
                             SelectParams(element::i32,                                // if/else/output data type
                                          op::AutoBroadcastType::NONE,                 // broadcasting type
                                          PartialShape{2, 2, 2},                       // select shape
                                          std::vector<char>{0, 1, 1, 0, 0, 1, 0, 1},   // select data
                                          PartialShape{2, 2, 2},                       // if shape
                                          std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8},  // if data
                                          PartialShape{2, 2, 2},                       // else shape
                                          std::vector<float>{11, 12, 13, 14, 15, 16, 17, 18},  // else data
                                          std::vector<float>{11, 2, 3, 14, 15, 6, 17, 8}),     // expected output data
                             // fp32, numpy brodcasting
                             SelectParams(element::f32,                    // if/else/output data type
                                          op::AutoBroadcastType::NUMPY,    // broadcasting type
                                          PartialShape{4},                 // select shape
                                          std::vector<char>{0, 1, 1, 0},   // select data
                                          PartialShape{4},                 // if shape
                                          std::vector<float>{1, 2, 3, 4},  // if data
                                          PartialShape{2, 4},              // else shape
                                          std::vector<float>{11, 12, 13, 14, 15, 16, 17, 18},  // else data
                                          std::vector<float>{11, 2, 3, 14, 15, 2, 3, 18}),     // expected output data
                             // i32, numpy brodcasting
                             SelectParams(element::i32,                    // if/else/output data type
                                          op::AutoBroadcastType::NUMPY,    // broadcasting type
                                          PartialShape{4},                 // select shape
                                          std::vector<char>{0, 1, 1, 0},   // select data
                                          PartialShape{4},                 // if shape
                                          std::vector<float>{1, 2, 3, 4},  // if data
                                          PartialShape{2, 4},              // else shape
                                          std::vector<float>{11, 12, 13, 14, 15, 16, 17, 18},  // else data
                                          std::vector<float>{11, 2, 3, 14, 15, 2, 3, 18}),     // expected output data
                             // fp32, pdpd brodcasting
                             SelectParams(element::f32,                                      // if/else/output data type
                                          {op::AutoBroadcastType::PDPD, -1},                 // broadcasting type
                                          PartialShape{2, 4},                                // select shape
                                          std::vector<char>{0, 0, 0, 0, 0, 1, 1, 1},         // select data
                                          PartialShape{2, 4},                                // if shape
                                          std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8},        // if data
                                          PartialShape{4},                                   // else shape
                                          std::vector<float>{11, 12, 13, 14},                // else data
                                          std::vector<float>{11, 12, 13, 14, 11, 6, 7, 8}),  // expected output data
                             // i32, pdpd brodcasting
                             SelectParams(element::i32,                                // if/else/output data type
                                          {op::AutoBroadcastType::PDPD, -1},           // broadcasting type
                                          PartialShape{2, 4},                          // select shape
                                          std::vector<char>{0, 0, 0, 0, 0, 1, 1, 1},   // select data
                                          PartialShape{2, 4},                          // if shape
                                          std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8},  // if data
                                          PartialShape{4},                             // else shape
                                          std::vector<float>{11, 12, 13, 14},          // else data
                                          std::vector<float>{11, 12, 13, 14, 11, 6, 7, 8})),  // expected output data
                         ReferenceSelectLayerTest::getTestCaseName);
