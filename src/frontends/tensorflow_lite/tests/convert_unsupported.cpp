// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_assertions.hpp"
#include "convert_model.hpp"
#include "tf_utils.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::frontend;

// Tests where load() succeeds but convert() throws an exception
class MalformedModelConvertTest : public ::testing::TestWithParam<std::string> {
protected:
    void SetUp() override {
        FrontEndManager fem;
        OV_ASSERT_NO_THROW(m_frontEnd = fem.load_by_framework(TF_LITE_FE));
        ASSERT_NE(m_frontEnd, nullptr);
    }
    FrontEnd::Ptr m_frontEnd;
};

TEST_P(MalformedModelConvertTest, convert_throws) {
    auto model_filename = FrontEndTestUtils::make_model_path(string(TEST_TENSORFLOW_LITE_MODELS_DIRNAME) + GetParam());
    InputModel::Ptr inputModel;
    OV_ASSERT_NO_THROW(inputModel = m_frontEnd->load(model_filename));
    ASSERT_NE(inputModel, nullptr);
    shared_ptr<ov::Model> model;
    ASSERT_THROW(model = m_frontEnd->convert(inputModel), std::exception);
}

INSTANTIATE_TEST_SUITE_P(BadHeader,
                         MalformedModelConvertTest,
                         ::testing::Values("bad_header/zerolen.tflite",
                                           "bad_header/wrong_len_3.tflite",
                                           "bad_header/wrong_pos.tflite"));

// Tests where load() itself throws an exception (malformed indices)
class MalformedModelLoadTest : public ::testing::TestWithParam<std::string> {
protected:
    void SetUp() override {
        FrontEndManager fem;
        OV_ASSERT_NO_THROW(m_frontEnd = fem.load_by_framework(TF_LITE_FE));
        ASSERT_NE(m_frontEnd, nullptr);
    }
    FrontEnd::Ptr m_frontEnd;
};

TEST_P(MalformedModelLoadTest, load_throws) {
    auto model_filename = FrontEndTestUtils::make_model_path(string(TEST_TENSORFLOW_LITE_MODELS_DIRNAME) + GetParam());
    ASSERT_THROW(m_frontEnd->load(model_filename), std::exception);
}

INSTANTIATE_TEST_SUITE_P(OobIndices,
                         MalformedModelLoadTest,
                         ::testing::Values("malformed_indices/oob_output_tensor_index.tflite",
                                           "malformed_indices/oob_input_tensor_index.tflite",
                                           "malformed_indices/oob_opcode_index.tflite",
                                           "malformed_indices/oob_graph_io_tensor_index.tflite",
                                           "malformed_indices/oob_buffer_index.tflite"));

// quantized_dimension=-1: load() throws in get_quantization() (non-negative axis check)
INSTANTIATE_TEST_SUITE_P(NegativeQuantDim,
                         MalformedModelLoadTest,
                         ::testing::Values("oob_quant_dim/negative_axis.tflite"));

// quantized_dimension=100 on rank-2 tensor: load() succeeds, convert() throws
// in get_quant_shape() (axis >= rank check)
INSTANTIATE_TEST_SUITE_P(OobQuantDim,
                         MalformedModelConvertTest,
                         ::testing::Values("oob_quant_dim/axis_exceeds_rank.tflite"));
