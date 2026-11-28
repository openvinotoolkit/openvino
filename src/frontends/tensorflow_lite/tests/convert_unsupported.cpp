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

// quantized_dimension=100 on rank-2 tensor: load() succeeds, convert() throws
// in get_quant_shape() (axis >= rank check)
INSTANTIATE_TEST_SUITE_P(OobQuantDim,
                         MalformedModelConvertTest,
                         ::testing::Values("oob_quant_dim/axis_exceeds_rank.tflite"));

// Tests where load() itself throws an exception
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

// CVS-181019: sparse tensors with invalid index/segment values or overflow-inducing shapes
INSTANTIATE_TEST_SUITE_P(SparseOob,
                         MalformedModelLoadTest,
                         ::testing::Values("sparse_oob/sparse_oob_index.tflite",
                                           "sparse_oob/sparse_negative_index.tflite",
                                           "sparse_oob/sparse_non_monotonic_segments.tflite",
                                           "sparse_oob/sparse_overflow_shape.tflite"));

// buffer size smaller than shape requires: load() throws in load_model() buffer size validation
INSTANTIATE_TEST_SUITE_P(BadBufferSize,
                         MalformedModelLoadTest,
                         ::testing::Values("bad_buffer_size/undersized_buffer.tflite",
                                           "bad_buffer_size/empty_buffer_nonempty_shape.tflite"));

// Tensor with no `name` field set in its vtable: load() throws in safe_tensor_name() instead of segfault
INSTANTIATE_TEST_SUITE_P(MissingTensorName,
                         MalformedModelLoadTest,
                         ::testing::Values("malformed_tensor_name/missing_tensor_name.tflite"));

// SDL505 fuzzing finding: out-of-range operator codes must throw, not crash (CWE-125).
INSTANTIATE_TEST_SUITE_P(OobBuiltinOpcode,
                         MalformedModelLoadTest,
                         ::testing::Values("oob_builtin_opcode/oob_deprecated_builtin_code.tflite",
                                           "oob_builtin_opcode/oob_builtin_code.tflite",
                                           "oob_builtin_opcode/null_sentinel_builtin_code.tflite"));

// Verifier-valid files whose load() must throw, not crash:
//   empty_dense_buffer        - dense constant with a present-but-empty Buffer.data vector; before the
//                               decoder check the loader copied the 16 KiB its shape requires out of it
//   empty_dense_buffer_large  - the same with a 1 MiB shape; before the check loading it crashed
//   operator_without_inputs   - operator whose 'inputs' vector field is absent
//   negative_input_index      - operator input tensor index negative and not the -1 optional sentinel
// The last two are already rejected by the graph iterator; they pin that rejection in place.
// (squeeze_without_dims is handled gracefully rather than rejected; see the SqueezeWithoutDims test.)
INSTANTIATE_TEST_SUITE_P(DecoderNullAndEmptyBuffer,
                         MalformedModelLoadTest,
                         ::testing::Values("decoder_null_and_empty_buffer/empty_dense_buffer.tflite",
                                           "decoder_null_and_empty_buffer/empty_dense_buffer_large.tflite",
                                           "decoder_null_and_empty_buffer/operator_without_inputs.tflite",
                                           "decoder_null_and_empty_buffer/negative_input_index.tflite"));

// A SQUEEZE op whose SqueezeOptions table is present but whose optional squeeze_dims vector is absent
// previously dereferenced the null vector. It must now load and convert, treating the missing field
// as an empty axis list (squeeze all size-1 dimensions).
TEST(TensorFlowLiteSqueezeWithoutDims, loads_and_converts) {
    FrontEndManager fem;
    auto fe = fem.load_by_framework(TF_LITE_FE);
    ASSERT_NE(fe, nullptr);
    auto model_filename = FrontEndTestUtils::make_model_path(
        std::string(TEST_TENSORFLOW_LITE_MODELS_DIRNAME) + "decoder_null_and_empty_buffer/squeeze_without_dims.tflite");
    InputModel::Ptr input_model;
    OV_ASSERT_NO_THROW(input_model = fe->load(model_filename));
    ASSERT_NE(input_model, nullptr);
    std::shared_ptr<ov::Model> model;
    OV_ASSERT_NO_THROW(model = fe->convert(input_model));
    ASSERT_NE(model, nullptr);
}
