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

// End-to-end regression coverage for the get_sparsity() fix in
// src/frontends/tensorflow_lite/src/utils.cpp. Each test model carries a
// non-null SparsityParameters whose required sub-fields are deliberately
// incomplete (missing or empty traversal_order / block_map / dim_metadata).
// With the fix the SparsityInfo::enable() call inside get_sparsity()
// recognizes the metadata as non-sparse and the constant tensor falls back
// to its raw buffer, so both load() and convert() must succeed. Without the
// fix dense_data() throws inside TensorLitePlace's ctor and convert() fails.
class IncompleteSparsityLoadConvertTest : public ::testing::TestWithParam<std::string> {
protected:
    void SetUp() override {
        FrontEndManager fem;
        OV_ASSERT_NO_THROW(m_frontEnd = fem.load_by_framework(TF_LITE_FE));
        ASSERT_NE(m_frontEnd, nullptr);
    }
    FrontEnd::Ptr m_frontEnd;
};

TEST_P(IncompleteSparsityLoadConvertTest, load_and_convert_succeed) {
    auto model_filename = FrontEndTestUtils::make_model_path(string(TEST_TENSORFLOW_LITE_MODELS_DIRNAME) + GetParam());

    InputModel::Ptr inputModel;
    OV_ASSERT_NO_THROW(inputModel = m_frontEnd->load(model_filename));
    ASSERT_NE(inputModel, nullptr);

    shared_ptr<ov::Model> model;
    OV_ASSERT_NO_THROW(model = m_frontEnd->convert(inputModel));
    ASSERT_NE(model, nullptr);
    EXPECT_GT(model->get_ops().size(), 0u)
        << "Converted model must contain at least one op (Parameter + ADD + Result for the simple "
           "graphs, with a DEQUANTIZE inserted for the int8 variant).";
}

INSTANTIATE_TEST_SUITE_P(SparseIncomplete,
                         IncompleteSparsityLoadConvertTest,
                         ::testing::Values(
                             // Same shape as the HandsLandmarkFull regression: traversal_order
                             // and block_map populated, dim_metadata field omitted entirely.
                             "sparse_incomplete/missing_dim_metadata.tflite",
                             // dim_metadata present but vector length 0.
                             "sparse_incomplete/empty_dim_metadata.tflite",
                             // traversal_order omitted; block_map and dim_metadata present.
                             "sparse_incomplete/missing_traversal_order.tflite",
                             // block_map absent (valid for standard CSR tensors); verifies
                             // densification succeeds without block_map.
                             "sparse_incomplete/missing_block_map.tflite",
                             // SparsityParameters table referenced but all three fields omitted.
                             "sparse_incomplete/empty_sparsity.tflite",
                             // INT8 const with empty SparsityParameters consumed by DEQUANTIZE
                             // then ADD - mirrors the original failure context (DEQUANTIZE op
                             // on an incomplete-sparsity constant).
                             "sparse_incomplete/dequantize_incomplete_sparsity.tflite"));
