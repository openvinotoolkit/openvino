// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/test_case.hpp"
#include "common_test_utils/test_control.hpp"
#include "convert_model.hpp"
#include "tf_utils.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::frontend;

// End-to-end regression coverage for the get_sparsity() fix in
// src/frontends/tensorflow_lite/src/utils.cpp. Two test groups, matching the
// two outcomes that get_sparsity() / SparsityInfo::enable() must produce:
//
//   * Fallback group (IncompleteSparsityLoadConvertTest) — the metadata is
//     malformed (missing/empty traversal_order, dim_metadata, dim_format, or
//     fully empty SparsityParameters). enable() must disable the tensor and
//     TensorLitePlace must expose the raw constant buffer instead of trying
//     to densify(). The unifying contract is that load() and convert() both
//     succeed; without the fix dense_data() throws inside TensorLitePlace's
//     ctor and convert() fails.
//
//   * Densify group (IncompleteSparsityDensifyTest) — the standard-CSR
//     corner case where block_map is legitimately absent. enable() must
//     keep the tensor enabled and densify() must run. Pure load+convert is
//     not enough here: a future regression that disabled the tensor would
//     keep load+convert green by silently falling back to the raw buffer.
//     The test asserts the densified all-zero constant by running inference
//     on CPU and checking that ADD(input, const) == input.
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
                             // SparsityParameters table referenced but all three fields omitted.
                             "sparse_incomplete/empty_sparsity.tflite",
                             // INT8 const with empty SparsityParameters consumed by DEQUANTIZE
                             // then ADD - mirrors the original failure context (DEQUANTIZE op
                             // on an incomplete-sparsity constant).
                             "sparse_incomplete/dequantize_incomplete_sparsity.tflite"));

// missing_block_map.tflite carries a valid standard-CSR layout
// (traversal_order length == tensor rank, dim_metadata fully populated) and
// intentionally omits block_map. The generator emits segments=[0, 0, 0] and
// indices=[], which densifies the (2, 2) constant to all zeros, while the
// raw FLOAT32 buffer behind the same tensor is [1, 2, 3, 4]. The graph is
// ADD(input, const), so:
//
//   * if densify() ran    => output == input                (asserted here)
//   * if fallback fired   => output == input + [1, 2, 3, 4] (would fail)
//
// Without that observable, a future change that disabled this tensor
// unconditionally and silently fell back to the raw buffer would keep
// load+convert green and the regression would slip past the suite.
static std::string s_manifest = "";

OPENVINO_TEST(IncompleteSparsityDensify, missing_block_map_runs_densify) {
    auto model = ov::frontend::tensorflow_lite::tests::convert_model("sparse_incomplete/missing_block_map.tflite");

    auto test_case = ov::test::TestCase(model, ov::test::utils::DEVICE_CPU);
    test_case.add_input<float>(Shape{2, 2}, {10, 20, 30, 40});
    test_case.add_expected_output<float>(Shape{2, 2}, {10, 20, 30, 40});
    test_case.run();
}
