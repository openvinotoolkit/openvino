// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <numeric>

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/test_case.hpp"
#include "common_test_utils/test_control.hpp"
#include "common_test_utils/test_tools.hpp"
#include "common_test_utils/type_prop.hpp"
#include "conversion_extension.hpp"
#include "gtest/gtest.h"
#include "openvino/op/select.hpp"
#include "openvino/op/slice.hpp"
#include "tf_utils.hpp"

using namespace ov;
using namespace ov::frontend::tensorflow_lite::tests;

static std::string s_manifest = "";

using Inputs = std::vector<std::vector<float>>;
using Outputs = std::vector<std::vector<float>>;

OPENVINO_TEST(TensorFlowLiteTrickyModels, tflite_densify) {
    auto model = convert_model("densify.tflite");

    auto test_case = ov::test::TestCase(model, ov::test::utils::DEVICE_CPU);
    test_case.add_input<float>(Shape{1, 2, 3, 3}, {0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 2, 0, 0, 0, 1});
    test_case.add_expected_output<float>(Shape{1, 2, 2, 4}, {2, 1, 0, 0, 0, 3, 1, 0, 0, 2, 0, 0, 2, 0, 1, 0});
    test_case.run();
}

namespace {
std::shared_ptr<Node> find_slice(const std::shared_ptr<Model>& model) {
    for (const auto& op : model->get_ordered_ops()) {
        if (ov::as_type_ptr<op::v8::Slice>(op)) {
            return op;
        }
    }
    return nullptr;
}
}  // namespace

OPENVINO_TEST(TensorFlowLiteTrickyModels, tflite_slice_const_size_yields_static_output) {
    // SLICE with non-negative constant `size` should produce a statically
    // shaped output: the translator must skip the ShapeOf+Select cascade
    // emitted only for the size=-1 ("to end") path.
    auto model = convert_model("slice_const_size.tflite");

    const auto slice = find_slice(model);
    ASSERT_NE(slice, nullptr) << "Slice op not found in converted model";

    const auto& pshape = slice->get_output_partial_shape(0);
    ASSERT_TRUE(pshape.is_static()) << "Slice output is dynamic: " << pshape;
    EXPECT_EQ(pshape.to_shape(), (Shape{1, 128, 4, 128}));
}

OPENVINO_TEST(TensorFlowLiteTrickyModels, tflite_slice_neg_size_keeps_select_cascade) {
    // SLICE with `size = -1` on an axis legitimately requires the negative-
    // size handling. Guards that the translator's fast path does NOT trigger:
    // the producer of Slice's `stop` input must still be a Select node (the
    // size=-1 cascade). Value-correctness for this path is gated separately
    // by tflite_slice_neg_size_matches_tf_ground_truth below.
    auto model = convert_model("slice_neg_size.tflite");

    const auto slice = find_slice(model);
    ASSERT_NE(slice, nullptr) << "Slice op not found in converted model";

    const auto stop_producer = slice->get_input_node_shared_ptr(2);
    EXPECT_NE(ov::as_type_ptr<op::v1::Select>(stop_producer), nullptr)
        << "Slice 'stop' should be produced by Select (size=-1 cascade), got " << stop_producer->get_type_name();
}

OPENVINO_TEST(TensorFlowLiteTrickyModels, tflite_slice_neg_size_matches_tf_ground_truth) {
    // End-to-end correctness gate for the negative-size cascade. The fixture
    // is tf.slice(x[1,128,8,256], begin=[0,0,0,0], size=[1,128,4,-1]) — the
    // -1 on the last axis means "to end", so the expected output is
    // x[:, :, :4, :] reshaped to (1,128,4,256).
    auto model = convert_model("slice_neg_size.tflite");

    constexpr size_t kInputSize = 1 * 128 * 8 * 256;
    constexpr size_t kOutputSize = 1 * 128 * 4 * 256;
    std::vector<float> input(kInputSize);
    std::iota(input.begin(), input.end(), 0.0f);

    // Expected = input[:, :, :4, :] — strip the last 4 elements of axis 2.
    // Linear index of input[n,c,h,w] is ((n*128 + c)*8 + h)*256 + w; output
    // takes h ∈ [0, 4), so we copy the first 4*256 floats of every (n,c)
    // pair.
    std::vector<float> expected(kOutputSize);
    constexpr size_t kRowStride = 256;
    constexpr size_t kInputPlaneStride = 8 * kRowStride;
    constexpr size_t kOutputPlaneStride = 4 * kRowStride;
    for (size_t plane = 0; plane < 128; ++plane) {
        const float* src = input.data() + plane * kInputPlaneStride;
        float* dst = expected.data() + plane * kOutputPlaneStride;
        std::copy(src, src + kOutputPlaneStride, dst);
    }

    auto test_case = ov::test::TestCase(model, ov::test::utils::DEVICE_CPU);
    test_case.add_input<float>(Shape{1, 128, 8, 256}, input);
    test_case.add_expected_output<float>(Shape{1, 128, 4, 256}, expected);
    test_case.run();
}
