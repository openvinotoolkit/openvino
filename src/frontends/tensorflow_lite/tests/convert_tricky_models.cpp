// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/test_case.hpp"
#include "common_test_utils/test_control.hpp"
#include "common_test_utils/test_tools.hpp"
#include "common_test_utils/type_prop.hpp"
#include "conversion_extension.hpp"
#include "gtest/gtest.h"
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

OPENVINO_TEST(TensorFlowLiteTrickyModels, tflite_slice_neg_size_stays_dynamic) {
    // SLICE with `size = -1` on an axis legitimately requires the negative-
    // size handling, which today produces a dynamic dim on that axis through
    // the Select cascade. Guards that the translator's fast path doesn't
    // mistakenly trigger for negative sizes.
    auto model = convert_model("slice_neg_size.tflite");

    const auto slice = find_slice(model);
    ASSERT_NE(slice, nullptr) << "Slice op not found in converted model";

    const auto& pshape = slice->get_output_partial_shape(0);
    EXPECT_TRUE(pshape.is_dynamic()) << "Slice output is statically shaped; "
                                        "expected dynamic on at least one axis. Got: "
                                     << pshape;
}
