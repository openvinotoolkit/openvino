// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/base/layer_test_utils.hpp>
#include <ngraph/opsets/opset8.hpp>
#include "ov_models/builders.hpp"

namespace {

using namespace ngraph;

// Validate scenario permute-reshape of Autodest3D accuracy issue
class PermuteReorderReshapeGPUTest : virtual public LayerTestsUtils::LayerTestsCommon {
protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        auto type = element::f32;

        Shape transeposeInputShape{4, 64, 512, 512};
        Shape concatInputShape{1, 4, 512, 512, 64};

        auto input1 = std::make_shared<opset8::Parameter>(type, transeposeInputShape);
        auto input2 = std::make_shared<opset8::Parameter>(type, concatInputShape);

        auto input_order = opset8::Constant::create(element::i64, {4}, {0, 2, 3, 1});
        auto transposeOp = std::make_shared<opset8::Transpose>(input1, input_order);

        auto shape_pattern = opset8::Constant::create(element::i32, {5}, {-1, 4, 512, 512, 64});
        auto reshapeOp = std::make_shared<opset8::Reshape>(transposeOp, shape_pattern, false);

        auto concatOp = ngraph::builder::makeConcat({reshapeOp, input2}, 4);

        // explicitly set the output name, to avoid global conflict
        transposeOp->set_friendly_name("Transpose_0");
        reshapeOp->set_friendly_name("Reshape_0");
        concatOp->set_friendly_name("Concat_0");

        ngraph::ResultVector results = {std::make_shared<ngraph::opset1::Result>(concatOp)};
        function = std::make_shared<ngraph::Function>(results, ParameterVector{input1, input2}, "sconcat_out");
    }
};

TEST_F(PermuteReorderReshapeGPUTest, smoke_PermuteReorderReshapeGPUTest) {
    Run();
}

}  // namespace
