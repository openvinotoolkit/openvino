// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/parameter.hpp"

namespace {
// The added tensor is the input of the same convolution, so a sum post-op would let onednn
// accumulate into the buffer the convolution reads from.
class ConvolutionOwnInputSum : public testing::WithParamInterface<ov::Shape>,
                               virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ov::Shape>& params) {
        std::ostringstream result;
        result << "IS=" << ov::test::utils::partialShape2str({params.param});
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        const auto type = ov::element::f16;
        const auto shape = GetParam();
        // the kernel window accumulates many terms in f16, so keep the tolerance relative
        rel_threshold = 0.02f;

        auto input = std::make_shared<ov::op::v0::Parameter>(type, shape);
        auto weights = ov::op::v0::Constant::create(type, ov::Shape{shape[1], shape[1], 3, 3}, {1});
        auto conv = std::make_shared<ov::op::v1::Convolution>(input, weights, ov::Strides{1, 1},
                                                              ov::CoordinateDiff{1, 1}, ov::CoordinateDiff{1, 1},
                                                              ov::Strides{1, 1});
        auto sum = std::make_shared<ov::op::v1::Add>(conv, input);

        function = std::make_shared<ov::Model>(ov::OutputVector{sum}, ov::ParameterVector{input});
    }
};

TEST_P(ConvolutionOwnInputSum, Inference) {
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_GPU_Convolution, ConvolutionOwnInputSum,
                         ::testing::Values(ov::Shape{1, 1, 4, 4},
                                           ov::Shape{1, 16, 16, 16}),
                         ConvolutionOwnInputSum::getTestCaseName);

}  // namespace
