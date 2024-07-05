// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/subgraph_tests/src/classes/fuse_transpose_reorder.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/node_builders/convolution.hpp"
#include "common_test_utils/subgraph_builders/preprocess_builders.hpp"
#include "openvino/openvino.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

/*  FuseTransposeAndReorderTest3 graph
    Parameter
        \
         \
       Convolution (nhwc)
           \
            \  Parameter
             \ /
             Add
              |
          Transpose (0,2,3,1)
              |
            Result
*/

void FuseTransposeAndReorderTest3::create_model() {
    OPENVINO_ASSERT(input_shape.size() == 4);

    auto memFmt = nhwc;
    ov::op::PadType padType = ov::op::PadType::SAME_UPPER;
    ov::Shape kernel{3, 3}, stride{1, 1}, dilation{1, 1};
    std::vector<ptrdiff_t> padBegin{0, 0}, padEnd{0, 0};
    size_t convOutChannels = 32;

    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(in_prec, ov::Shape(input_shape))};
    OPENVINO_ASSERT(input_shape[1] >= 8 && (input_shape[1] % 8 == 0));
    auto convolutionNode = ov::test::utils::make_convolution(params.front(),
                                                             in_prec,
                                                             kernel,
                                                             stride,
                                                             padBegin,
                                                             padEnd,
                                                             dilation,
                                                             padType,
                                                             convOutChannels);
    convolutionNode->get_rt_info() = makeCPUInfo({memFmt}, {memFmt}, {});

    auto sndAddIn = std::make_shared<ov::op::v0::Parameter>(in_prec, convolutionNode->get_output_shape(0));
    params.push_back(sndAddIn);
    auto add = std::make_shared<ov::op::v1::Add>(convolutionNode->output(0), sndAddIn);

    auto order = std::vector<int64_t>{0, 2, 3, 1};
    auto constOrder = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{order.size()}, order);
    auto transpose = std::make_shared<ov::op::v1::Transpose>(add, constOrder);
    transpose->get_rt_info() = makeCPUInfo({memFmt}, {memFmt}, {});

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(transpose)};
    function = std::make_shared<ov::Model>(results, params, "TransposeReorder");
}

TEST_P(FuseTransposeAndReorderTest3, CompareWithRefs) {
    run();
    check_transpose_count(1);
}

const auto convSumTranposeParams = ::testing::Combine(::testing::Values(ov::Shape{1, 16, 32, 35}),
                                                      ::testing::Values(ov::element::f32)
);

INSTANTIATE_TEST_SUITE_P(smoke_Basic, FuseTransposeAndReorderTest3, convSumTranposeParams, FuseTransposeAndReorderTest::getTestCaseName);

}  // namespace test
}  // namespace ov
