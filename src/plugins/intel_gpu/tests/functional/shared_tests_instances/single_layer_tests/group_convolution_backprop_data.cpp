// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/group_convolution_backprop_data.hpp"
#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {

class GroupConvBackpropDilationTest : public GroupConvBackpropLayerTest {
protected:
    void SetUp() override {
        GroupConvBackpropLayerTest::SetUp();
        // FP16 deconv with dilation has inherent quantization error (~0.25 for values >128)
        const auto& [convParams, model_type, shapes, output_shapes, dev] = this->GetParam();
        if (model_type == ov::element::f16) {
            abs_threshold = 0.5;
            rel_threshold = 0.01;
        }
    }
};

TEST_P(GroupConvBackpropDilationTest, Inference) {
    run();
}

}  // namespace test
}  // namespace ov

namespace {
using ov::test::GroupConvBackpropLayerTest;

const std::vector<ov::element::Type> netPrecisions = {
        ov::element::f32
};

const std::vector<ov::Shape> emptyOutputShape = {{}};
const std::vector<std::vector<ptrdiff_t>> emptyOutputPadding = {{}};

const std::vector<size_t> numOutChannels = {16, 32};
const std::vector<size_t> numGroups = {2, 8, 16};

/* ============= 2D GroupConvolution ============= */
const std::vector<std::vector<ov::Shape>> inputShapes2D = {
        {{1, 16, 10, 10}},
        {{1, 32, 10, 10}}
};

const std::vector<std::vector<size_t >> kernels2D = {{1, 1}, {3, 3}};
const std::vector<std::vector<size_t >> strides2D = {{1, 1}};
const std::vector<std::vector<ptrdiff_t>> padBegins2D = {{0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds2D = {{0, 0}};
const std::vector<std::vector<size_t >> dilations2D = {{1, 1}};

const auto groupConvBackpropData2DParams_ExplicitPadding = ::testing::Combine(
        ::testing::ValuesIn(kernels2D),
        ::testing::ValuesIn(strides2D),
        ::testing::ValuesIn(padBegins2D),
        ::testing::ValuesIn(padEnds2D),
        ::testing::ValuesIn(dilations2D),
        ::testing::ValuesIn(numOutChannels),
        ::testing::ValuesIn(numGroups),
        ::testing::Values(ov::op::PadType::EXPLICIT),
        ::testing::ValuesIn(emptyOutputPadding)
);
const auto groupConvBackpropData2DParams_AutoPadValid = ::testing::Combine(
        ::testing::ValuesIn(kernels2D),
        ::testing::ValuesIn(strides2D),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::ValuesIn(dilations2D),
        ::testing::ValuesIn(numOutChannels),
        ::testing::ValuesIn(numGroups),
        ::testing::Values(ov::op::PadType::VALID),
        ::testing::ValuesIn(emptyOutputPadding)
);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvBackpropData2D_ExplicitPadding, GroupConvBackpropLayerTest,
                        ::testing::Combine(
                                groupConvBackpropData2DParams_ExplicitPadding,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes2D)),
                                ::testing::ValuesIn(emptyOutputShape),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        GroupConvBackpropLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvBackpropData2D_AutoPadValid, GroupConvBackpropLayerTest,
                        ::testing::Combine(
                                groupConvBackpropData2DParams_AutoPadValid,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes2D)),
                                ::testing::ValuesIn(emptyOutputShape),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        GroupConvBackpropLayerTest::getTestCaseName);

/* ============= 3D GroupConvolution ============= */
const std::vector<std::vector<ov::Shape>> inputShapes3D = {
        {{1, 16, 5, 5, 5}},
        {{1, 32, 5, 5, 5}}
};

const std::vector<std::vector<size_t >> kernels3D = {{1, 1, 1}, {3, 3, 3}};
const std::vector<std::vector<size_t >> strides3D = {{1, 1, 1}};
const std::vector<std::vector<ptrdiff_t>> padBegins3D = {{0, 0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds3D = {{0, 0, 0}};
const std::vector<std::vector<size_t >> dilations3D = {{1, 1, 1}};

const auto groupConvBackpropData3DParams_ExplicitPadding = ::testing::Combine(
        ::testing::ValuesIn(kernels3D),
        ::testing::ValuesIn(strides3D),
        ::testing::ValuesIn(padBegins3D),
        ::testing::ValuesIn(padEnds3D),
        ::testing::ValuesIn(dilations3D),
        ::testing::ValuesIn(numOutChannels),
        ::testing::ValuesIn(numGroups),
        ::testing::Values(ov::op::PadType::EXPLICIT),
        ::testing::ValuesIn(emptyOutputPadding)
);
const auto groupConvBackpropData3DParams_AutoPadValid = ::testing::Combine(
        ::testing::ValuesIn(kernels3D),
        ::testing::ValuesIn(strides3D),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),
        ::testing::ValuesIn(dilations3D),
        ::testing::ValuesIn(numOutChannels),
        ::testing::ValuesIn(numGroups),
        ::testing::Values(ov::op::PadType::VALID),
        ::testing::ValuesIn(emptyOutputPadding)
);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvBackpropData3D_ExplicitPadding, GroupConvBackpropLayerTest,
                        ::testing::Combine(
                                groupConvBackpropData3DParams_ExplicitPadding,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes3D)),
                                ::testing::ValuesIn(emptyOutputShape),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        GroupConvBackpropLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvBackpropData3D_AutoPadValid, GroupConvBackpropLayerTest,
                        ::testing::Combine(
                                groupConvBackpropData3DParams_AutoPadValid,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes3D)),
                                ::testing::ValuesIn(emptyOutputShape),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        GroupConvBackpropLayerTest::getTestCaseName);

/* ============= 2D GroupConvolutionBackpropData with Dilation > 1 ============= */
const std::vector<ov::element::Type> netPrecisionsDilation = {ov::element::f32, ov::element::f16};

const std::vector<std::vector<ov::Shape>> inputShapesDilation2D = {{{1, 16, 5, 5}}, {{1, 32, 10, 10}}};
const std::vector<std::vector<size_t>> kernelsDilation2D = {{2, 2}, {3, 3}};
const std::vector<std::vector<size_t>> stridesDilation2D = {{1, 1}};
const std::vector<std::vector<ptrdiff_t>> padBeginsDilation2D = {{0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEndsDilation2D = {{0, 0}};
const std::vector<std::vector<size_t>> dilationsDilation2D = {{2, 2}};
const std::vector<size_t> numOutChannelsDilation = {16, 32};
const std::vector<size_t> numGroupsDilation = {2, 8};

const auto groupConvBackpropData2DParams_Dilation = ::testing::Combine(::testing::ValuesIn(kernelsDilation2D),
                                                                       ::testing::ValuesIn(stridesDilation2D),
                                                                       ::testing::ValuesIn(padBeginsDilation2D),
                                                                       ::testing::ValuesIn(padEndsDilation2D),
                                                                       ::testing::ValuesIn(dilationsDilation2D),
                                                                       ::testing::ValuesIn(numOutChannelsDilation),
                                                                       ::testing::ValuesIn(numGroupsDilation),
                                                                       ::testing::Values(ov::op::PadType::EXPLICIT),
                                                                       ::testing::ValuesIn(emptyOutputPadding));

using ov::test::GroupConvBackpropDilationTest;
INSTANTIATE_TEST_SUITE_P(smoke_GroupConvBackpropData2D_Dilation,
                         GroupConvBackpropDilationTest,
                         ::testing::Combine(groupConvBackpropData2DParams_Dilation,
                                            ::testing::ValuesIn(netPrecisionsDilation),
                                            ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapesDilation2D)),
                                            ::testing::ValuesIn(emptyOutputShape),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         GroupConvBackpropLayerTest::getTestCaseName);

}  // namespace
