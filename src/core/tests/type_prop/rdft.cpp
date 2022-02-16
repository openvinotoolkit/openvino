//*****************************************************************************
// Copyright 2017-2022 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace ngraph;

struct RDFTConstantAxesAndConstantSignalSizeTestParams {
    PartialShape input_shape;
    Shape axes_shape;
    Shape signal_size_shape;
    PartialShape ref_output_shape;
    std::vector<int64_t> axes;
    std::vector<int64_t> signal_size;
};

struct RDFTConstantAxesAndConstantSignalSizeTest : ::testing::TestWithParam<RDFTConstantAxesAndConstantSignalSizeTestParams> {};

TEST_P(RDFTConstantAxesAndConstantSignalSizeTest, rdft_constant_axes_and_signal_size) {
    auto params = GetParam();

    auto data = std::make_shared<op::Parameter>(element::f32, params.input_shape);
    auto axes_input = op::Constant::create<int64_t>(element::i64, params.axes_shape, params.axes);

    std::shared_ptr<op::v9::RDFT> rdft;
    if (params.signal_size.empty()) {
        rdft = std::make_shared<op::v9::RDFT>(data, axes_input);
    } else {
        auto signal_size_input =
            op::Constant::create<int64_t>(element::i64, params.signal_size_shape, params.signal_size);
        rdft = std::make_shared<op::v9::RDFT>(data, axes_input, signal_size_input);
    }

    EXPECT_EQ(rdft->get_element_type(), element::f32);
    ASSERT_TRUE(rdft->get_output_partial_shape(0).same_scheme(params.ref_output_shape));
}

INSTANTIATE_TEST_SUITE_P(
    type_prop,
    RDFTConstantAxesAndConstantSignalSizeTest,
    ::testing::Values(
        RDFTConstantAxesAndConstantSignalSizeTestParams{{2, 180, 180}, {2}, Shape{}, {2, 91, 91, 2}, {1, 2}, {}},
        RDFTConstantAxesAndConstantSignalSizeTestParams{{2, 180, 180}, {2}, Shape{}, {2, 180, 91, 2}, {2, 0}, {}},
        RDFTConstantAxesAndConstantSignalSizeTestParams{{16, 500, 180, 369},
                                                        {3},
                                                        Shape{},
                                                        {9, 251, 180, 185, 2},
                                                        {0, 3, 1},
                                                        {}},
        RDFTConstantAxesAndConstantSignalSizeTestParams{{2, 180, Dimension(1, 18)},
                                                        {2},
                                                        Shape{},
                                                        {2, 91, Dimension(1, 18), 2},
                                                        {1, 2},
                                                        {}},
        RDFTConstantAxesAndConstantSignalSizeTestParams{{2, 180, Dimension(7, 500)},
                                                        {2},
                                                        Shape{},
                                                        {2, 91, Dimension(7, 500), 2},
                                                        {1, 2},
                                                        {}}
    ),
    PrintToDummyParamName());
