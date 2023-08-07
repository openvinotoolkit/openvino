// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <type_traits>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "gmock/gmock.h"
#include "openvino/openvino.hpp"
#include "openvino/opsets/opset12.hpp"

namespace fft_base_test {
using namespace std;
using namespace ov;
using namespace op;
using namespace testing;

struct FFTConstantAxesAndConstantSignalSizeTestParams {
    PartialShape input_shape;
    Shape axes_shape;
    Shape signal_size_shape;
    PartialShape ref_output_shape;
    std::vector<int64_t> axes;
    std::vector<int64_t> signal_size;
};

template <class TOp>
class FFTConstantAxesAndConstantSignalSizeTest : public TypePropOpTest<TOp> {
public:
    std::vector<FFTConstantAxesAndConstantSignalSizeTestParams> test_params{
        FFTConstantAxesAndConstantSignalSizeTestParams{{2, 180, 180, 2}, {2}, Shape{}, {2, 180, 180, 2}, {1, 2}, {}},
        FFTConstantAxesAndConstantSignalSizeTestParams{{2, 180, 180, 2}, {2}, Shape{}, {2, 180, 180, 2}, {2, 0}, {}},
        FFTConstantAxesAndConstantSignalSizeTestParams{{16, 500, 180, 369, 2},
                                                       {3},
                                                       Shape{},
                                                       {16, 500, 180, 369, 2},
                                                       {0, 3, 1},
                                                       {}},
        FFTConstantAxesAndConstantSignalSizeTestParams{{2, 180, 180, Dimension(1, 18)},
                                                       {2},
                                                       Shape{},
                                                       {2, 180, 180, Dimension(1, 18)},
                                                       {1, 2},
                                                       {}},
        FFTConstantAxesAndConstantSignalSizeTestParams{{2, 180, Dimension(7, 500), 2},
                                                       {2},
                                                       Shape{},
                                                       {2, 180, Dimension(7, 500), 2},
                                                       {1, 2},
                                                       {}},
        FFTConstantAxesAndConstantSignalSizeTestParams{{2, 180, Dimension(7, 500), Dimension(1, 18)},
                                                       {2},
                                                       Shape{},
                                                       {2, 180, Dimension(7, 500), Dimension(1, 18)},
                                                       {1, 2},
                                                       {}},
        FFTConstantAxesAndConstantSignalSizeTestParams{{2, Dimension(7, 500), 180, 2},
                                                       {2},
                                                       Shape{},
                                                       {2, Dimension(7, 500), 180, 2},
                                                       {1, 2},
                                                       {}},
        FFTConstantAxesAndConstantSignalSizeTestParams{{2, Dimension(7, 500), 180, Dimension(1, 18)},
                                                       {2},
                                                       Shape{},
                                                       {2, Dimension(7, 500), 180, Dimension(1, 18)},
                                                       {1, 2},
                                                       {}},
        FFTConstantAxesAndConstantSignalSizeTestParams{{2, Dimension(7, 500), Dimension(7, 500), 2},
                                                       {2},
                                                       Shape{},
                                                       {2, Dimension(7, 500), Dimension(7, 500), 2},
                                                       {1, 2},
                                                       {}},
        FFTConstantAxesAndConstantSignalSizeTestParams{{2, Dimension(7, 500), Dimension(7, 500), Dimension(1, 18)},
                                                       {2},
                                                       Shape{},
                                                       {2, Dimension(7, 500), Dimension(7, 500), Dimension(1, 18)},
                                                       {1, 2},
                                                       {}},
        FFTConstantAxesAndConstantSignalSizeTestParams{{Dimension(0, 2), 180, 180, 2},
                                                       {2},
                                                       Shape{},
                                                       {Dimension(0, 2), 180, 180, 2},
                                                       {1, 2},
                                                       {}},
        FFTConstantAxesAndConstantSignalSizeTestParams{{Dimension(0, 2), 180, 180, Dimension(1, 18)},
                                                       {2},
                                                       Shape{},
                                                       {Dimension(0, 2), 180, 180, Dimension(1, 18)},
                                                       {1, 2},
                                                       {}},
        FFTConstantAxesAndConstantSignalSizeTestParams{{Dimension(0, 2), 180, Dimension(7, 500), 2},
                                                       {2},
                                                       Shape{},
                                                       {Dimension(0, 2), 180, Dimension(7, 500), 2},
                                                       {1, 2},
                                                       {}},
        FFTConstantAxesAndConstantSignalSizeTestParams{{Dimension(0, 2), 180, Dimension(7, 500), Dimension(1, 18)},
                                                       {2},
                                                       Shape{},
                                                       {Dimension(0, 2), 180, Dimension(7, 500), Dimension(1, 18)},
                                                       {1, 2},
                                                       {}},
        FFTConstantAxesAndConstantSignalSizeTestParams{{Dimension(0, 2), Dimension(7, 500), 180, 2},
                                                       {2},
                                                       Shape{},
                                                       {Dimension(0, 2), Dimension(7, 500), 180, 2},
                                                       {1, 2},
                                                       {}},
        FFTConstantAxesAndConstantSignalSizeTestParams{{Dimension(0, 2), Dimension(7, 500), 180, Dimension(1, 18)},
                                                       {2},
                                                       Shape{},
                                                       {Dimension(0, 2), Dimension(7, 500), 180, Dimension(1, 18)},
                                                       {1, 2},
                                                       {}},
        FFTConstantAxesAndConstantSignalSizeTestParams{{Dimension(0, 2), Dimension(7, 500), Dimension(7, 500), 2},
                                                       {2},
                                                       Shape{},
                                                       {Dimension(0, 2), Dimension(7, 500), Dimension(7, 500), 2},
                                                       {1, 2},
                                                       {}},
        FFTConstantAxesAndConstantSignalSizeTestParams{
            {Dimension(0, 2), Dimension(7, 500), Dimension(7, 500), Dimension(1, 18)},
            {2},
            Shape{},
            {Dimension(0, 2), Dimension(7, 500), Dimension(7, 500), Dimension(1, 18)},
            {1, 2},
            {}},
        FFTConstantAxesAndConstantSignalSizeTestParams{{2, 180, 180, 2}, {2}, {2}, {2, 180, 77, 2}, {1, 2}, {-1, 77}},
        FFTConstantAxesAndConstantSignalSizeTestParams{{2, 180, 180, 2},
                                                       {2},
                                                       {2},
                                                       {87, 180, 390, 2},
                                                       {2, 0},
                                                       {390, 87}},
        FFTConstantAxesAndConstantSignalSizeTestParams{{7, 50, 130, 400, 2},
                                                       {3},
                                                       {3},
                                                       {7, 40, 130, 600, 2},
                                                       {3, 0, 1},
                                                       {600, -1, 40}},
        FFTConstantAxesAndConstantSignalSizeTestParams{{2, Dimension(0, 200), 180, 2},
                                                       {2},
                                                       {2},
                                                       {2, Dimension(0, 200), 77, 2},
                                                       {1, 2},
                                                       {-1, 77}},
        FFTConstantAxesAndConstantSignalSizeTestParams{{Dimension(0, 18), 180, Dimension(0, 400), 2},
                                                       {2},
                                                       {2},
                                                       {87, 180, 390, 2},
                                                       {2, 0},
                                                       {390, 87}},
        FFTConstantAxesAndConstantSignalSizeTestParams{{Dimension(8, 129), 50, 130, Dimension(0, 500), 2},
                                                       {3},
                                                       {3},
                                                       {Dimension(8, 129), 40, 130, 600, 2},
                                                       {3, 0, 1},
                                                       {600, -1, 40}}};
};

TYPED_TEST_SUITE_P(FFTConstantAxesAndConstantSignalSizeTest);

TYPED_TEST_P(FFTConstantAxesAndConstantSignalSizeTest, dft_idft_constant_axes_and_signal_size) {
    for (auto params : this->test_params) {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, params.input_shape);
        auto axes_input = op::v0::Constant::create<int64_t>(element::i64, params.axes_shape, params.axes);

        std::shared_ptr<TypeParam> dft;
        if (params.signal_size.empty()) {
            dft = std::make_shared<TypeParam>(data, axes_input);
        } else {
            auto signal_size_input =
                op::v0::Constant::create<int64_t>(element::i64, params.signal_size_shape, params.signal_size);
            dft = std::make_shared<TypeParam>(data, axes_input, signal_size_input);
        }

        EXPECT_EQ(dft->get_element_type(), element::f32);
        EXPECT_EQ(dft->get_output_partial_shape(0), params.ref_output_shape);
    }
}

REGISTER_TYPED_TEST_SUITE_P(FFTConstantAxesAndConstantSignalSizeTest, dft_idft_constant_axes_and_signal_size);

using FFTBaseTypes = Types<op::v7::DFT, op::v7::IDFT>;
INSTANTIATE_TYPED_TEST_SUITE_P(type_prop, FFTConstantAxesAndConstantSignalSizeTest, FFTBaseTypes);

}  // namespace fft_base_test
