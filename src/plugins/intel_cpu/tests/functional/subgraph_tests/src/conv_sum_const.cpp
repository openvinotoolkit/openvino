// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/activation.hpp"
#include "common_test_utils/node_builders/convolution.hpp"
#include "internal_properties.hpp"
#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "test_utils/convolution_params.hpp"
#include "test_utils/fusing_test_utils.hpp"

#include <regex>

using namespace CPUTestUtils;

namespace ov {
namespace test {

TEST(smoke_Basic, ConvSumConstTest) {
    size_t inpChannel = 8;
    size_t outpChannel = 8;
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::Type_t::f32, ov::PartialShape{-1, static_cast<int>(inpChannel), -1, -1});
    auto reshapeInput = ngraph::builder::makeConstant<float>(ov::element::Type_t::f32, ov::Shape{1, outpChannel}, {}, true);
    auto reshapeOutput = ngraph::builder::makeConstant<int32_t>(ov::element::Type_t::i32,
                                                                ov::Shape{4},
                                                                {1, static_cast<int>(outpChannel), 1, 1},
                                                                false);
    // auto reshapeOutput = std::make_shared<ov::op::v0::Parameter>(ov::element::Type_t::i32, ov::PartialShape{1, static_cast<int>(outpChannel), 1, 1});

    auto reshape = std::make_shared<ov::op::v1::Reshape>(reshapeInput, reshapeOutput, false);

    auto conv = ov::test::utils::make_convolution(input,
                                                  ov::element::f32,
                                                  ov::Shape{3, 3},
                                                  ov::Shape{1, 1},
                                                  {1, 1},
                                                  {1, 1},
                                                  ov::Shape{1, 1},
                                                  ov::op::PadType::EXPLICIT,
                                                  outpChannel);

    // Add bias
    auto biasNode = ngraph::builder::makeConstant<float>(ov::element::Type_t::f32, ov::Shape({1, outpChannel, 1, 1}), {}, true);
    auto sum1 = std::make_shared<ov::op::v1::Add>(conv, biasNode);
    auto sum2 = std::make_shared<ov::op::v1::Add>(sum1, reshape);

    auto result = std::make_shared<ov::op::v0::Result>(sum2->output(0));
    auto model = std::make_shared<ov::Model>(ov::NodeVector{result}, ov::ParameterVector{input}, "ConvSumConst");

    auto inpShape = ov::Shape{1, inpChannel, 6, 6};
    auto inpBuf = new float(shape_size(inpShape));
    EXPECT_NE(inpBuf, nullptr);

    auto core = ov::Core();
    model->reshape({1, static_cast<int>(inpChannel), {6, 12}, {6, 12}});
    auto compiled_model = core.compile_model(model, "CPU");
    auto reqest = compiled_model.create_infer_request();



    ov::Tensor inpTensor = ov::Tensor(ov::element::f32, inpShape, inpBuf);
    reqest.set_input_tensor(inpTensor);
    ASSERT_NO_THROW(reqest.infer());
    if (inpBuf) {
        delete[] inpBuf;
    }
}

}  // namespace test
}  // namespace ov
