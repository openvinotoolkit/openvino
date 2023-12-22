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
    auto in = std::make_shared<ov::op::v0::Parameter>(ov::element::Type_t::f32,
                                                      ov::PartialShape{-1, static_cast<int>(inpChannel), -1, -1});
    in->set_friendly_name("in");
    auto in2 = std::make_shared<ov::op::v0::Parameter>(ov::element::Type_t::f32,
                                                       ov::Shape{1, outpChannel});
    in2->set_friendly_name("in2");
    auto newShape = ngraph::builder::makeConstant<int>(ov::element::Type_t::i32,
                                                       ov::Shape{4},
                                                       {1, static_cast<int>(outpChannel), 1, 1},
                                                       false);

    auto reshape = std::make_shared<ov::op::v1::Reshape>(in2, newShape, false);

    auto biasWeights = std::vector<float>();
    biasWeights.resize(outpChannel);
    auto conv = ov::test::utils::make_convolution(in,
                                                  ov::element::f32,
                                                  ov::Shape{3, 3},
                                                  ov::Shape{1, 1},
                                                  {1, 1},
                                                  {1, 1},
                                                  ov::Shape{1, 1},
                                                  ov::op::PadType::EXPLICIT,
                                                  outpChannel);

    auto biasNode = ngraph::builder::makeConstant<float>(ov::element::Type_t::f32, ov::Shape({1, outpChannel, 1, 1}), {}, true);
    conv = std::make_shared<ov::op::v1::Add>(conv, biasNode);
    conv = std::make_shared<ov::op::v1::Add>(conv, reshape);

    auto result = std::make_shared<ov::op::v0::Result>(conv->output(0));
    auto model = std::make_shared<ov::Model>(ov::NodeVector{result}, ov::ParameterVector{in, in2}, "ConvSumConst");

    auto inpShape = ov::Shape{1, inpChannel, 6, 6};
    auto inpBuf = new float(shape_size(inpShape));
    EXPECT_NE(inpBuf, nullptr);

    auto inpShape2 = ov::Shape{1, outpChannel};
    auto inpBuf2 = new float(shape_size(inpShape2));
    EXPECT_NE(inpBuf2, nullptr);

    auto core = ov::Core();
    std::map<size_t, ov::PartialShape> newInpShape{
        {static_cast<size_t>(0), ov::PartialShape{1, static_cast<int>(inpChannel), {6, 12}, {6, 12}}}};
    model->reshape(newInpShape);
    auto compiled_model = core.compile_model(model, "CPU");
    auto reqest = compiled_model.create_infer_request();

    ov::Tensor inpTensor1 = ov::Tensor(ov::element::f32, inpShape, inpBuf);
    ov::Tensor inpTensor2 = ov::Tensor(ov::element::f32, ov::Shape{1, outpChannel}, inpBuf2);
    reqest.set_input_tensor(0, inpTensor1);
    reqest.set_input_tensor(1, inpTensor2);
    ASSERT_NO_THROW(reqest.infer());
    if (inpBuf) {
        delete[] inpBuf;
    }
    if (inpBuf2) {
        delete[] inpBuf2;
    }
}

}  // namespace test
}  // namespace ov
