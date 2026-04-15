// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "embedding/remove_empty_kv_inputs.hpp"
#include "llm_pass_test_fixture.hpp"
#include "openvino/opsets/opset13.hpp"

namespace {

class RemoveEmptyKVInputsPassTest : public ov::test::npuw::LLMPassTestFixture {};

// Direct unit test for RemoveEmptyKVInputs: only the minimal past/current KV concat subgraph
// is needed. This avoids relying on LLMCompiledModel orchestration to validate the matcher.
TEST_F(RemoveEmptyKVInputsPassTest, HandlesDownUpProjSubgraph) {
    using namespace ov::opset13;

    auto past_k = std::make_shared<Parameter>(ov::element::f8e4m3, ov::Shape{1, 4, 0, 16});
    past_k->set_friendly_name("past_key_values.0.key");
    past_k->output(0).set_names({"past_key_values.0.key"});

    auto current_k = std::make_shared<Parameter>(ov::element::f8e4m3, ov::Shape{1, 4, 1, 16});
    current_k->set_friendly_name("current_key_values.0.key");
    current_k->output(0).set_names({"current_key_values.0.key"});

    auto upconvert = std::make_shared<Convert>(past_k, ov::element::f32);
    auto upscale = Constant::create(ov::element::f32, ov::Shape{}, {1.0f});
    auto upmul = std::make_shared<Multiply>(upconvert, upscale);
    auto downscale = Constant::create(ov::element::f32, ov::Shape{}, {1.0f});
    auto downmul = std::make_shared<Multiply>(upmul, downscale);
    auto downconvert = std::make_shared<Convert>(downmul, ov::element::f8e4m3);

    auto concat = std::make_shared<Concat>(ov::OutputVector{downconvert, current_k}, 2);
    concat->set_friendly_name("past_key_values.0.keypresent.0.key_concat");

    // ShapeOf on the original parameter exercises the replacement-to-constant path.
    auto shapeof = std::make_shared<ShapeOf>(past_k, ov::element::i64);
    auto concat_result = std::make_shared<Result>(concat);
    concat_result->output(0).set_names({"present.0.key"});
    auto shape_result = std::make_shared<Result>(shapeof);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{concat_result, shape_result},
                                             ov::ParameterVector{past_k, current_k},
                                             "remove_empty_kv_inputs_test");

    ov::npuw::RemoveEmptyKVInputs pass;
    EXPECT_TRUE(pass.run_on_model(model));

    EXPECT_EQ(model->get_parameters().size(), 1u);
    EXPECT_EQ(model->get_parameters().front()->get_friendly_name(), "current_key_values.0.key");
    EXPECT_EQ(count_ops<ov::op::v0::Concat>(model), 0u);
    EXPECT_EQ(count_ops<ov::op::v3::ShapeOf>(model), 0u);

    const auto outputs = model->outputs();
    ASSERT_EQ(outputs.size(), 2u);
    EXPECT_EQ(outputs[0].get_partial_shape(), ov::PartialShape({1, 4, 1, 16}));
}

}  // namespace
