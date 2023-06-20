// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformer/model_transformer_app.hpp"

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <openvino/cc/pass/itt.hpp>
#include <openvino/core/rt_info.hpp>
#include <openvino/opsets/opset11.hpp>
#include <openvino/pass/graph_rewrite.hpp>
#include <openvino/pass/pattern/matcher.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>

using namespace transformation_sample;

class DummyPass : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("DummyPass", "0");
    DummyPass(bool& invoked);
};

DummyPass::DummyPass(bool& invoked) {
    MATCHER_SCOPE(DummyPass);

    const auto pattern = ov::pass::pattern::any_input();

    ov::matcher_pass_callback callback = [&invoked](ov::pass::pattern::Matcher& m) {
        invoked = true;
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(pattern, matcher_name);

    this->register_matcher(m, callback);
}

class ModelTransformerAppFixture : public ::testing::Test {
protected:
    void SetUp() override;

    std::shared_ptr<ov::Model> m_model;
};

void ModelTransformerAppFixture::SetUp() {
    const std::string input_friendly_name = "input_1";

    size_t size = 1;
    const ov::element::Type precision = ov::element::f32;
    auto shape = ov::Shape{size};

    auto input_param = std::make_shared<ov::opset11::Parameter>(precision, shape);

    std::vector<float> vector_data(size, 1.0);
    auto constant = std::make_shared<ov::opset11::Constant>(precision, shape, vector_data);
    auto add = std::make_shared<ov::opset11::Add>(input_param, constant);

    auto result = std::make_shared<ov::opset11::Result>(add);

    m_model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input_param});
}

TEST_F(ModelTransformerAppFixture, transform) {
    bool invoked = false;
    auto transformation = std::make_shared<DummyPass>(invoked);

    std::shared_ptr<ModelTransformerApp> transformer;
    EXPECT_NO_THROW({ transformer = std::make_shared<ModelTransformerApp>(transformation); });
    EXPECT_NO_THROW(transformer->transform(m_model));
    EXPECT_TRUE(invoked);
}

