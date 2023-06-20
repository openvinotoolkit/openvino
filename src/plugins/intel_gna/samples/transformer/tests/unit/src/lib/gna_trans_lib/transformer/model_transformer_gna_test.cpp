// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformer/model_transformer_gna.hpp"

#include <gtest/gtest.h>

#include <openvino/opsets/opset11.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <ops/pwl.hpp>
#include <transformations/pwl_approximation.hpp>
#include <transformations/utils/utils.hpp>

using namespace transformation_sample;

class ModelTransformerGNAFixture : public ::testing::Test {
protected:
    void SetUp() override;

    std::shared_ptr<ov::Model> m_model;
};

void ModelTransformerGNAFixture::SetUp() {
    const std::string input_friendly_name = "input_1";

    const size_t size = 1;
    const auto shape = ov::Shape{size};
    auto precision = ov::element::f32;
    auto parameter = std::make_shared<ov::opset11::Parameter>(precision, shape);
    auto activation = std::make_shared<ov::opset11::SoftSign>(parameter);
    auto result = std::make_shared<ov::opset11::Result>(activation);
    m_model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{parameter});
}

TEST_F(ModelTransformerGNAFixture, transform_specific) {
    TransformerConfiguration config = {{{"GNA_HW_EXECUTION_TARGET", "GNA_3_0"},
                                        {"GNA_PWL_MAX_ERROR_PERCENT", "1.0"},
                                        {"INFERENCE_PRECISION_HINT", "i16"}},
                                       {"ov::intel_gna::pass::PWLApproximation_0"}};

    std::shared_ptr<ModelTransformerGNA> transformer;
    EXPECT_NO_THROW({ transformer = std::make_shared<ModelTransformerGNA>(config); });
    EXPECT_NO_THROW(transformer->transform(m_model));
    EXPECT_TRUE(ov::op::util::has_op_with_type<ov::intel_gna::op::Pwl>(m_model));
}

TEST_F(ModelTransformerGNAFixture, transform_run_all) {
    TransformerConfiguration config;

    std::shared_ptr<ModelTransformerGNA> transformer;
    EXPECT_NO_THROW({ transformer = std::make_shared<ModelTransformerGNA>(config); });
    EXPECT_NO_THROW(transformer->transform(m_model));
}
