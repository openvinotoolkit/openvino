// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <behavior/compiled_model/model_cache.hpp>
#include <memory>

#include "common_test_utils/test_assertions.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "openvino/core/model_util.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/openvino.hpp"

namespace {

std::shared_ptr<ov::Model> createTestModel(const bool addWeightlessCacheAttribute = true,
                                           const bool alternativeWeights = false) {
    constexpr auto precision = ov::element::f32;

    const float weightsValue = !alternativeWeights ? 1.0f : 2.0f;
    auto weights = std::make_shared<ov::op::v0::Constant>(precision, ov::Shape{5}, std::vector<float>{weightsValue});
    auto input = std::make_shared<ov::op::v0::Parameter>(precision, ov::Shape{1});
    auto add = std::make_shared<ov::op::v1::Add>(input, weights);

    weights->set_friendly_name("weights");
    input->set_friendly_name("input");
    add->set_friendly_name("add");

    if (addWeightlessCacheAttribute) {
        weights->get_rt_info()[ov::WeightlessCacheAttribute::get_type_info_static()] =
            ov::WeightlessCacheAttribute(weights->get_byte_size(), 0, weights->get_element_type());
    }

    auto model = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input}, "Simple with weights");
    ov::util::set_tensors_names(ov::AUTO, *model, {}, {{0, {"add"}}});
    return model;
}

// This is a special model that has weightless constants that are guaranteed
// to be skipped by weights schedule. This tests cases where compiler
// produces "blob with weights" when "weightless blob" is requested: in
// theory, this may happen, and must not cause any errors.
std::shared_ptr<ov::Model> createTestModelWeightlessWithDummyConstants() {
    constexpr auto precision = ov::element::f32;

    const auto reshapeWeights =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{1, 2, 3});

    const auto input1 = std::make_shared<ov::op::v0::Parameter>(precision, ov::Shape{6});
    const auto input2 = std::make_shared<ov::op::v0::Parameter>(precision, ov::Shape{1, 2, 3});
    const auto reshapedInput1 = std::make_shared<ov::op::v1::Reshape>(input1, reshapeWeights, /*special_zero=*/false);
    auto add = std::make_shared<ov::op::v1::Add>(reshapedInput1, input2);

    reshapeWeights->set_friendly_name("weights");
    input1->set_friendly_name("input1");
    input2->set_friendly_name("input2");
    reshapedInput1->set_friendly_name("reshapedInput1");
    add->set_friendly_name("add");

    // Note: Reshape weights with weightless cache attribute satisfy the
    // basic requirement to create weights schedule. However, since this is
    // a static reshape, these weights would "disappear" during compilation,
    // causing the compiler to put nothing into the weights schedule.
    reshapeWeights->get_rt_info()[ov::WeightlessCacheAttribute::get_type_info_static()] =
        ov::WeightlessCacheAttribute(reshapeWeights->get_byte_size(), 0, reshapeWeights->get_element_type());

    auto model = std::make_shared<ov::Model>(ov::OutputVector{add},
                                             ov::ParameterVector{input1, input2},
                                             "Dummy weightless model");
    ov::util::set_tensors_names(ov::AUTO, *model, {}, {{0, {"add"}}});
    return model;
}

/**
 * @brief This model was fine-tuned in order to compile fast and yield a light init schedule.
 */
std::shared_ptr<ov::Model> createTestModelLightInitSchedule() {
    ov::ParameterVector parameter_vector;
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1});
    parameter_vector.push_back(data);
    auto add_constant = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{50}, {0.5});
    auto add = std::make_shared<ov::op::v1::Add>(data, add_constant);

    for (int i = 0; i < 10; i++) {
        auto intermediate_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1});
        parameter_vector.push_back(intermediate_input);
        add_constant = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{10, 10, 20, 50}, {i});
        add_constant->get_rt_info()[ov::WeightlessCacheAttribute::get_type_info_static()] =
            ov::WeightlessCacheAttribute(add_constant->get_byte_size(), i, add_constant->get_element_type());
        add = std::make_shared<ov::op::v1::Add>(add, intermediate_input);
        add = std::make_shared<ov::op::v1::Add>(add, add_constant);
    }

    return std::make_shared<ov::Model>(ov::OutputVector{add}, parameter_vector);
}

// This is a special test model with multiple constants using the same weights buffer.
std::shared_ptr<ov::Model> createTestModelWeightlessWithDuplicateConstants() {
    const std::vector<float> sharedData{1.0f, 2.0f, 3.0f, 4.0f};
    constexpr auto precision = ov::element::f32;

    auto input1 = std::make_shared<ov::op::v0::Parameter>(precision, ov::Shape{1, 1, 4});
    auto input2 = std::make_shared<ov::op::v0::Parameter>(precision, ov::Shape{1, 4, 1});

    auto weights1 = std::make_shared<ov::op::v0::Constant>(precision, ov::Shape{1, 1, 4}, sharedData);
    auto multiply1 = std::make_shared<ov::op::v1::Multiply>(input1, weights1);

    auto weights2 = std::make_shared<ov::op::v0::Constant>(precision, ov::Shape{1, 4, 1}, sharedData);
    auto multiply2 = std::make_shared<ov::op::v1::Multiply>(input2, weights2);

    auto reshapeWeights =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{1, 4, 1});
    auto reshape = std::make_shared<ov::op::v1::Reshape>(multiply1, reshapeWeights, false);

    auto add = std::make_shared<ov::op::v1::Add>(reshape, multiply2);

    input1->set_friendly_name("input1");
    input2->set_friendly_name("input2");
    weights1->set_friendly_name("weights");
    multiply1->set_friendly_name("multiply1");
    weights2->set_friendly_name("weights_new_shape");
    multiply2->set_friendly_name("multiply2");
    reshapeWeights->set_friendly_name("reshapeWeights");
    reshape->set_friendly_name("reshape");
    add->set_friendly_name("add");

    // Note: if this offset is changed, compiled_model->export_model() =>
    // core->import_model() would fail as the weightless bin offset is
    // "reset" through this boundary; right now this is by design
    constexpr size_t theOnlyFunctioningBinOffset = 0;
    weights1->get_rt_info()[ov::WeightlessCacheAttribute::get_type_info_static()] =
        ov::WeightlessCacheAttribute(weights1->get_byte_size(),
                                     theOnlyFunctioningBinOffset,
                                     weights1->get_element_type());
    weights2->get_rt_info()[ov::WeightlessCacheAttribute::get_type_info_static()] =
        ov::WeightlessCacheAttribute(weights2->get_byte_size(),
                                     theOnlyFunctioningBinOffset,
                                     weights2->get_element_type());

    auto model = std::make_shared<ov::Model>(ov::OutputVector{add},
                                             ov::ParameterVector{input1, input2},
                                             "duplicate_weights_model");
    ov::util::set_tensors_names(ov::AUTO, *model, {}, {{0, {"add"}}});
    return model;
}

}  // namespace

namespace ov {

namespace test {

namespace behavior {

using OVWeightlessCacheAccuracyNPU = WeightlessCacheAccuracy;

TEST_P(OVWeightlessCacheAccuracyNPU, SimpleTestModel) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    OV_ASSERT_NO_THROW(m_model = createTestModel());
    OV_ASSERT_NO_THROW(run());
}

TEST_P(OVWeightlessCacheAccuracyNPU, TestModelWeightlessWithDummyConstants) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    OV_ASSERT_NO_THROW(m_model = createTestModelWeightlessWithDummyConstants());
    OV_ASSERT_NO_THROW(run());
}

TEST_P(OVWeightlessCacheAccuracyNPU, TestModelLightInitSchedule) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    OV_ASSERT_NO_THROW(m_model = createTestModelLightInitSchedule());
    OV_ASSERT_NO_THROW(run());
}

TEST_P(OVWeightlessCacheAccuracyNPU, TestModelWeightlessWithDuplicateConstants) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    OV_ASSERT_NO_THROW(m_model = createTestModelWeightlessWithDuplicateConstants());
    OV_ASSERT_NO_THROW(run());
}

}  // namespace behavior

}  // namespace test

}  // namespace ov
