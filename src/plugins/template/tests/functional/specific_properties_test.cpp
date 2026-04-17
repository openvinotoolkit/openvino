// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/ov_plugin_cache.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/util/common_util.hpp"

namespace ov::test {

std::shared_ptr<ov::Model> make_simple_model() {
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3});
    auto relu = std::make_shared<ov::op::v0::Relu>(data);
    auto result = std::make_shared<ov::op::v0::Result>(relu);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{data});
}

TEST(PropertyTest, CompiledModelExposesRuntimeRequirements) {
    auto core = ov::test::utils::PluginCache::get().core("TEMPLATE");
    auto compiled = core->compile_model(make_simple_model(), "TEMPLATE");

    const auto supported = compiled.get_property(ov::supported_properties);
    ASSERT_TRUE(ov::util::contains(supported, ov::runtime_requirements.name()));

    const auto reqs = compiled.get_property(ov::runtime_requirements);
    EXPECT_TRUE(reqs);
    EXPECT_EQ(reqs.get_size(), 1U);
    EXPECT_EQ(reqs.get_element_type(), element::string);
}

TEST(PropertyTest, PluginReportsRequirementsMetForValidRequirements) {
    auto core = ov::test::utils::PluginCache::get().core("TEMPLATE");
    auto compiled = core->compile_model(make_simple_model(), "TEMPLATE");

    const auto reqs = compiled.get_property(ov::runtime_requirements);
    ASSERT_TRUE(reqs);

    const auto compat = core->get_property("TEMPLATE", ov::blob_compatibility, ov::runtime_requirements(reqs));
    EXPECT_EQ(compat, ov::BlobCompatibility::OPTIMAL);
}

TEST(PropertyTest, PluginRejectsModifiedRequirements) {
    auto core = ov::test::utils::PluginCache::get().core("TEMPLATE");
    auto compiled = core->compile_model(make_simple_model(), "TEMPLATE");

    const auto reqs = compiled.get_property(ov::runtime_requirements);
    ASSERT_TRUE(reqs);

    ov::Tensor tampered(ov::element::string, {});
    tampered.data<std::string>()[0] = "_tampered";

    EXPECT_EQ(core->get_property("TEMPLATE", ov::blob_compatibility, ov::runtime_requirements(tampered)),
              ov::BlobCompatibility::UNSUPPORTED);
}

TEST(PropertyTest, PluginAcceptModifiedRequirements) {
    auto core = ov::test::utils::PluginCache::get().core("TEMPLATE");
    auto compiled = core->compile_model(make_simple_model(), "TEMPLATE");

    const auto reqs = compiled.get_property(ov::runtime_requirements);
    ASSERT_TRUE(reqs);

    ov::Tensor tampered(ov::element::string, {});
    tampered.data<std::string>()[0] = "tampered_" + reqs.data<std::string>()[0];

    EXPECT_EQ(core->get_property("TEMPLATE", ov::blob_compatibility, ov::runtime_requirements(tampered)),
              ov::BlobCompatibility::PREFER_RECOMPILATION);
}

TEST(PropertyTest, PluginRejectsEmptyRequirements) {
    auto core = ov::test::utils::PluginCache::get().core("TEMPLATE");

    const ov::Tensor empty_reqs;
    EXPECT_EQ(core->get_property("TEMPLATE", ov::blob_compatibility, ov::runtime_requirements(empty_reqs)),
              ov::BlobCompatibility::NOT_APPLICABLE);
}

TEST(PropertyTest, PluginReturnsNotApplicableWithoutArguments) {
    auto core = ov::test::utils::PluginCache::get().core("TEMPLATE");
    const auto compat = core->get_property("TEMPLATE", ov::blob_compatibility);
    EXPECT_EQ(compat, ov::BlobCompatibility::NOT_APPLICABLE);
}

TEST(PropertyTest, RuntimeRequirementsMetListedInSupportedProperties) {
    auto core = ov::test::utils::PluginCache::get().core("TEMPLATE");
    const auto supported = core->get_property("TEMPLATE", ov::supported_properties);
    EXPECT_TRUE(ov::util::contains(supported, ov::blob_compatibility.name()));
}
}  // namespace ov::test
