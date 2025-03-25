// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <queue>
#include <string>

#include "common_test_utils/graph_comparator.hpp"
#include "common_test_utils/ov_plugin_cache.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/parameter.hpp"
#include "template/properties.hpp"

TEST(DisableTransformationsTests, TestTemplatePluginProperty) {
    std::shared_ptr<ov::Model> m(nullptr), m_ref(nullptr);
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3, 1, 2});
        auto like = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
        auto cvtlike = std::make_shared<ov::op::v1::ConvertLike>(data, like);

        m = std::make_shared<ov::Model>(ov::NodeVector{cvtlike}, ov::ParameterVector{data});
    }
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3, 1, 2});
        auto cvt = std::make_shared<ov::op::v0::Convert>(data, ov::element::i32);

        m_ref = std::make_shared<ov::Model>(ov::NodeVector{cvt}, ov::ParameterVector{data});
    }

    auto core = ov::test::utils::PluginCache::get().core("TEMPLATE");

    auto transformed_comp_model = core->compile_model(m, "TEMPLATE");
    auto no_transformed_comp_model =
        core->compile_model(m, "TEMPLATE", ov::template_plugin::disable_transformations(true));

    // Clone is needed only for comparison
    auto transformed_model = transformed_comp_model.get_runtime_model()->clone();
    auto no_transformed_model = no_transformed_comp_model.get_runtime_model()->clone();

    auto res = compare_functions(m, m_ref);
    ASSERT_FALSE(res.first);
    res = compare_functions(transformed_model, no_transformed_model);
    ASSERT_FALSE(res.first);
    res = compare_functions(transformed_model, m_ref);
    ASSERT_TRUE(res.first) << res.second;
    res = compare_functions(no_transformed_model, m);
    ASSERT_TRUE(res.first) << res.second;
}
