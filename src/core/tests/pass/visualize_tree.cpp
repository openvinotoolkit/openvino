// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/visualize_tree.hpp"

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/if.hpp"

namespace ov::test {

using ov::op::v0::Constant;
using ov::op::v0::Parameter;
using ov::op::v0::Result;
using ov::op::v1::Add;
using ov::op::v8::If;

class VisualizeTreeTest : public testing::Test {
protected:
    void TearDown() override {
        if (util::file_exists(vt_svg_file_path)) {
            std::filesystem::remove(vt_svg_file_path);
        }

        if (util::file_exists(dot_file_path)) {
            std::filesystem::remove(dot_file_path);
        }
    }

    static std::shared_ptr<Model> make_dummy_add_model(const element::Type& precision) {
        const auto c = Constant::create(precision, Shape{3}, {1.0f, -23.21f, std::numeric_limits<float>::infinity()});
        const auto input = std::make_shared<Parameter>(precision, Shape{});
        const auto add = std::make_shared<Add>(c, input);
        const auto output = std::make_shared<Result>(add);
        return std::make_shared<Model>(ResultVector{output}, ParameterVector{input});
    }

    // Model with a MultiSubGraphOp (If), so its friendly name drives a subgraph dump file name.
    static std::shared_ptr<Model> make_dummy_if_model(const std::string& if_friendly_name) {
        const auto x = std::make_shared<Parameter>(element::f32, Shape{1});
        const auto y = std::make_shared<Parameter>(element::f32, Shape{1});
        const auto cond = std::make_shared<Constant>(element::boolean, Shape{}, true);

        const auto xt = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
        const auto yt = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
        const auto then_res = std::make_shared<Result>(std::make_shared<Add>(xt, yt));
        const auto then_body = std::make_shared<Model>(OutputVector{then_res}, ParameterVector{xt, yt});

        const auto xe = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
        const auto ye = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
        const auto else_res = std::make_shared<Result>(std::make_shared<Add>(xe, ye));
        const auto else_body = std::make_shared<Model>(OutputVector{else_res}, ParameterVector{xe, ye});

        const auto if_op = std::make_shared<If>(cond);
        if_op->set_friendly_name(if_friendly_name);
        if_op->set_then_body(then_body);
        if_op->set_else_body(else_body);
        if_op->set_input(x, xt, xe);
        if_op->set_input(y, yt, ye);
        const auto out = if_op->set_output(then_res, else_res);
        const auto output = std::make_shared<Result>(out);
        return std::make_shared<Model>(ResultVector{output}, ParameterVector{x, y});
    }

    const std::filesystem::path vt_svg_file_path =
        ov::util::make_path(utils::getExecutableDirectory()) / (utils::generateTestFilePrefix() + "_tree.svg");
    const std::filesystem::path dot_file_path = vt_svg_file_path.string() + ".dot";
};

TEST_F(VisualizeTreeTest, model_has_constant_with_inf) {
    constexpr auto precision = element::f32;
    const auto model = make_dummy_add_model(precision);

    pass::VisualizeTree vt(vt_svg_file_path);

    OV_ASSERT_NO_THROW(vt.run_on_model(model));
    ASSERT_TRUE(util::file_exists(dot_file_path)) << dot_file_path;
}

TEST_F(VisualizeTreeTest, model_has_constant_with_no_inf) {
    constexpr auto precision = element::f16;
    const auto model = make_dummy_add_model(precision);

    pass::VisualizeTree vt(vt_svg_file_path);

    OV_ASSERT_NO_THROW(vt.run_on_model(model));
    ASSERT_TRUE(util::file_exists(dot_file_path)) << dot_file_path;
}

struct VisualizeTreeSanitizeParam {
    std::string friendly_name;
    std::string sanitized_name;
};

class VisualizeTreeSanitizeTest : public VisualizeTreeTest,
                                  public testing::WithParamInterface<VisualizeTreeSanitizeParam> {};

// CWE-31: any character outside the allowlist [A-Za-z0-9._-] must be neutralized the same way,
// covering both the Windows path separator '\' and the NTFS alternate-data-stream separator ':',
// and the dump must still succeed (no exception).
TEST_P(VisualizeTreeSanitizeTest, subgraph_friendly_name_with_disallowed_characters_is_sanitized) {
    const auto& param = GetParam();
    const auto model = make_dummy_if_model(param.friendly_name);

    pass::VisualizeTree vt(vt_svg_file_path);

    OV_ASSERT_NO_THROW(vt.run_on_model(model));
    ASSERT_TRUE(util::file_exists(dot_file_path)) << dot_file_path;

    auto subgraph_file_path = vt_svg_file_path;
    subgraph_file_path.replace_extension("._node_" + param.sanitized_name + "_subgraph_#0");
    subgraph_file_path += ".dot";
    ASSERT_TRUE(util::file_exists(subgraph_file_path)) << subgraph_file_path;
    std::filesystem::remove(subgraph_file_path);
}

INSTANTIATE_TEST_SUITE_P(CWE31,
                         VisualizeTreeSanitizeTest,
                         testing::Values(VisualizeTreeSanitizeParam{"x\\..\\..\\..\\Users\\Public\\kb_poc_visualize",
                                                                    "x_.._.._.._Users_Public_kb_poc_visualize"},
                                         VisualizeTreeSanitizeParam{"evil:stream", "evil_stream"}));

// Regression: an allowed friendly name still dumps the subgraph file as before.
TEST_F(VisualizeTreeTest, subgraph_friendly_name_with_safe_characters_is_dumped) {
    const auto model = make_dummy_if_model("safe_name-1.2");

    pass::VisualizeTree vt(vt_svg_file_path);

    OV_ASSERT_NO_THROW(vt.run_on_model(model));
    ASSERT_TRUE(util::file_exists(dot_file_path)) << dot_file_path;

    auto subgraph_file_path = vt_svg_file_path;
    subgraph_file_path.replace_extension("._node_safe_name-1.2_subgraph_#0");
    subgraph_file_path += ".dot";
    ASSERT_TRUE(util::file_exists(subgraph_file_path)) << subgraph_file_path;
    std::filesystem::remove(subgraph_file_path);
}
}  // namespace ov::test
