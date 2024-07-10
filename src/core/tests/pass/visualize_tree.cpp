// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/visualize_tree.hpp"

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"

namespace ov {
namespace test {

using ov::op::v0::Constant;
using ov::op::v0::Parameter;
using ov::op::v0::Result;
using ov::op::v1::Add;

class VisualizeTreeTest : public testing::Test {
protected:
    void TearDown() override {
        if (util::file_exists(vt_svg_file_path)) {
            std::remove(vt_svg_file_path.c_str());
        }

        if (util::file_exists(dot_file_path)) {
            std::remove(dot_file_path.c_str());
        }
    }

    static std::shared_ptr<Model> make_dummy_add_model(const element::Type& precision) {
        const auto c = Constant::create(precision, Shape{3}, {1.0f, -23.21f, std::numeric_limits<float>::infinity()});
        const auto input = std::make_shared<Parameter>(precision, Shape{});
        const auto add = std::make_shared<Add>(c, input);
        const auto output = std::make_shared<Result>(add);
        return std::make_shared<Model>(ResultVector{output}, ParameterVector{input});
    }

    const std::string vt_svg_file_path =
        util::path_join({utils::getExecutableDirectory(), utils::generateTestFilePrefix() + "_tree.svg"});
    const std::string dot_file_path = vt_svg_file_path + ".dot";
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
}  // namespace test
}  // namespace ov
