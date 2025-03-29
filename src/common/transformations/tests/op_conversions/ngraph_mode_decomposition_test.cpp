// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/op_conversions/convert_mod.hpp"
using namespace ov;
using namespace testing;

TEST(TransformationTests, ModDecompositionTests) {
    auto data1 = op::v0::Constant::create(element::f32, Shape{1, 1, 3}, {1, 2, 3});
    auto data2 = op::v0::Constant::create(element::f32, Shape{3}, {1, 2, 3});

    std::shared_ptr<ov::Model> f(nullptr);
    {
        auto mod = std::make_shared<op::v1::Mod>(data1, data2);

        f = std::make_shared<ov::Model>(ov::NodeVector{mod}, ParameterVector{});
        auto unh = std::make_shared<ov::pass::UniqueNamesHolder>();
        pass::Manager m;
        m.register_pass<ov::pass::InitUniqueNames>(unh);
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::ConvertMod>();
        m.register_pass<ov::pass::CheckUniqueNames>(unh);
        m.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }
    ASSERT_EQ(f->get_ops().size(), 12);
}
