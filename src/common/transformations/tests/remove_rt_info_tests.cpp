// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define _USE_MATH_DEFINES

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/runtime_attribute.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/remove_rt_info.hpp"

using namespace testing;
using namespace ov;
using namespace ov::op;

namespace {
class TRANSFORMATIONS_API TestRtAttr : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("TestRtAttr", "0");
};

void set_rt_info(const std::shared_ptr<Node>& node) {
    RTMap& rt_info = node->get_rt_info();
    rt_info["test_attr"] = "test_attr_value";
}
}  // namespace

TEST(TransformationTests, RemoveRtInfo) {
    std::shared_ptr<ov::Model> model, model_ref;
    {
        auto data = std::make_shared<v0::Parameter>(element::f32, Shape{2, 2});

        auto add1_const = v0::Constant::create(element::f32, Shape{1}, {1.0});
        auto add2_const = v0::Constant::create(element::f32, Shape{1}, {2.0});
        auto add1 = std::make_shared<v1::Add>(add1_const, add2_const);
        auto add2 = std::make_shared<v1::Add>(data, add1);

        set_rt_info(add1_const);
        set_rt_info(add1);

        model = std::make_shared<Model>(NodeVector{add2}, ParameterVector{data});
        model_ref = model->clone();

        pass::Manager manager;
        manager.register_pass<ov::pass::RemoveRtInfo>();
        manager.run_passes(model);
        ASSERT_THROW(check_rt_info(model), ov::Exception);
    }

    const auto ops = model->get_ops();
    auto it = std::find_if(ops.begin(), ops.end(), [](const std::shared_ptr<Node>& node) {
        RTMap& rt_info = node->get_rt_info();
        return !rt_info.empty();
    });
    ASSERT_EQ(it, ops.end()) << "found rt_info";

    const FunctionsComparator func_comparator = FunctionsComparator::with_default();
    const FunctionsComparator::Result result = func_comparator(model, model_ref);
    ASSERT_TRUE(result.valid);
}
