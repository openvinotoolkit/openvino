// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/runtime_attribute.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/clean_rt_info.hpp"

using namespace testing;
using namespace ov;
using namespace ov::op;
using namespace ov::pass;

namespace {

class TestAttrVisitTrue : public RuntimeAttribute {
public:
    TestAttrVisitTrue() = default;

    bool visit_attributes(AttributeVisitor& visitor) override {
        return true;
    }
};

class TestAttrVisitFalse : public RuntimeAttribute {
public:
    TestAttrVisitFalse() = default;

    bool visit_attributes(AttributeVisitor& visitor) override {
        return false;
    }
};

void set_rt_info(const std::shared_ptr<Node>& node, const std::string& key, const Any& value) {
    RTMap& rt_info = node->get_rt_info();
    rt_info[key] = value;
}

}  // namespace

TEST(TransformationTests, RemoveRtInfo) {
    std::shared_ptr<Model> model, model_ref;
    {
        auto data = std::make_shared<v0::Parameter>(element::f32, Shape{2, 2});

        auto add1_const = v0::Constant::create(element::f32, Shape{1}, {1.0f});
        auto add2_const = v0::Constant::create(element::f32, Shape{1}, {2.0f});
        auto add1 = std::make_shared<v1::Add>(add1_const, add2_const);
        auto add2 = std::make_shared<v1::Add>(data, add1);

        set_rt_info(add1, "test_attr_str", "test_attr_value");
        set_rt_info(add1, "test_attr_visit_true", TestAttrVisitTrue());
        set_rt_info(add2, "test_attr_visit_false", TestAttrVisitFalse());

        model = std::make_shared<Model>(NodeVector{add2}, ParameterVector{data});

        Manager manager;
        manager.register_pass<CleanRtInfo>();
        manager.run_passes(model);
    }
    {
        auto data = std::make_shared<v0::Parameter>(element::f32, Shape{2, 2});

        auto add1_const = v0::Constant::create(element::f32, Shape{1}, {1.0f});
        auto add2_const = v0::Constant::create(element::f32, Shape{1}, {2.0f});
        auto add1 = std::make_shared<v1::Add>(add1_const, add2_const);
        auto add2 = std::make_shared<v1::Add>(data, add1);

        set_rt_info(add1, "test_attr_visit_true", TestAttrVisitTrue());

        model_ref = std::make_shared<Model>(NodeVector{add2}, ParameterVector{data});
    }

    FunctionsComparator func_comparator = FunctionsComparator::with_default();
    func_comparator.enable(FunctionsComparator::RUNTIME_KEYS);
    const FunctionsComparator::Result result = func_comparator(model, model_ref);
    ASSERT_TRUE(result.valid) << result.message;
}
