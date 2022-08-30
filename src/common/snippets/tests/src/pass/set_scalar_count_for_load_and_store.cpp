// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ngraph/function.hpp>
#include <ngraph/pass/manager.hpp>

#include <snippets/snippets_isa.hpp>
#include <snippets/pass/vector_to_scalar.hpp>

#include <transformations/init_node_info.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

//  todo: Rewrite this test using Snippets test infrastructure. See ./include/canonicalization.hpp for example

template<typename T>
size_t get_count(const std::shared_ptr<Function>& f, const std::string& name) {
    size_t load_count = std::numeric_limits<size_t>::max();
    for (auto op : f->get_ops()) {
        if (op->get_friendly_name() == name) {
            load_count = ov::as_type_ptr<T>(op)->get_count();
        }
    }
    return load_count;
}

TEST(TransformationTests, SetScalarCountForLoad) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    const auto count = 16;
    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 2});
        auto load = std::make_shared<snippets::isa::Load>(data, count);
        load->set_friendly_name("load");
        auto neg = std::make_shared<opset1::Negative>(load);
        auto store = std::make_shared<snippets::isa::Store>(neg, count);
        f = std::make_shared<Function>(NodeVector{store}, ParameterVector{data});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<snippets::pass::SetScalarCountForLoad>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 2});
        auto load = std::make_shared<snippets::isa::Load>(data, 1lu);
        load->set_friendly_name("load_ref");
        auto neg = std::make_shared<opset1::Negative>(load);
        auto store = std::make_shared<snippets::isa::Store>(neg, count);
        f_ref = std::make_shared<Function>(NodeVector{store}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;

    auto load_count = get_count<ngraph::snippets::op::Load>(f, "load");
    auto load_count_ref = get_count<ngraph::snippets::op::Load>(f_ref, "load_ref");
    ASSERT_EQ(load_count, load_count_ref);
}

TEST(TransformationTests, SetScalarCountForStore) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    const auto count = 16;
    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 2});
        auto load = std::make_shared<snippets::isa::Load>(data, count);
        auto neg = std::make_shared<opset1::Negative>(load);
        auto store = std::make_shared<snippets::isa::Store>(neg, count);
        store->set_friendly_name("store");
        f = std::make_shared<Function>(NodeVector{store}, ParameterVector{data});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<snippets::pass::SetScalarCountForStore>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 2});
        auto load = std::make_shared<snippets::isa::Load>(data, count);
        auto neg = std::make_shared<opset1::Negative>(load);
        auto store = std::make_shared<snippets::isa::Store>(neg, 1lu);
        store->set_friendly_name("store_ref");
        f_ref = std::make_shared<Function>(NodeVector{store}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;

    int64_t store_count = get_count<ngraph::snippets::op::Store>(f, "store");
    int64_t store_count_ref = get_count<ngraph::snippets::op::Store>(f_ref, "store_ref");
    ASSERT_EQ(store_count, store_count_ref);
}