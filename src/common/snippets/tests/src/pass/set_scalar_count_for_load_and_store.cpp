// Copyright (C) 2018-2023 Intel Corporation
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

size_t get_count(const std::shared_ptr<Function>& f, const std::string& name, bool is_load = true) {
    size_t count = std::numeric_limits<size_t>::max();
    for (auto op : f->get_ops()) {
        if (op->get_friendly_name() == name) {
            if (const auto memory_access = std::dynamic_pointer_cast<snippets::op::MemoryAccess>(op)) {
                count = is_load ? memory_access->get_input_offset(0)
                                : memory_access->get_output_offset(0);
            }
        }
    }
    return count;
}

TEST(TransformationTests, SetScalarCountForLoadStore) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    const auto count = 16;
    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 2});
        auto load = std::make_shared<snippets::isa::Load>(data, count);
        load->set_friendly_name("load");
        auto neg = std::make_shared<opset1::Negative>(load);
        auto store = std::make_shared<snippets::isa::Store>(neg, count);
        store->set_friendly_name("store");
        f = std::make_shared<Function>(NodeVector{store}, ParameterVector{data});

        pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<snippets::pass::SetScalarCountForLoad>();
        m.register_pass<snippets::pass::SetScalarCountForStore>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 2});
        auto load = std::make_shared<snippets::isa::Load>(data, 1lu);
        load->set_friendly_name("load_ref");
        auto neg = std::make_shared<opset1::Negative>(load);
        auto store = std::make_shared<snippets::isa::Store>(neg, 1lu);
        store->set_friendly_name("store_ref");
        f_ref = std::make_shared<Function>(NodeVector{store}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;

    auto load_count = get_count(f, "load");
    auto load_count_ref = get_count(f_ref, "load_ref");
    ASSERT_EQ(load_count, load_count_ref);

    auto store_count = get_count(f, "store", false);
    auto store_count_ref = get_count(f_ref, "store_ref", false);
    ASSERT_EQ(store_count, store_count_ref);
}
