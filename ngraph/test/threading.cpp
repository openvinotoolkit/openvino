// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <chrono>
#include <mutex>
#include <thread>
#include <vector>

#include "atomic_guard.hpp"
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"

using namespace ngraph;
using namespace std;

TEST(threading, get_friendly_name) {
    const size_t number = 20;
    Shape shape{};
    auto a = make_shared<op::Parameter>(element::i32, shape);
    auto iconst0 = op::Constant::create(element::i32, Shape{}, {0});
    auto add_a1 = make_shared<op::v1::Add>(a, iconst0);
    auto add_a2 = make_shared<op::v1::Add>(add_a1, iconst0);
    auto add_a3 = make_shared<op::v1::Add>(add_a2, iconst0);
    auto abs_add_a3 = std::make_shared<op::Abs>(add_a3);

    auto b = make_shared<op::Parameter>(element::i32, shape);
    auto add_b1 = make_shared<op::v1::Add>(b, iconst0);
    auto add_b2 = make_shared<op::v1::Add>(add_b1, iconst0);
    auto abs_add_b2 = std::make_shared<op::Abs>(add_b2);

    auto graph = make_shared<op::v1::Multiply>(abs_add_a3, abs_add_b2);

    auto f = std::make_shared<Function>(ngraph::NodeVector{graph}, ParameterVector{a, b});

    const auto compare_names = [](const std::vector<std::string>& names) {
        static std::unordered_set<std::string> ref_names;
        static std::once_flag flag;
        std::call_once(flag, [&]() {
            for (const auto& name : names)
                ref_names.insert(name);
        });
        for (const auto& name : names) {
            ASSERT_TRUE(ref_names.count(name));
        }
    };

    const auto get_friendly_name = [&](const std::shared_ptr<ngraph::Function>& f) {
        std::vector<std::string> names;
        for (const auto& op : f->get_ops()) {
            names.emplace_back(op->get_friendly_name());
        }
        compare_names(names);
    };

    std::vector<std::thread> threads(number);

    for (auto&& thread : threads)
        thread = std::thread(get_friendly_name, f);

    for (auto&& th : threads) {
        th.join();
    }
}

TEST(threading, check_atomic_guard) {
    std::atomic_bool test_val{false};
    int result = 2;
    const auto& thread1_fun = [&]() {
        ov::AtomicGuard lock(test_val);
        std::chrono::milliseconds ms{2000};
        std::this_thread::sleep_for(ms);
        result += 3;
    };
    const auto& thread2_fun = [&]() {
        std::chrono::milliseconds ms{500};
        std::this_thread::sleep_for(ms);
        ov::AtomicGuard lock(test_val);
        result *= 3;
    };
    std::vector<std::thread> threads(2);
    threads[0] = std::thread(thread1_fun);
    threads[1] = std::thread(thread2_fun);

    for (auto&& th : threads) {
        th.join();
    }
    ASSERT_EQ(result, 15);
}
