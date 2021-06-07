// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

#include <mutex>
#include <thread>
#include <vector>
#include "ngraph/ngraph.hpp"

using namespace ngraph;
using namespace std;

TEST(threading, get_friendly_name)
{
    const size_t number = 200;
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

    const auto get_friendly_name = [&](const std::shared_ptr<ngraph::Function>& f) {
        std::vector<std::string> names;
        for (const auto& op : f->get_ops())
        {
            names.emplace_back(op->get_friendly_name());
        }
    };

    std::vector<std::thread> threads(number);

    for (size_t i = 0; i < threads.size(); i++)
        threads[i] = std::thread(get_friendly_name, f);

    for (auto& th : threads)
    {
        th.join();
    }
}

