// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <chrono>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "atomic_guard.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/opsets/opset8.hpp"
#include "ov_ops/type_relaxed.hpp"

using namespace ov;
using namespace std;

static std::shared_ptr<ov::Model> create_complex_function(size_t wide = 50) {
    const auto& split_subgraph = [](const ov::Output<ov::Node>& input) -> ov::OutputVector {
        auto relu = std::make_shared<ov::opset8::Relu>(input);
        auto type_relaxed =
            std::make_shared<ov::op::TypeRelaxed<ov::opset8::Asin>>(std::vector<element::Type>{element::f32},
                                                                    std::vector<element::Type>{element::f32},
                                                                    relu);
        auto axis_node = ov::opset8::Constant::create(ov::element::i64, Shape{}, {1});
        auto split = std::make_shared<ov::opset8::Split>(type_relaxed, axis_node, 2);
        return split->outputs();
    };
    const auto& concat_subgraph = [](const ov::OutputVector& inputs) -> ov::Output<ov::Node> {
        auto concat = std::make_shared<ov::opset8::Concat>(inputs, 1);
        auto type_relaxed =
            std::make_shared<ov::op::TypeRelaxed<ov::opset8::Asin>>(std::vector<element::Type>{element::f32},
                                                                    std::vector<element::Type>{element::f32},
                                                                    concat);
        auto relu = std::make_shared<ov::opset8::Relu>(concat);
        return relu->output(0);
    };

    auto parameter = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
    std::queue<ov::Output<ov::Node>> nodes;
    {
        auto outputs = split_subgraph(parameter->output(0));
        for (const auto& out : outputs) {
            nodes.push(out);
        }
    }

    while (nodes.size() < wide) {
        auto first = nodes.front();
        nodes.pop();
        auto outputs = split_subgraph(first);

        for (const auto& out : outputs) {
            nodes.push(out);
        }
    }

    while (nodes.size() > 1) {
        auto first = nodes.front();
        nodes.pop();
        auto second = nodes.front();
        nodes.pop();
        auto out = concat_subgraph(ov::OutputVector{first, second});

        nodes.push(out);
    }
    auto result = std::make_shared<ov::opset8::Result>(nodes.front());
    return std::make_shared<Model>(ov::ResultVector{result}, ov::ParameterVector{parameter});
}

TEST(threading, get_friendly_name) {
    const size_t number = 20;
    Shape shape{};
    auto a = make_shared<ov::opset8::Parameter>(element::i32, shape);
    auto iconst0 = ov::opset8::Constant::create(element::i32, Shape{}, {0});
    auto add_a1 = make_shared<ov::opset8::Add>(a, iconst0);
    auto add_a2 = make_shared<ov::opset8::Add>(add_a1, iconst0);
    auto add_a3 = make_shared<ov::opset8::Add>(add_a2, iconst0);
    auto abs_add_a3 = std::make_shared<ov::opset8::Abs>(add_a3);

    auto b = make_shared<ov::op::v0::Parameter>(element::i32, shape);
    auto add_b1 = make_shared<ov::opset8::Add>(b, iconst0);
    auto add_b2 = make_shared<ov::opset8::Add>(add_b1, iconst0);
    auto abs_add_b2 = std::make_shared<ov::opset8::Abs>(add_b2);

    auto graph = make_shared<ov::opset8::Multiply>(abs_add_a3, abs_add_b2);

    auto f = std::make_shared<Model>(ov::NodeVector{graph}, ParameterVector{a, b});

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

    const auto get_friendly_name = [&](const std::shared_ptr<ov::Model>& f) {
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

TEST(threading, clone_with_new_inputs) {
    auto function = create_complex_function(100);
    const auto cloneNodes = [&](const std::shared_ptr<const ov::Model>& f) {
        auto orderedOps = function->get_ordered_ops();
        std::vector<std::shared_ptr<ov::Node>> nodes;
        for (const auto& op : orderedOps) {
            ov::OutputVector inputsForShapeInfer;
            std::shared_ptr<ov::Node> opToShapeInfer;

            const auto inSize = op->get_input_size();
            for (size_t i = 0; i < inSize; i++) {
                if (ov::as_type<ov::opset8::Constant>(op->get_input_node_ptr(i))) {
                    inputsForShapeInfer.push_back(op->get_input_node_ptr(i)->clone_with_new_inputs(ov::OutputVector{}));
                } else {
                    inputsForShapeInfer.push_back(
                        std::make_shared<ov::opset8::Parameter>(op->get_input_element_type(i),
                                                                op->get_input_partial_shape(i)));
                }
            }

            opToShapeInfer = op->clone_with_new_inputs(inputsForShapeInfer);
            nodes.push_back(opToShapeInfer);
        }
    };

    const size_t numThreads = 6;
    std::vector<std::thread> threads(numThreads);

    for (auto&& thread : threads)
        thread = std::thread(cloneNodes, function);

    for (auto&& th : threads) {
        th.join();
    }
}
