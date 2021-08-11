// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_blob.h>
#include <gtest/gtest.h>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/ngraph.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <chrono>

#include <inference_engine.hpp>
#include <transformations/common_optimizations/common_optimizations.hpp>
#include <transformations/op_conversions/convert_subtract.hpp>

using namespace ngraph;

bool is_topological_order(const NodeVector & nodes) {
    std::unordered_set<Node *> visited;
    for (const auto & node : nodes) {
        for (auto in : node->input_values()) {
            if (!visited.count(in.get_node())) {
                std::cout << "For Node: " << node << " input: " << *in.get_node() << " is not visited\n";
                return false;
            }
        }
        visited.insert(node.get());
    }
    std::cout << "[ INFO ] Order is ok!\n";
    return true;
}

TEST(check, bert) {
    auto core = InferenceEngine::Core();
    //auto net = core.ReadNetwork("/tmp/googlenet-v4.xml");
    //auto net = core.ReadNetwork("/tmp/text-to-speech-en-multi-0001-regression.xml");
    //auto net = core.ReadNetwork("/Users/gleb_dmitrievich/Work/repos/openvino/yolo-v2-ava-0001.xml");
    auto net = core.ReadNetwork("/Users/gleb_dmitrievich/Work/repos/openvino/bert-small-uncased-whole-word-masking-squad-0002.xml");
    auto f = net.getFunction();

    const int iter = 100;

    std::cout << "get_ordered_ops()\n";
    {
        int64_t count_ops;
        std::vector<double> t1(iter);
        for (auto &d : t1) {
            auto before = std::chrono::high_resolution_clock::now();
            const auto & ops = f->get_ordered_ops();
            auto after = std::chrono::high_resolution_clock::now();
            d = std::chrono::duration_cast<std::chrono::nanoseconds>(after - before).count();
            count_ops = ops.size();
        }
        auto diff = std::accumulate(t1.begin(), t1.end(), 0.0) / t1.size();
        printf("Elapsed time is %lf nanoseconds.\n", diff);
        std::cout << "Nodes: " << count_ops << std::endl;
    }

    std::cout << "traverse manually\n";
    {
        auto ops = f->get_ordered_ops();
        std::vector<double> t2(iter);
        size_t count = 0;
        for (auto &d : t2) {
            auto before = std::chrono::high_resolution_clock::now();
            for (auto node : ops) {
                size_t input_size = node->get_input_size();
                for (size_t i = 0; i < input_size; ++i) {
                    // auto s = node->get_input_partial_shape(i);
                }
            }
            count += ops.size();
            auto after = std::chrono::high_resolution_clock::now();
            d = std::chrono::duration_cast<std::chrono::nanoseconds>(after - before).count();
        }
        auto diff = std::accumulate(t2.begin(), t2.end(), 0.0) / t2.size();
        printf("Elapsed time is %lf nanoseconds.\n", diff);
        std::cout << "Nodes: " << count << std::endl;
    }

    std::cout << "optimized: iterator\n";
    {
        auto order = (*f->get_parameters().begin())->m_order;

        std::vector<double> t3(iter);
        size_t count_ops = 0;
        for (auto &d : t3) {
            NodeVector topological_order;
            auto before = std::chrono::high_resolution_clock::now();
            auto el = order->begin();
            while (el) {
                auto node = el->node;
                // topological_order.push_back(node->shared_from_this());
                size_t input_size = node->get_input_size();
                for (size_t i = 0; i < input_size; ++i) {
                    // auto s = node->get_input_partial_shape(i);
                }
                // std::cout << el->node->get_type_name() << " : " << el->node->get_friendly_name() << std::endl;
                el = el->output;
                ++count_ops;
            }
            auto after = std::chrono::high_resolution_clock::now();
            d = std::chrono::duration_cast<std::chrono::nanoseconds>(after - before).count();
            // ASSERT_TRUE(is_topological_order(topological_order));
        }
        auto diff = std::accumulate(t3.begin(), t3.end(), 0.0) / t3.size();
        printf("Elapsed time is %lf nanoseconds.\n", diff);
        std::cout << "Nodes: " << count_ops << std::endl;
    }

    std::cout << "optimized: get_cached_ordered_ops\n";
    {
        std::vector<double> t3(iter);
        size_t count_ops = 0;
        for (auto &d : t3) {
            auto before = std::chrono::high_resolution_clock::now();
            const auto & topological_order = f->get_cached_ordered_ops();
            auto after = std::chrono::high_resolution_clock::now();
            d = std::chrono::duration_cast<std::chrono::nanoseconds>(after - before).count();
            // ASSERT_TRUE(is_topological_order(topological_order));
            count_ops = topological_order.size();
        }
        auto diff = std::accumulate(t3.begin(), t3.end(), 0.0) / t3.size();
        printf("Elapsed time is %lf nanoseconds.\n", diff);
        std::cout << "Nodes: " << count_ops << std::endl;
    }

    ngraph::pass::Manager m;
    m.register_pass<ngraph::pass::CommonOptimizations>();

    std::cout << "apply common optimizations\n";
    {
        std::vector<double> t1(iter);
        for (auto &d : t1) {
            auto before = std::chrono::high_resolution_clock::now();
            m.run_passes(f);
            auto after = std::chrono::high_resolution_clock::now();
            d = std::chrono::duration_cast<std::chrono::nanoseconds>(after - before).count();
        }
        auto diff = std::accumulate(t1.begin(), t1.end(), 0.0) / t1.size();
        printf("Elapsed time is %lf nanoseconds.\n", diff);
    }

    std::cout << "again apply common optimizations\n";
    {
        std::vector<double> t1(iter);
        for (auto &d : t1) {
            auto before = std::chrono::high_resolution_clock::now();
            m.run_passes(f);
            auto after = std::chrono::high_resolution_clock::now();
            d = std::chrono::duration_cast<std::chrono::nanoseconds>(after - before).count();
        }
        auto diff = std::accumulate(t1.begin(), t1.end(), 0.0) / t1.size();
        printf("Elapsed time is %lf nanoseconds.\n", diff);
    }
}

TEST(check, simple_graph) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{2, 2});
        auto min_const = opset5::Constant::create(element::f32, Shape{1}, {0.1});
        auto max_const = opset5::Constant::create(element::f32, Shape{1}, {5});
        auto max = std::make_shared<opset5::Maximum>(data, min_const);
        auto min = std::make_shared<opset5::Minimum>(max, max_const);
        f = std::make_shared<Function>(NodeVector{min}, ParameterVector{data});
    }

    auto order = (*f->get_parameters().begin())->m_order;
    auto el = order->begin();
    NodeVector topological_order;
    while (el)
    {
        topological_order.push_back(el->node->shared_from_this());
        std::cout << el->node->get_type_name() << std::endl;
        el = el->output;
    }
    ASSERT_TRUE(is_topological_order(topological_order));
}

TEST(check, multiple_parameters_graph) {
    std::shared_ptr<Function> f(nullptr);
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{2, 2});
        auto min_const = opset5::Constant::create(element::f32, Shape{1}, {0.1});
        auto max_const = opset5::Constant::create(element::f32, Shape{1}, {5});
        auto max = std::make_shared<opset5::Maximum>(data, min_const);
        auto min = std::make_shared<opset5::Minimum>(max, max_const);

        auto data2 = std::make_shared<opset5::Parameter>(element::f32, Shape{2, 2});
        auto relu = std::make_shared<opset5::Relu>(data2);

        auto mul = std::make_shared<opset5::Multiply>(min, relu);
        auto concat = std::make_shared<opset5::Concat>(OutputVector{mul, data2}, 0);

        f = std::make_shared<Function>(NodeVector{concat}, ParameterVector{data, data2});
    }

    auto order = (*f->get_parameters().begin())->m_order;
    auto el = order->begin();
    NodeVector topological_order;
    while (el)
    {
        topological_order.push_back(el->node->shared_from_this());
        std::cout << el->node->get_type_name() << std::endl;
        el = el->output;
    }
    ASSERT_TRUE(is_topological_order(topological_order));
}

TEST(check, node_elimination)
{
    std::shared_ptr<Function> f;
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{2, 2});
        auto relu = std::make_shared<opset5::Relu>(data);
        // useless node
        auto tmp = std::make_shared<opset5::Relu>(data);
        f = std::make_shared<Function>(NodeVector{relu}, ParameterVector{data});
    }

    auto order = (*f->get_parameters().begin())->m_order;
    auto el = order->begin();
    NodeVector topological_order;
    while (el)
    {
        topological_order.push_back(el->node->shared_from_this());
        std::cout << el->node->get_type_name() << std::endl;
        el = el->output;
    }
    ASSERT_EQ(topological_order.size(), 3);
    ASSERT_TRUE(is_topological_order(topological_order));
}

TEST(check, apply_transformation)
{
    std::shared_ptr<Function> f;
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{2, 2});
        auto value = opset5::Constant::create(element::f32, Shape{}, {1});
        auto sub = std::make_shared<opset5::Subtract>(data, value);
        f = std::make_shared<Function>(NodeVector{sub}, ParameterVector{data});
    }

    ngraph::pass::Manager m;
    m.register_pass<ngraph::pass::ConvertSubtract>();
    m.run_passes(f);

    auto order = (*f->get_parameters().begin())->m_order;
    auto el = order->begin();
    NodeVector topological_order;
    while (el)
    {
        topological_order.push_back(el->node->shared_from_this());
        std::cout << el->node->get_type_name() << std::endl;
        el = el->output;
    }
    ASSERT_EQ(topological_order.size(), 6);
    ASSERT_TRUE(is_topological_order(topological_order));
}

TEST(check, constant_folding)
{
    std::shared_ptr<Function> f;
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{2, 2});
        auto shapeof = std::make_shared<opset5::ShapeOf>(data);
        auto mul = std::make_shared<opset5::Multiply>(shapeof, opset5::Constant::create(element::i64, Shape{}, {1}));
        auto reshape = std::make_shared<opset5::Reshape>(data, mul, true);
        f = std::make_shared<Function>(NodeVector{reshape}, ParameterVector{data});
    }

    ngraph::pass::Manager m;
    m.register_pass<ngraph::pass::ConstantFolding>();
    m.run_passes(f);

    auto order = (*f->get_parameters().begin())->m_order;
    auto el = order->begin();
    NodeVector topological_order;
    while (el)
    {
        topological_order.push_back(el->node->shared_from_this());
        std::cout << el->node->get_type_name() << std::endl;
        el = el->output;
    }
    ASSERT_EQ(topological_order.size(), 4);
    ASSERT_TRUE(is_topological_order(topological_order));
}