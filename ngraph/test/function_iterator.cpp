//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <climits>
#include <random>

#include "gtest/gtest.h"

#include "inference_engine.hpp"
#include "ngraph/opsets/opset5.hpp"

using namespace std;
using namespace ngraph;

auto get_ordered_ops = [](Function* f) -> NodeVector {
    auto it = f->get_iterator();
    NodeVector nodes;
    while (it.get())
    {
        nodes.push_back(it.get());
        it.next();
    }
    return nodes;
};

TEST(function_iterator, test1)
{
    {
        auto param = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3});
        auto relu = std::make_shared<opset5::Relu>(param);
        auto mul = std::make_shared<opset5::Multiply>(param, relu);
        auto res = std::make_shared<opset5::Result>(mul);
        auto f = std::make_shared<Function>(ResultVector{res}, ParameterVector{param});

        ASSERT_EQ(get_ordered_ops(f.get()), NodeVector({param, relu, mul, res}));
    }

    {
        auto param1 = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3});
        auto param2 = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3});
        auto mul = std::make_shared<opset5::Multiply>(param1, param2);
        auto shape = opset5::Constant::create(element::i64, Shape{2}, {1, 3});
        auto reshape = std::make_shared<opset5::Reshape>(mul, shape, true);
        auto res = std::make_shared<opset5::Result>(reshape);
        auto f = std::make_shared<Function>(ResultVector{res}, ParameterVector{param1, param2});

        auto ordered_ops = get_ordered_ops(f.get());
        ASSERT_EQ(ordered_ops, NodeVector({param1, param2, mul, shape, reshape, res}));
    }

    {
        auto param1 = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3});
        auto param2 = std::make_shared<opset5::Parameter>(element::i64, Shape{2});
        auto mul = std::make_shared<opset5::Multiply>(param1, param1);
        auto relu = std::make_shared<opset5::Relu>(mul);
        auto reshape = std::make_shared<opset5::Reshape>(relu, param2, true);
        auto res = std::make_shared<opset5::Result>(reshape);
        auto f = std::make_shared<Function>(ResultVector{res}, ParameterVector{param1, param2});

        auto ordered_ops = get_ordered_ops(f.get());
        ASSERT_EQ(ordered_ops, NodeVector({param1, param2, mul, relu, reshape, res}));
    }
}

TEST(function_iterator, replace_node)
{
    {
        auto param1 = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3});
        auto mul = std::make_shared<opset5::Multiply>(param1, param1);
        auto relu = std::make_shared<opset5::Relu>(mul);
        auto res = std::make_shared<opset5::Result>(relu);
        auto f = std::make_shared<Function>(ResultVector{res}, ParameterVector{param1});

        auto it = f->get_iterator();
        it.next();

        {
            auto add = std::make_shared<opset5::Add>(param1, param1);
            replace_node(it.get(), add);
            it.next();
        }

        NodeVector ordered_ops;
        while (auto node = it.get())
        {
            ordered_ops.push_back(node);
            it.next();
        }
        ASSERT_EQ(ordered_ops, NodeVector({relu, res}));
    }
}

//TEST(function_iterator, get_ordered_ops)
//{
//    InferenceEngine::Core core;
//    auto net = core.ReadNetwork("/home/gkazanta/openvino/model-optimizer/person_detection_cpu.xml");
//    auto f = net.getFunction();
//
//    auto start = std::chrono::high_resolution_clock::now();
//    for (int i = 0; i < 1000; ++i)
//    {
//        int cnt = 0;
//        for (auto&& node : f->get_ordered_ops())
//        {
//            if (node.get())
//                ++cnt;
//        }
//        //        std::cout << cnt;
//    }
//    auto stop = std::chrono::high_resolution_clock::now();
//    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
//    std::cout << "Time: " << duration.count() << " ms" << std::endl;
//}
//
//TEST(function_iterator, iterator)
//{
//    InferenceEngine::Core core;
//    auto net = core.ReadNetwork("/home/gkazanta/openvino/model-optimizer/vgg16_tf_cpu.xml");
//    auto f = net.getFunction();
//
//    auto start = std::chrono::high_resolution_clock::now();
//    for (int i = 0; i < 1000; ++i)
//    {
//        auto it = f->get_iterator();
//        int cnt = 0;
//        while (it.get())
//        {
//            it.next();
//            ++cnt;
//        }
//        //        std::cout << "Max depth: " << it.max_depth << std::endl;
//        //        std::cout << "Count: " << cnt << std::endl;
//    }
//    auto stop = std::chrono::high_resolution_clock::now();
//    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
//    std::cout << "Time: " << duration.count() << " ms" << std::endl;
//}