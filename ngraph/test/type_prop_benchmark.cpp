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

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, DISABLED_benchmark_type_prop_add)
{
    auto p1 = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto p2 = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});

    constexpr size_t num_iterations = 1000000;
    size_t total_nanosec = 0;

    stopwatch sw;

    for (size_t i = 0; i < num_iterations; i++)
    {
        sw.start();
        auto n = make_shared<op::v1::Add>(p1, p2);
        sw.stop();

        total_nanosec += sw.get_nanoseconds();
    }

    std::cout.imbue(std::locale(""));
    std::cout << "Constructed " << std::fixed << num_iterations << " Add ops in " << std::fixed
              << total_nanosec << " ns" << std::endl;
}

TEST(type_prop, DISABLED_benchmark_type_prop_convolution)
{
    auto d = make_shared<op::Parameter>(element::f32, Shape{64, 3, 224, 224});
    auto f = make_shared<op::Parameter>(element::f32, Shape{64, 3, 7, 7});
    auto strides = Strides{1, 1};
    auto dilation = Strides{1, 1};
    auto padding_below = CoordinateDiff{1, 1};
    auto padding_above = CoordinateDiff{1, 1};

    constexpr size_t num_iterations = 1000000;
    size_t total_nanosec = 0;

    stopwatch sw;

    for (size_t i = 0; i < num_iterations; i++)
    {
        sw.start();
        auto n =
            make_shared<op::Convolution>(d, f, strides, dilation, padding_below, padding_above);
        sw.stop();

        total_nanosec += sw.get_nanoseconds();
    }

    std::cout.imbue(std::locale(""));
    std::cout << "Constructed " << std::fixed << num_iterations << " Convolution ops in "
              << std::fixed << total_nanosec << " ns" << std::endl;
}
