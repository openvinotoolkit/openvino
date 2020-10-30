/*
// Copyright (c) 2017 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include <../api/engine.hpp>
#include <../api/input_layout.hpp>
#include <../api/memory.hpp>
#include <../api/data.hpp>
#include <../api/topology.hpp>
#include <../api/network.hpp>
#include <../api/convolution.hpp>
#include <iostream>
#include <chrono>

#include "helper_functions.h"

/*! @page c6 How to add my own kernel implementation.
* @section intro Introduction
* In this chapter we will learn how to add a new Convolution kernel into clDNN kernel selector.
* 
* Please take a look in the files:
*   "convolution_kernel_tutorial.cpp"
*   "convolution_kernel_tutorial.h"
*   "convolution_tutorial.cl"
*
* @include chapter_6.cpp
*
*
*/

using namespace cldnn;

void chapter_6(engine& engine)
{
    std::cout << std::endl << "-- Chapter 6 --" << std::endl;

    // We are going to implement a simple network with Convolution layer:
    //      input:          227x227 with 3 feature maps
    //      filter size:    3x3
    //      stride:         1,1
    //      offset:         0,0
    //
    // We use this code as an helper to test our new convolution kernel

    // Create input memory for convolution layer
    memory input_prim = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 3, 227, 227 } });
    memory weights    = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 3, 3, 3 } });
    memory biases     = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });

    set_values(input_prim, get_simple_data<float>(input_prim));
    set_values(weights,    get_simple_data<float>(weights));
    set_values(biases,     get_simple_data<float>(biases));

    // Create a topology with a simple Convolution layer
    topology topology(
        input_layout("conv_input", input_prim.get_layout()),
        data("conv_weights", weights),
        data("conv_biases", biases),
        convolution(
            "conv",
            "conv_input",
            { "conv_weights" },
            { "conv_biases" })
    );

    build_options build_opt;
    // Optimize_data flag can change weights and outputs layouts. Let take a look at 
    // Set option to optimize data.
    build_opt.set_option(build_option::optimize_data(true));

    network network(engine, topology, build_opt);

    // Set input.
    network.set_input_data("conv_input", input_prim);
    // Ready to go.
    auto outputs = network.execute();

    // Get primitives that were executed and their events needed for profiling
    auto executed_primitives = network.get_executed_primitives();

    // Now, we want to check what is the time of execution of each primitive:
    std::vector<cldnn::instrumentation::profiling_info> profiling_table;
    for (auto& p : executed_primitives)
    {
        profiling_table.push_back({ p.first, p.second.get_profiling_info() });
    }

    // We have table of profiling metrics.
    for (auto& p : profiling_table)
    {
        std::cout << p.name << ":" << std::endl;
        for (auto& q : p.intervals)
        {
            std::cout << "\t" << q.name << ": " << std::chrono::duration_cast<std::chrono::duration<double, std::chrono::milliseconds::period>>(q.value->value()).count()
                << " milliseconds" << std::endl;
        }
    }
}
