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

#include <../api/CPP/cldnn_defs.h>
#include <../api/CPP/engine.hpp>
#include <../api/CPP/input_layout.hpp>
#include <../api/CPP/memory.hpp>
#include <../api/CPP/data.hpp>
#include <../api/CPP/topology.hpp>
#include <../api/CPP/network.hpp>
#include <iostream>
#include <chrono>

#include "helper_functions.h"

/*! @page c5 Performance building option.
* @section intro Introduction
* In this chapter we will present network build option that improves performance. Note this option 
* can change memory layouts. This chapter also shows how to get primitives profiling info.
* @include chapter_5.cpp
*
*
*/

using namespace cldnn;


void chapter_5(engine& engine, topology& topology)
{
    std::cout << std::endl << "-- Chapter 5 --" << std::endl;

    build_options build_opt;
    // Optimize_data flag can change weights and outputs layouts. Let take a look at 
    // final result and fc weights.
    build_opt.set_option(build_option::outputs(topology.get_primitive_ids()));
    // Set option to optimize data.
    build_opt.set_option(build_option::optimize_data(true));
    network network(engine, topology, build_opt);
    memory input_prim = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 3, 1 } });
    set_values(input_prim, { -3.0f, -2.0f, 2.5f });
    // Set input.
    network.set_input_data("input", input_prim);
    // Ready to go.
    auto outputs = network.execute();

    for (auto& it : outputs)
    {
        // Print id and output values.
        std::cout << "optimized " << it.first << std::endl;
        auto mem_pointer = it.second.get_memory().pointer<float>();
        for (auto i : mem_pointer)
        {
            std::cout << i << " ";
        }
        std::cout << std::endl;
    }

    // Now, we want to check what is the time of execution of each primitive:
    std::vector<cldnn::instrumentation::profiling_info> profiling_table;
    for (auto& p : outputs)
    {
        profiling_table.push_back({ p.first, p.second.get_event().get_profiling_info() });
    }

    // We have table of profiling metrics.
    for (auto& p : profiling_table)
    {
        std::cout << p.name << ":" << std::endl;
        for (auto& q : p.intervals)
        {
            std::cout << "\t" << q.name << ": " << std::chrono::duration_cast<std::chrono::duration<double, std::chrono::nanoseconds::period>>(q.value->value()).count()
                << " nanoseconds" << std::endl;
        }
    }
}
