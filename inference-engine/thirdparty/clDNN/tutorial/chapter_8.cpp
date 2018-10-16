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
#include <../api/CPP/activation.hpp>
#include <../api/CPP/crop.hpp>
#include <../api/CPP/upsampling.hpp>
#include <iostream>
#include <chrono>

#include "helper_functions.h"

/*! @page c8 Extended profiling for networks built with optimized data.
* @section intro Introduction
* In this chapter we will learn how to properly get profiling data and primitive_ids for networks built with optimized data.
* Comparing to chapter_5, this time topology will be built two times - with and without optimize_data build option.
* Intermediate primitives will not be set as outputs (user will not have access to intermediate layers data).
* This way all applicable optimizations such as in-place optimizations (crop, concatenation) or primitives fusing for relu,
* will be applied to the network built with optimize_data set to true. The difference in executed primitives
* will be shown for both cases.
* 
* Please take a look in the files:
*   "convolution_kernel_tutorial.cpp"
*   "convolution_kernel_tutorial.h"
*   "convolution_tutorial.cl"
*
* @include chapter_8.cpp
*
*
*/

using namespace cldnn;

//Helper function for printing primitive ids and profiling info
void print_info(std::map<primitive_id, primitive_id>& all_primitives, std::map<primitive_id, event>& executed_primitives)
{
    std::cout << std::endl << "Org_primitive_id, Primitive_id_after_optimization" << std::endl;
    for (auto& p : all_primitives)
    {
        std::cout << p.first << ", " << p.second << std::endl;
    }

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

void chapter_8(engine& engine)
{
    std::cout << std::endl << "-- Chapter 8 --" << std::endl;

    // We are going to implement a network with activation and two crops that will be optimized on graph level:
    //                                             _ CROP_1(1x3x2x2,offset(0x0x0x0)) --> RELU
    //                                            |
    //  INPUT(1x4x1x1)--UPSAMPLING(1x4x2x2)----RELU  
    //                                            |_
    //                                               CROP_2(1x1x2x2,offset(0x3x0x0)) --> RELU
    // 

    // Create input memory for convolution layer
    memory input_prim = memory::allocate(engine, { data_types::f32, format::bfyx,{ tensor(spatial(1, 1), feature(4), batch(1)) } });

    set_values(input_prim, get_simple_data<float>(input_prim));

    // Create a topology and add primitives
    topology topology;
    topology.add(input_layout("input", input_prim.get_layout()));
    topology.add(upsampling("upsampling", "input", 2, 4, upsampling_sample_type::nearest));
    topology.add(activation("relu", "upsampling", activation_relu));
    topology.add(crop("crop1", "relu", tensor(batch(1), spatial(2, 2), feature(3)), { tensor(feature(0), spatial(0,0),batch(0)) }));
    topology.add(crop("crop2", "relu", tensor(batch(1), spatial(2, 2), feature(1)), { tensor(feature(3), spatial(0,0),batch(0)) }));
    topology.add(activation("relu1", "crop1", activation_relu));
    topology.add(activation("relu2", "crop2", activation_relu));

    // Build network without optimize data build option
    network network_1(engine, topology);

    // Set input.
    network_1.set_input_data("input", input_prim);
    // Ready to go.
    auto outputs_1 = network_1.execute();

    // Get primitives that were executed and their events needed for profiling
    // Please note that since optimize data is not set, then all primitives from created topology
    // that are not constant (such as data primitives) will be on this list
    auto executed_primitives_1 = network_1.get_executed_primitives();

    // Get all primitives names that are part of built network with their orginal names that were provided by user
    // Please note that since optimize data is not set, this list will match topology.get_primitives()
    // and all primitives names will be not changed, or optimized
    auto all_primitives_1 = network_1.get_all_primitives();

    //Print list of primitives with orginal ids, and profiling info
    std::cout << std::endl << "Primitives list and profiling info for network without optimize data build option." << std::endl;
    print_info(all_primitives_1, executed_primitives_1);

    // Now lets build and execute the same network but with optimize data build option
    build_options build_opt;
    build_opt.set_option(build_option::optimize_data(true));
    network network_2(engine, topology, build_opt);
    network_2.set_input_data("input", input_prim);
    auto outputs = network_2.execute();

    // Get primitives that were executed and their events needed for profiling
    // Please note that first relu and two crop primtives are not on the list, since they were optimized during graph optimization.
    // The list takes into account only primitives that were really executed.
    auto executed_primitives_2 = network_2.get_executed_primitives();

    // Get all primitives names that are part of built network with their orginal names that were provided by user
    // Please note that since optimize data is set, this list may no longer match topology.get_primitives(), and that is the case here.
    // Some of the primitives that were fused are removed from the list like the first activation primitive.
    // The primitives that were optimized (will not be executed) are now marked as "_optimized_" - please see crop primitives.
    // There can be also cases when primitive name will no longer match the primitive provided by the user, this will happen only
    // when primitive is set as output.
    auto all_primitives_2 = network_2.get_all_primitives();

    //Print list of primitives with orginal ids, and profiling info
    //Expected output from all_primitives_2 in this case will be:
    //Org_primitive_id, Primitive_id_after_optimization
    //    crop1, _optimized_
    //    crop2, _optimized_
    //    input, input
    //    relu1, relu1
    //    relu2, relu2
    //    upsampling, upsampling
    //
    //As mentioned before, "relu" is not on the list as upsampling will perform built-in relu. Crop primitives are marked as _optimized_.
    //Profiling data from executed_primitives_2 should contain 4 primitives - input, relu1, relu2 and upsampling
    std::cout << std::endl << "Primitives list and profiling info for network with optimize data build option." << std::endl;
    print_info(all_primitives_2, executed_primitives_2);

}
