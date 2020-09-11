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

#include "helper_functions.h"

/*! @page c4 Hidden layers.
* @section intro Introduction
* In this chapter we show, how to get access to hidden layers using build options 
*
*
* @include chapter_4.cpp
*
*
*/

using namespace cldnn;


void chapter_4(engine& engine, topology& topology)
{

    std::cout << std::endl << "-- Chapter 4 --" << std::endl;

    // To get access to intermediate results of our network. To get special features we need to set custom building options:
    build_options build_opt;
    // Prepare vector of primitives that we want to have as an output:
    std::vector<cldnn::primitive_id> outputs_list(0);
    // Put every primitive from topology into this container:
    for (auto prim_id : topology.get_primitive_ids())
        outputs_list.push_back(prim_id);
    // Note: output from get_primitive_ids() can be used directly as a parameter in building option.
    // Set option.
    build_opt.set_option(build_option::outputs(outputs_list));
    // Add build options to network build.
    network network(engine, topology, build_opt);
    // We are almost ready to go. Need to create and set input for network:
    memory input_prim = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 3, 1 } });
    set_values(input_prim, { -3.0f, -2.0f, 2.5f });
    // Set input.
    network.set_input_data("input", input_prim);
    // Ready to go:
    auto outputs = network.execute();

    for (auto& it : outputs)
    {
        // Print id and output values.
        std::cout << it.first << std::endl;
        auto mem_pointer = it.second.get_memory().pointer<float>();
        for (auto i : mem_pointer)
        {
            std::cout << i << " ";
        }
        std::cout << std::endl;
    }
}
