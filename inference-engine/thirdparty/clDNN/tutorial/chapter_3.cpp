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

/*! @page c3 Network and execution.
* @section intro Introduction
* In this chapter we will explain how to create and execute network.
*
* @include chapter_3.cpp
*
*/

using namespace cldnn;


void chapter_3(engine& engine, topology& topology)
{
    std::cout << std::endl << "-- Chapter 3 --" << std::endl;

    // Since we have topology and engine, we are ready to create network. Network is compiled graph/topology. During network creation
    // all kernels are compiled and memory is allocated.
    network network(engine, topology);
    // We are almost ready to go. Need to create and set input for network:
    memory input_mem = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 3, 1 } });
    set_values(input_mem, { -3.0f, -2.0f, 2.5f });
    // Set input
    network.set_input_data("input", input_mem);
    // Ready to go:
    auto outputs = network.execute();

    for (auto it : outputs)
    {
        // Print primtive info for all outputs.
        std::cout << network.get_primitive_info(it.first) << std::endl;
        // OUTPUT:
        // id: fc, type : fully connected
        //     input : softmax, count : 3, size : [b:1, f : 1, x : 3, y : 1]
        //     weights id : fc_weights, count : 9, bias id : fc_bias, count : 3
        //     with activation : false
        //     output padding lower size : [b:0, f : 0, x : 0, y : 0]
        //     output padding upper size : [b:0, f : 0, x : 0, y : 0]
        //     output : count : 3, size : [b:1, f : 1, x : 3, y : 1]

        auto mem_pointer = it.second.get_memory().pointer<float>();
        for (auto i : mem_pointer)
        {
            std::cout << i << " ";
        }
        std::cout << std::endl;

        // As you probably noticed network output is a result of the last one primitive "fc". By the last one we mean, the one that 
        // is not input to any other primitive.
    }

    // What if someone may want to look into intermediate results (hidden layers). This will be described in next chapter (4).
}