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
#include <../api/CPP/custom_gpu_primitive.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>

#include "helper_functions.h"

/*! @page c7 Introduction to custom primitives.
* @section intro Introduction
* In this chapter we will create a custom primitive implemented with an OpenCL kernel and  
* execute it in a network.
* @include chapter_7.cpp
*
*
*/

using namespace cldnn;


void chapter_7(engine& my_engine)
{
    std::cout << std::endl << "-- Chapter 7 --" << std::endl;

    // We are going to implement a custom primitive that will execute a simple
    // addition kernel.
    // The Kernel will be implemented in OpenCL and will simply add 2 input buffers

    // Define a memory layout we'll use for input/output
    layout my_layout({ data_types::f32, format::bfyx,{ 1, 1, 3, 3 } });

    // Create input memory primitives
    memory input_prim1 = memory::allocate(my_engine, my_layout);
    memory input_prim2 = memory::allocate(my_engine, my_layout);

    set_values(input_prim1, get_simple_data<float>(input_prim1));
    set_values(input_prim2, get_simple_data<float>(input_prim2));

    // OpenCL kernel for the custom primitive
    std::string custom_primitive_kernel_code =
        R"__krnl(
            __kernel void add_kernel(const __global float* input0, const __global float* input1, __global float* output)
            {
                const unsigned idx = get_global_id(0);
                output[idx] = input0[idx] + input1[idx];
            }
        )__krnl";

    // The name of the OpenCL Entry point function
    std::string entry_point = "add_kernel";
    
    // Parameter binding for the custom primitive
    std::vector<cldnn_arg> parameters = { { arg_input, 0 }, { arg_input, 1 }, { arg_output, 0 } };
    
    // Output layout for the custom primitive
    layout output_layout = my_layout;

    // Compiler options to be handed to the OpenCL runtime
    std::string compilation_options = "-cl-mad-enable";

    // Now we can create a topology holding the inputs and add the custom primitive to it
    topology my_topology;
    my_topology.add(input_layout("input1", input_prim1.get_layout()));
    my_topology.add(input_layout("input2", input_prim2.get_layout()));
    my_topology.add(custom_gpu_primitive(
        "my_custom_primitive",
        { "input1", "input2" },
        { custom_primitive_kernel_code },
        entry_point,
        parameters,
        compilation_options,
        output_layout));

    // We can now build the network
    network my_network(my_engine, my_topology);

    // Set the inputs
    my_network.set_input_data("input1", input_prim1);
    my_network.set_input_data("input2", input_prim2);

    // And Execute
    auto outputs = my_network.execute();

    // Finally, we print out the input and output data
    std::cout << "input1:" << std::endl;
    for (const auto value : input_prim1.pointer<float>())
    {
        std::cout << std::setw(3) << value << ", ";
    }
    std::cout << std::endl;

    std::cout << "input2:" << std::endl;
    for (const auto value : input_prim2.pointer<float>())
    {
        std::cout << std::setw(3) << value << ", ";
    }
    std::cout << std::endl;

    std::cout << "output:" << std::endl;
    for (const auto value : outputs.at("my_custom_primitive").get_memory().pointer<float>())
    {
        std::cout << std::setw(3) << value << ", ";
    }
    std::cout << std::endl;
}
