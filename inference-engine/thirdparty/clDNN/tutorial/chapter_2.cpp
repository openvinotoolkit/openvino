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
#include <../api/CPP/activation.hpp>
#include <../api/CPP/softmax.hpp>
#include <../api/CPP/memory.hpp>
#include <../api/CPP/fully_connected.hpp>
#include <../api/CPP/data.hpp>
#include <../api/CPP/topology.hpp>
#include <iostream>

#include "helper_functions.h"

/*! @page c2 Primitives and topology
* @section intro Introduction
* In this chapter we will explain how to create primitives, show some kinds of primitives and explain how to build topology.
*
* @include chapter_2.cpp
*
*/

using namespace cldnn;

topology chapter_2(engine& engine)
{
    std::cout << std::endl << "-- Chapter 2 --" << std::endl;

    // The most trivial primitive is activation, lets start with that:
    activation relu(
        "relu",  // primitive identifier
        "input", // identifier of input ( output of primitive with provided name is input to current )
        activation_relu);

    // Softmax is also very easy to create:
    softmax softmax(
        "softmax", // primitive identifier
        "relu"); // relu will be input to softmax

    // Fully connected is little bit more complex. 
    // Need to create weights and biases, that are primitives with 'data' type (chapter 1).
    // We will have fc layer with 3 inputs and 3 outputs. Weights have to be 3x3:
    auto weights_mem = memory::allocate(engine, { data_types::f32,format::bfyx,{
        3, // b - stands for size of the input 
        1, // ignored in fc
        3, // x - stands for size of output ( number of neurons in fully connected layer )
        1 } }); // y ignored
    // Use function to fill data:
    set_values(weights_mem, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f });
    // Create data primitive.
    data fc_weights("fc_weights", weights_mem);

    // Biases are optional but we can use those in this example. Create 'data' in the same way:
    auto bias_mem = memory::allocate(engine, { data_types::f32,format::bfyx,{ spatial(3) } }); // y, b and f will be set to ones by default
    // Use function to fill data:
    set_values(bias_mem, { 0.0f, 1.0f, 0.5f });
    // Create data primitive.
    data fc_bias("fc_bias", bias_mem);

    // Now we are ready to create fc primitive.
    fully_connected fc(
        "fc",        // primitive identifier
        "softmax",   // softmax will be input to fully connected
        "fc_weights",// weigths identifier
        "fc_bias"    // bias identifier
    );

    // Now we have 3 primitives created. Relation is defined by input->output setting. The only thing that we miss to create topology
    // is input declaration. To declare input we need input_layout(chapter 1):
    input_layout in_layout("input", layout(data_types::f32, format::bfyx, tensor(spatial(3))));
    // Now, we are ready to put those into topology.
    // Don't forget to put all data primitives inside.
    topology topology(
        in_layout,
        softmax,
        fc,
        fc_bias,
        fc_weights
    );
    // If you want to add another primitive to existing topology, you can use add method. 
    topology.add(relu);
    // Take a look what is inside:
    std::cout << "Topology contains:" << std::endl;
    for (auto it : topology.get_primitive_ids())
    {
        std::cout << it << std::endl;
    }
    return topology;
}