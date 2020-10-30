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
#include <../api/memory.hpp>
#include <../api/tensor.hpp>
#include <../api/input_layout.hpp>
#include <../api/data.hpp>
#include <iostream>
/*! @page c1 Engine, layout, tensor, memory, data and input
* @section intro Introduction
* In this chapter we will explain how to create engine, define and allocate memory. What is and how to use: tensor, layout, input_layout and data.
*
* @include chapter_1.cpp
* 
*
*/

using namespace cldnn;

engine chapter_1()
{

    std::cout << std::endl << "-- Chapter 1 --" << std::endl;
    // To create memory we have to create engine first. Engine is responsible for memory and kernel handling (creation, compilation, allocation).
    // Currently OCL backend implementation only is available.

    // Add profiling information
    const bool profiling = true;

    // Create an engine
    engine engine(profiling);
    // We have to choose data type (f32 or f16):
    data_types data_type = data_types::f32;
    // Format (order of dimensions in memory), bfyx is the most optimal and common:
    format::type format = format::byxf;

    // Before memory allocation we have to create tensor that describes memory size. We can do it in serveral ways:
    tensor tensor1(
        4, // batches
        1, // features
        32, // width (spatial x)
        32); // height (spatial y)

    tensor tensor2(spatial(32, 32), batch(4), feature(1));
    tensor tensor3(spatial(32, 32), batch(4)); // default value for non-initialized dimension is 1

    std::cout << "Is tensor1 == tensor2 == tensor3?:" <<
        (((tensor1 == tensor2) && (tensor2 == tensor3)) ? "yes" : "no") << std::endl;
    std::cout << "print tensor:" << tensor1 << std::endl;

    // Now we are ready to create layout:
    layout layout1(data_type, format, tensor1);
    // which can be used to allocate memory for given engine:
    memory memory1 = memory::allocate(engine, layout1);

    // Special type of layout is input layout. It is named layout. Name is a string with identifier of layout.
    input_layout in_layout("input", layout1);

    // You can also give name to memory to create a data.
    data data("named_memory", memory1);

    return engine;
}
