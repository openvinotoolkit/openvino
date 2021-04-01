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

/*! @page tutorial clDNN Tutorial  
* @section intro Introduction
*  This section contains 8 chapters of tutorial demonstrating how to work with clDNN. If you are new in clDNN, we recommend to start with
*  <a href="https://01org.github.io/clDNN/index.html">"clDNN documentation"</a> that describes API. We assume that user is familiar with C++ or C and Deep Learining terminology.
*  
* @subpage c1 <br>
* @subpage c2 <br>
* @subpage c3 <br>
* @subpage c4 <br>
* @subpage c5 <br>
* @subpage c6 <br>
* @subpage c7 <br>
* @subpage c8 <br>
*
*/
#include <../api/engine.hpp>
#include <../api/topology.hpp>


cldnn::engine   chapter_1();                                    // Engine, layout, tensor, memory, data and input
cldnn::topology chapter_2(cldnn::engine&);                      // Primitives and topology
void            chapter_3(cldnn::engine&, cldnn::topology&);    // Network and execution
void            chapter_4(cldnn::engine&, cldnn::topology&);    // Hidden layers access
void            chapter_5(cldnn::engine&, cldnn::topology&);    // Other building options
void            chapter_6(cldnn::engine&);                      // How to add a kernel to clDNN
void            chapter_7(cldnn::engine&);                      // How to create a custom primitive (without changing clDNN)
void            chapter_8(cldnn::engine&);                      // Extended profiling for networks built with optimized data

int main()
{
    try {
        cldnn::engine eng = chapter_1();
        cldnn::topology topology = chapter_2(eng);
        chapter_3(eng, topology);
        chapter_4(eng, topology);
        chapter_5(eng, topology);
        chapter_6(eng);
        chapter_7(eng);
        chapter_8(eng);
    }
    catch (const std::exception& ex)
    {
        std::cout << ex.what();
    }
    return 0;
}
