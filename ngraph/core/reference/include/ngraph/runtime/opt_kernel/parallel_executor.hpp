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

#pragma once
#include <future>
#include <vector>
#include <algorithm>
#include <iostream>

namespace ngraph
{
    namespace runtime
    {
        namespace parallel
        {
            // template <typename Iterator>
            // struct ContainerView
            // {
            //     Iterator first;
            //     Iterator last;
            // };

            // template <typename InputIterator, typename OutputIterator, typename Function>
            // void execute(ContainerView<InputIterator> input1,
            //              InputIterator input2,
            //              OutputIterator output,
            //              Function f)
            // {
            //     std::transform(input1.first, input1.last, input2, output, f);
            // }

            bool forced_single_threaded_execution();

            template <typename T, typename Operation>
            void
                execute(const T* arg0, const T* arg1, T* out, const uint64_t elements, Operation op)
            {
                if (forced_single_threaded_execution())
                {
                    std::cout << "Single threaded execution\n";
                    // TODO: execute in a single thread
                    std::transform(arg0, arg0 + elements, arg1, out, op);
                }
                else
                {
                    std::cout << "Multi threaded execution\n";
                    const size_t TASKS_COUNT = 8;
                    std::vector<std::future<void>> tasks;
                    tasks.reserve(TASKS_COUNT);

                    const uint64_t chunk_length = elements / TASKS_COUNT;
                    for (size_t chunk = 0; chunk < TASKS_COUNT; ++chunk)
                    {
                        auto in0_chunk_begin = arg0 + chunk_length * chunk;
                        auto in0_chunk_end = arg0 + chunk_length * chunk + chunk_length;
                        auto in1_chunk_begin = arg1 + chunk_length * chunk;
                        auto out_chunk_begin = out + chunk_length * chunk;
                        tasks.push_back(std::async(
                            [in0_chunk_begin, in0_chunk_end, in1_chunk_begin, out_chunk_begin, op] {
                                std::transform(in0_chunk_begin,
                                               in0_chunk_end,
                                               in1_chunk_begin,
                                               out_chunk_begin,
                                               op);
                            }));
                    }

                    for (auto&& task : tasks)
                    {
                        task.get();
                    }
                }
            }
        } // namespace parallel
    }     // namespace runtime
} // namespace ngraph
