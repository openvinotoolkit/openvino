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
#include <algorithm>
#include <future>
#include <iostream>
#include <type_traits>
#include <vector>

namespace ngraph
{
    namespace runtime
    {
        namespace detail
        {
            /**
             * @brief Returns the number of tasks to be created when running a kernel in parallel
             *
             * This function relies on an environment variable REF_TASKS_NUMBER which the
             * user can set to control the number of created tasks. If the variable is not set
             * or contains invalid number, a default number of tasks is returned.
             **/
            size_t parallel_tasks_number();

            /**
             * @brief Returns true if an anvironment variable indicating forced single threaded
             *        execution is set, false otherwise
             *
             * The env variable is REF_SINGLE_THREADED and can be set to any value
             * in order to force single-threaded execution of all reference implementations
             * that use the parallel executor.
             **/
            bool forced_single_threaded_execution();

            /**
             * @brief Returns the minimum number of elements for each a kernel execution should
             *        be attempted in parallel. If a reference implementation using the parallel
             *        executor is spawned with a tensor containing less elements, the executor
             *        should force single-threaded execution.
             **/
            uint64_t parallelism_threshold();
        } // namespace detail

        namespace parallel
        {
            /**
             * @brief Attempts to execute the unary op kernel in parallel
             **/
            template <typename T, typename Kernel>
            void execute(const T* arg0, T* out, const uint64_t elements, Kernel op)
            {
                if (detail::forced_single_threaded_execution() ||
                    elements < detail::parallelism_threshold())
                {
                    std::transform(arg0, arg0 + elements, out, op);
                }
                else
                {
                    const size_t num_tasks = detail::parallel_tasks_number();
                    std::vector<std::future<void>> tasks;
                    tasks.reserve(num_tasks);

                    const uint64_t chunk_length = elements / num_tasks;
                    for (size_t chunk = 0; chunk < num_tasks; ++chunk)
                    {
                        auto in0_chunk_begin = arg0 + chunk_length * chunk;
                        auto in0_chunk_end = arg0 + chunk_length * chunk + chunk_length;
                        auto out_chunk_begin = out + chunk_length * chunk;
                        tasks.push_back(
                            std::async([in0_chunk_begin, in0_chunk_end, out_chunk_begin, op] {
                                std::transform(in0_chunk_begin, in0_chunk_end, out_chunk_begin, op);
                            }));
                    }

                    for (auto&& task : tasks)
                    {
                        task.get();
                    }
                }
            }

            /**
             * @brief Attempts to execute the binary op kernel in parallel
             **/
            template <typename T, typename Kernel>
            void execute(const T* arg0, const T* arg1, T* out, const uint64_t elements, Kernel op)
            {
                if (detail::forced_single_threaded_execution() ||
                    elements < detail::parallelism_threshold())
                {
                    std::transform(arg0, arg0 + elements, arg1, out, op);
                }
                else
                {
                    const size_t num_tasks = detail::parallel_tasks_number();
                    std::vector<std::future<void>> tasks;
                    tasks.reserve(num_tasks);

                    const uint64_t chunk_length = elements / num_tasks;
                    for (size_t chunk = 0; chunk < num_tasks; ++chunk)
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
