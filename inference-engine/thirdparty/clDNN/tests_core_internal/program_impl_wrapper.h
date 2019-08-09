/*
// Copyright (c) 2019 Intel Corporation
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

namespace cldnn
{
    struct program_node;
    struct program_impl;
    // This class is intended to allow using private methods from program_impl within tests_core_internal project.
    // Once needed, more methods wrapper should be added here.
    class program_impl_wrapper
    {
    public:
        static void add_connection(program_impl& p, program_node& prev, program_node& next)
        {
            p.add_connection(prev, next);
        }
        template <class Pass, typename... Args>
        static void apply_opt_pass(program_impl& p, Args&&... args)
        {
            p.apply_opt_pass<Pass>(std::forward<Args>(args)...);
        }
        static void run_graph_compilation(program_impl& p)
        {
            p.run_graph_compilation();
        }
        static void prepare_memory_dependencies(program_impl& p)
        {
            p.prepare_memory_dependencies();
        }
    };

}
