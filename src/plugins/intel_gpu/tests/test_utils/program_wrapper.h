// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace cldnn
{
    struct program_node;
    struct program;
    // This class is intended to allow using private methods from program within tests_core_internal project.
    // Once needed, more methods wrapper should be added here.
    class program_wrapper
    {
    public:
        static void add_connection(program& p, program_node& prev, program_node& next)
        {
            p.add_connection(prev, next);
        }
        template <class Pass, typename... Args>
        static void apply_opt_pass(program& p, Args&&... args)
        {
            p.apply_opt_pass<Pass>(std::forward<Args>(args)...);
        }
        static void run_graph_compilation(program& p)
        {
            p.run_graph_compilation();
        }
        static void compile(program& p)
        {
            p.compile();
        }
        static void build(program& p)
        {
            program_wrapper::run_graph_compilation(p);
            program_wrapper::compile(p);
            program_wrapper::init_kernels(p);
        }
        static void init_kernels(program& p)
        {
            p.init_kernels();
        }
        static void prepare_memory_dependencies(program& p)
        {
            p.prepare_memory_dependencies();
        }
    };

}
