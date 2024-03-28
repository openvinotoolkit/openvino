// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass_manager.h"
#include "program_dump_graph.h"
#include "intel_gpu/graph/program.hpp"

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

using namespace cldnn;

pass_manager::pass_manager(program& p) {
    pass_count = 0;
    auto path = get_dir_path(p.get_config());
    if (!path.empty()) {
        graph_opt_log.open(path + std::to_string(p.get_prog_id()) + "_cldnn_graph_optimizer.log");
        if (graph_opt_log.is_open()) {
            graph_opt_log.setf(std::ios::fixed, std::ios::floatfield);
            graph_opt_log << std::setprecision(4);
            // print graph_opt_log header
            graph_opt_log << "program number: " << p.get_prog_id() << "\n"
                << "Pass\t"
                << "Proc.\t"
                << "primitives\t"
                << "Pass\t\t"
                << "Pass\n"

                << "ID  \t"
                << "order\t\t"
                << "optimized\t"
                << "time,\t\t"
                << "name\n"

                << "   \t"
                << "size\t"
                << "out\t\t\t"
                << "millisec\t"
                << "   \n";
        }
    }
}

void pass_manager::run(program& p, base_pass& pass) {
    using ms = std::chrono::duration<double, std::ratio<1, 1000>>;
    using Time = std::chrono::high_resolution_clock;

    GPU_DEBUG_LOG << "Run pass " << pass.get_name() << " (program_id=" << p.get_id() << ")" << std::endl;
    GPU_DEBUG_DEFINE_MEM_LOGGER(pass.get_name());
    auto start = Time::now();
    pass.run(p);
    auto stop = Time::now();
    std::chrono::duration<float> fs = stop - start;
    ms opt_pass_time = std::chrono::duration_cast<ms>(fs);
    GPU_DEBUG_LOG << "Pass " << pass.get_name() << " execution time: " << opt_pass_time.count() << " ms" << std::endl;

    p.save_pass_info(pass.get_name());

    if (graph_opt_log.is_open()) {
        graph_opt_log << std::setw(4) << get_pass_count() << "\t"
            << std::setw(5) << p.get_processing_order().size() << "\t"
            << std::setw(4) << p.get_optimized_out().size() << "\t\t"
            << std::setw(8) << opt_pass_time.count() << "\t"
            << pass.get_name() << "\n";
    }

    std::string dump_file_name;
    if (pass_count < 10)
        dump_file_name += "0";
    dump_file_name += std::to_string(pass_count) + "_" + pass.get_name();
    p.dump_program(dump_file_name.c_str(), true);
    pass_count++;
}
