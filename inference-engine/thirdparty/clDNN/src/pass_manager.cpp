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

#include "pass_manager.h"
#include "program_dump_graph.h"
#include "program_impl.h"

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

pass_manager::pass_manager(program_impl& p) {
    pass_count = 0;
    auto path = get_dir_path(p.get_options());
    if (!path.empty()) {
        graph_opt_log.open(path + std::to_string(p.get_prog_id()) + "_cldnn_graph_optimizer.log");
        if (graph_opt_log.is_open()) {
            graph_opt_log.setf(std::ios::fixed, std::ios::floatfield);
            graph_opt_log << std::setprecision(2);
            // print graph_opt_log header
            graph_opt_log << "program number: " << std::setw(4) << p.get_prog_id() << "\n"
                << "opt_pass \t"
                << "Proc. order size \t"
                << "primitives optimized out \t"
                << "opt_pass_time \t" << "opt_pass_name\n";
        }
    }
}

void pass_manager::run(program_impl& p, base_pass& pass) {
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    pass.run(p);
    std::chrono::high_resolution_clock::time_point stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> opt_pass_time = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);
    p.save_pass_info(pass.get_name());

    if (graph_opt_log.is_open()) {
        graph_opt_log << std::setw(4) << get_pass_count() << " \t"
            << std::setw(5) << p.get_processing_order().size() << " \t"
            << std::setw(4) << p.get_optimized_out().size() << " \t"
            << std::setw(6) << opt_pass_time.count() << " \t"
            << pass.get_name() << " completed.\n";
    }

    std::string dump_file_name;
    if (pass_count < 10)
        dump_file_name += "0";
    dump_file_name += std::to_string(pass_count) + "_" + pass.get_name();
    p.dump_program(dump_file_name.c_str(), true);
    pass.clean_marks(p);
    pass_count++;
}
