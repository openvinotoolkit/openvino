// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "program_impl.h"
#include "program_node.h"
#include "gpu/ocl_toolkit.h"
#include <fstream>
#include <string>

namespace cldnn {
std::string get_dir_path(build_options);
std::string get_serialization_network_name(build_options);

void dump_graph_optimized(std::ofstream&, const program_impl&);
void dump_graph_processing_order(std::ofstream&, const program_impl&);
void dump_graph_init(std::ofstream&, const program_impl&, std::function<bool(program_node const&)> const&);
void dump_graph_info(std::ofstream&, const program_impl&, std::function<bool(program_node const&)> const&);
}  // namespace cldnn