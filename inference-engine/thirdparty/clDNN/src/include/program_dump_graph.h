/*
// Copyright (c) 2016 Intel Corporation
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

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "program_impl.h"
#include "program_node.h"
#include "data_inst.h"
#include <fstream>

namespace cldnn
{
    std::string get_dir_path(build_options);
    std::string get_serialization_network_name(build_options);

    void dump_graph_optimized(std::ofstream&, const program_impl&);
    void dump_graph_processing_order(std::ofstream&, const program_impl&);
    void dump_graph_init(std::ofstream&, const program_impl&, std::function<bool(program_node const&)> const&);
    void dump_graph_info(std::ofstream&, const program_impl&, std::function<bool(program_node const&)> const&);
    void dump_to_xml(std::ofstream& graph, const program_impl& program, std::function<bool(program_node const&)> const& filter, std::vector<unsigned long long>& offsets, std::vector<std::string>& data_names);
    void dump_kernels(kernels_binaries_container program_binaries, std::vector<unsigned long long>& offsets, std::vector<std::string>& data_names, std::ofstream& file_stream);
    void dump_data(memory_impl& mem, std::ofstream& stream, unsigned long long& total_offset, unsigned long long type);
}