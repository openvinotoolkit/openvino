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
#pragma once
#include <map>
#include <mutex>
#include <vector>
#include <memory>
#include <atomic>
#include <string>

namespace cl {
class Kernel;
class KernelIntel;
}

namespace kernel_selector {
struct KernelString;
}

namespace kernel_selector {
using kernel_string = kernel_selector::KernelString;
}

namespace cldnn {
namespace gpu {

class gpu_toolkit;

class kernels_cache {
public:
    using source_code = std::vector<std::string>;

    struct program_code {
        std::vector<source_code> source;
        uint32_t kernels_counter = 0;
        std::string options;
        bool dump_custom_program = false;
        bool one_time = false;
        std::map<std::string, std::string> entry_point_to_id;
    };

    struct kernel_code {
        std::shared_ptr<kernel_selector::kernel_string> kernel_strings;
        std::string id;
        bool dump_custom_program;
        bool one_time_kernel;
    };

    typedef std::string kernel_id;
    typedef cl::KernelIntel kernel_type;
    using sorted_code = std::map<std::string, program_code>;
    using kernels_map = std::map<std::string, kernel_type>;
    using kernels_code = std::map<std::string, kernel_code>;

private:
    gpu_toolkit& _context;
    kernels_code _kernels_code;
    std::atomic<bool> _pending_compilation{false};
    std::map<std::string, kernel_type> _kernels;
    std::map<std::string, kernel_type> _one_time_kernels;  // These kernels are intended to be executed only once (can
                                                           // be removed later from the cache).
    uint32_t _prog_id;

    sorted_code get_program_source(const kernels_code& kernels_source_code) const;
    kernels_map build_program(const program_code& pcode) const;

public:
    explicit kernels_cache(gpu_toolkit& context, uint32_t prog_id);
    kernel_id set_kernel_source(const std::shared_ptr<kernel_selector::kernel_string>& kernel_string,
                                bool dump_custom_program,
                                bool one_time_kernel);
    kernel_type get_kernel(kernel_id id, bool one_time_kernel);
    gpu_toolkit& get_context() { return _context; }
    // forces compilation of all pending kernels/programs
    void build_all();
    void reset();
};

}  // namespace gpu
}  // namespace cldnn
