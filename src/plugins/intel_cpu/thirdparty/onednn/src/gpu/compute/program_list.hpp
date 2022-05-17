/*******************************************************************************
* Copyright 2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef GPU_COMPUTE_PROGRAM_LIST_HPP
#define GPU_COMPUTE_PROGRAM_LIST_HPP

#include <cassert>
#include <unordered_map>

#include "gpu/compute/compute_engine.hpp"
#include "gpu/compute/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace compute {

class program_list_t {
public:
    program_list_t(engine_t *engine) {
        auto compute_engine = utils::downcast<compute_engine_t *>(engine);
        deleter_ = compute_engine->get_program_list_deleter();
    }

    void add(const binary_t *binary, void *program) {
        assert(programs_.count(binary) == 0);
        auto it = programs_.insert({binary, program});
        assert(it.second);
        MAYBE_UNUSED(it);
    }

    template <typename program_t>
    program_t get(const binary_t *binary) {
        static_assert(std::is_pointer<program_t>::value,
                "program_t is expected to be a pointer.");

        auto it = programs_.find(binary);
        if (it == programs_.end()) return nullptr;
        return reinterpret_cast<program_t>(it->second);
    }

    ~program_list_t() {
        assert(deleter_);
        for (const auto &p : programs_)
            deleter_(p.second);
    }

private:
    program_list_t() = delete;
    DNNL_DISALLOW_COPY_AND_ASSIGN(program_list_t);

    std::function<void(void *)> deleter_;
    std::unordered_map<const binary_t *, void *> programs_;
};

} // namespace compute
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_COMPUTE_PROGRAM_LIST_HPP
