/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
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

#ifndef CPU_PRIMITIVE_HPP
#define CPU_PRIMITIVE_HPP

#include "mkldnn.h"

#include "c_types_map.hpp"
#include "event.hpp"
#include "memory_tracking.hpp"
#include "primitive.hpp"
#include "scratchpad.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct cpu_memory_t;

struct cpu_primitive_t: public primitive_t {
    cpu_primitive_t(const primitive_desc_t *pd, const input_vector &inputs,
            const output_vector &outputs, bool use_global_scratchpad = false)
        : primitive_t(pd, inputs, outputs), scratchpad_buffer_(nullptr)
        , global_scratchpad_(nullptr)
    {
        size_t scratchpad_size = this->pd()->scratchpad_registry().size();
        if (use_global_scratchpad)
            global_scratchpad_ = create_scratchpad(scratchpad_size);
        else
            scratchpad_buffer_ = malloc(scratchpad_size, 64);
    }

    virtual ~cpu_primitive_t() {
        delete global_scratchpad_;
        free(scratchpad_buffer_);
    }

    virtual char *memory(size_t output_index = 0) const {
        if (output_index >= this->outputs().size()) return nullptr;
        auto p = static_cast<const cpu_primitive_t *>(
                this->outputs()[output_index]);
        return p->memory();
    }
    virtual const char *const_memory(size_t output_index = 0) const {
        if (output_index >= this->outputs().size()) return nullptr;
        auto p = static_cast<const cpu_primitive_t *>(
                this->outputs()[output_index]);
        return p->const_memory();
    }

    const char *input_memory(size_t index = 0) const {
        if (index >= this->inputs().size()) return nullptr;
        const size_t oi = this->inputs()[index].output_index;
        auto p = static_cast<const cpu_primitive_t *>(
                this->inputs()[index].primitive);
        return p->const_memory(oi);
    }

    const cpu_memory_t *output_memory_primitive(size_t index = 0) const;

protected:
    memory_tracking::grantor_t scratchpad() const {
        return pd()->scratchpad_registry().grantor(global_scratchpad_
                ? global_scratchpad_->get() : scratchpad_buffer_);
    }

private:
    /* quite ugly, but luckily both will get away in v1.0 */
    void *scratchpad_buffer_;
    scratchpad_t *global_scratchpad_;
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
