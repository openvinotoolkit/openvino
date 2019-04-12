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

#ifndef CPU_MEMORY_HPP
#define CPU_MEMORY_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "cpu_primitive.hpp"
#include "event.hpp"
#include "memory_pd.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl;
using namespace mkldnn::impl::status;

struct cpu_memory_t: public cpu_primitive_t {
    struct pd_t: public memory_pd_t {
        pd_t(engine_t *engine): memory_pd_t(engine) {}
        pd_t(engine_t *engine, const memory_desc_t *adesc)
            : memory_pd_t(engine, adesc) {}
        virtual ~pd_t() {}
        virtual pd_t *clone() const { return new pd_t(engine(), desc()); }
        virtual status_t create_primitive(primitive_t **primitive,
                const primitive_at_t *inputs, const primitive_t **outputs) const
        {
            UNUSED(inputs); UNUSED(outputs);
            return safe_ptr_assign<primitive_t>(*primitive,
                    new cpu_memory_t(this));
        }
    };

    cpu_memory_t(const pd_t *apd)
        : cpu_primitive_t(apd, input_vector(), output_vector(1, this))
        , data_(nullptr) {}
    virtual ~cpu_memory_t() {}

    virtual void execute(mkldnn::impl::event_t *e) const
    { e->set_state(event_t::ready); }

    virtual status_t get_data_handle(void **handle) const {
        *handle = static_cast<void *>(data_);
        return success;
    }
    virtual mkldnn::impl::status_t set_data_handle(void *handle) {
        data_ = static_cast<char *>(handle);
        return zero_pad();
    }

    virtual char *memory(size_t output_index = 0) const
    { assert(output_index == 0); return data_; }
    virtual const char* const_memory(size_t output_index = 0) const
    { assert(output_index == 0); return data_; }

    mkldnn::impl::status_t zero_pad() const;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
    char *data_;

    template <mkldnn::impl::data_type_t>
    mkldnn::impl::status_t typed_zero_pad() const;
};

struct cpu_view_t: public cpu_primitive_t {
    struct pd_t: public view_pd_t {
        pd_t(engine_t *engine)
            : view_pd_t(engine), src_pd_(engine), dst_pd_(engine) {}
        virtual ~pd_t() {}

        status_t init(const cpu_memory_t::pd_t *src_pd, const dims_t dims,
                const dims_t offsets) {
            using namespace memory_format;
            using namespace status;

            if (src_pd->engine() != engine()) return invalid_arguments;

            src_pd_ = *src_pd;
            const memory_desc_t &src_d = *src_pd_.desc();
            if (src_d.format == wino_fmt) return unimplemented;
            const auto &src_d_blk = src_d.layout_desc.blocking;

            memory_desc_t dst_d = src_d;
            auto &dst_d_blk = dst_d.layout_desc.blocking;

            const int ndims = dst_d.ndims;
            for (int d = 0; d < ndims; ++d) {
                /* very limited functionality for now */
                const bool ok = true
                    && offsets[d] % src_d_blk.block_dims[d] == 0 /* [r1] */
                    && src_d_blk.offset_padding_to_data[d] == 0
                    && (false
                            || dims[d] % src_d_blk.block_dims[d] == 0
                            || dims[d] < src_d_blk.block_dims[d]);
                if (!ok)
                    return unimplemented;

                const bool is_right_border
                    = offsets[d] + dims[d] == src_d.dims[d];

                dst_d.dims[d] = dims[d];
                dst_d_blk.padding_dims[d] = is_right_border
                    ? src_d_blk.padding_dims[d] - offsets[d]
                    : dst_d.dims[d];
                dst_d_blk.offset_padding_to_data[d] =
                    src_d_blk.offset_padding_to_data[d];
                dst_d_blk.offset_padding +=
                    offsets[d] / src_d_blk.block_dims[d] /* [r1] */
                    * dst_d_blk.strides[0][d];
            }

            dst_pd_ = cpu_memory_t::pd_t(engine(), &dst_d);
            return success;
        }

        static status_t create(pd_t **cpu_view_pd,
                const cpu_memory_t::pd_t *src_pd, const dims_t dims,
                const dims_t offsets) {
            pd_t *pd;
            status_t status = safe_ptr_assign<pd_t>(pd,
                    new pd_t(src_pd->engine()));
            if (status != success) return status;
            status = pd->init(src_pd, dims, offsets);
            if (status != success) return status;
            *cpu_view_pd = pd;
            return success;
        }

        virtual pd_t *clone() const override { return new pd_t(*this); }
        virtual status_t create_primitive(primitive_t **primitive,
                const primitive_at_t *inputs, const primitive_t **outputs)
            const override
        {
            primitive_t::input_vector ins(inputs, inputs + 1);
            UNUSED(outputs);
            return safe_ptr_assign<primitive_t>(*primitive,
                    new cpu_view_t(this, ins));
        }

        virtual const cpu_memory_t::pd_t *src_pd(int index = 0) const override
        { return index == 0 ? &src_pd_ : nullptr; }
        virtual const cpu_memory_t::pd_t *dst_pd(int index = 0) const override
        { return index == 0 ? &dst_pd_ : nullptr; }

        cpu_memory_t::pd_t src_pd_;
        cpu_memory_t::pd_t dst_pd_;

    protected:
        pd_t(const cpu_memory_t::pd_t &src_pd, const cpu_memory_t::pd_t &dst_pd)
            : view_pd_t(src_pd.engine()), src_pd_(src_pd), dst_pd_(dst_pd) {}
    };

    cpu_view_t(const pd_t *apd, const input_vector &inputs)
        : cpu_primitive_t(apd, inputs, output_vector(1, this))
    {}
    virtual ~cpu_view_t() {}

    virtual void execute(mkldnn::impl::event_t *e) const
    { e->set_state(event_t::ready); }

    virtual char *memory(size_t output_index = 0) const
    { assert(output_index == 0); return const_cast<char *>(input_memory()); }
    virtual const char* const_memory(size_t output_index = 0) const
    { assert(output_index == 0); return input_memory(); }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
