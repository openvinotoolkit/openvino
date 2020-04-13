/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#ifndef CPU_REF_SHUFFLE_HPP
#define CPU_REF_SHUFFLE_HPP

#include <assert.h>

#include "cpu_isa_traits.hpp"
#include "c_types_map.hpp"
#include "cpu_shuffle_pd.hpp"
#include "cpu_engine.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template<int data_type_size>
struct ref_shuffle_t : public cpu_primitive_t {
    using shuffle_class = ref_shuffle_t<data_type_size>;

    struct pd_t: public cpu_shuffle_pd_t {
        pd_t(engine_t *engine, const shuffle_desc_t *adesc,
                const primitive_attr_t *attr,
                const shuffle_pd_t *hint_fwd_pd)
            : cpu_shuffle_pd_t(engine, adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("ref:any",shuffle_class);

        virtual status_t init() override {
            assert(this->engine()->kind() == engine_kind::cpu);

            bool ok = true
                    && data_type_size
                            == types::data_type_size(
                                       this->desc()->data_desc.data_type)
                    /*bf16<->f32 cvt operators don't work on non-avx512_core*/
                    && IMPLICATION(this->desc()->data_desc.data_type
                                       == data_type::bf16,
                               mayiuse(avx512_core));
            if (!ok)
                return status::unimplemented;
            return status::success;
        }
    };

    ref_shuffle_t(const pd_t *apd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs)
    {
        const int axis_size = pd()->axis_size();
        const int group_size = pd()->group_size();
        const int transpose_row = pd()->is_fwd() ? group_size
                                                 : axis_size / group_size;
        const int transpose_col = pd()->is_fwd() ? axis_size / group_size
                                                 : group_size;
        rev_transposed_ = (int *)malloc(axis_size * sizeof(int), 64);
        parallel_nd(transpose_col, transpose_row, [&](int i, int j) {
            rev_transposed_[j * transpose_col + i] = i * transpose_row + j;
        });
    }

    ~ref_shuffle_t() { free(rev_transposed_); }

    typedef typename typesize_traits<data_type_size>::type data_t;

    virtual void execute(event_t *e) const {
        using namespace memory_format;
        switch (pd()->data_pd()->desc()->format) {
        case nCdhw16c: execute_<nCdhw16c>(); break;
        case nChw16c:  execute_<nChw16c>(); break;
        case nCdhw8c:  execute_<nCdhw8c>(); break;
        case nChw8c:   execute_<nChw8c>(); break;
        case nCdhw4c:  execute_<nCdhw4c>(); break;
        case nChw4c:   execute_<nChw4c>(); break;
        case ncdhw:    execute_<ncdhw>(); break;
        case nchw:     execute_<nchw>(); break;
        case ndhwc:    execute_<ndhwc>(); break;
        case nhwc:     execute_<nhwc>(); break;
        default:       execute_<mkldnn_any>(); break;
        }

        e->set_state(event_t::ready);
    }

private:
    template<memory_format_t fmt>void execute_() const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
    int *rev_transposed_;
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
