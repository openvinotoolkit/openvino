/*******************************************************************************
* Copyright 2017 Intel Corporation
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

#include "mkldnn_types.h"
#include "mkldnn_thread.hpp"
#include "nstl.hpp"
#include "utils.hpp"
#include "jit_generator.hpp"

#include "jit_uni_softmax.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace Xbyak;

template <cpu_isa_t isa>
jit_uni_softmax_fwd_t<isa>::jit_uni_softmax_fwd_t(const pd_t *pd,
        const input_vector &inputs, const output_vector &outputs)
    : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd)
{
    kernel_ = new jit_uni_softmax_kernel_f32<isa>(conf_.jpp_);
}

template <cpu_isa_t isa>
jit_uni_softmax_fwd_t<isa>::~jit_uni_softmax_fwd_t() {
    delete kernel_;
}

template <cpu_isa_t isa>
void jit_uni_softmax_fwd_t<isa>::execute_forward()
{
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto dst = reinterpret_cast<data_t *>(this->memory(0));

    const memory_desc_wrapper data_d(conf_.src_pd());

    const auto &jpp = conf_.jpp_;

    size_t outer_size = utils::array_product(conf_.src_pd()->desc()->dims, conf_.desc()->softmax_axis);

    size_t dim = jpp.channels * jpp.inner_size;

    auto ker = [&](const int ithr, const int nthr) {
        size_t start{0}, end{0};

        const size_t work_amount = jpp.inner_size;
        balance211(work_amount, nthr, ithr, start, end);

        for (size_t ou = 0; ou < outer_size; ++ou) {
            jit_softmax_call_s args{};
            args.channels = jpp.channels;
            args.work = end - start;
            size_t off = data_d.off_l(ou*dim + start);
            args.src = src + off;
            args.dst = dst + off;

            (*kernel_)(&args);
        }
    };

#pragma omp parallel
    {
        ker(omp_get_thread_num(), omp_get_num_threads());
    }
}

template struct jit_uni_softmax_fwd_t<sse42>;
template struct jit_uni_softmax_fwd_t<avx2>;
template struct jit_uni_softmax_fwd_t<avx512_common>;

}
}
}
