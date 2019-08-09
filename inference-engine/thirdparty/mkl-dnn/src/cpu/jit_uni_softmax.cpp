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
#include "type_helpers.hpp"

#include "jit_uni_softmax.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace Xbyak;
using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;

template <cpu_isa_t isa>
jit_uni_softmax_fwd_t<isa>::jit_uni_softmax_fwd_t(const pd_t *apd,
        const input_vector &inputs, const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs)
{
    kernel_ = new jit_uni_softmax_kernel_f32<isa>(pd()->jpp_);
}

template <cpu_isa_t isa>
jit_uni_softmax_fwd_t<isa>::~jit_uni_softmax_fwd_t() {
    delete kernel_;
}

template <cpu_isa_t isa>
void jit_uni_softmax_fwd_t<isa>::execute_forward() const
{
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto dst = reinterpret_cast<data_t *>(this->memory(0));

    const memory_desc_wrapper data_d(pd()->src_pd());

    const auto &jpp = pd()->jpp_;

    size_t outer_size = utils::array_product(pd()->src_pd()->desc()->dims, pd()->desc()->softmax_axis);

    size_t dim = jpp.channels * jpp.inner_size;

    if (jpp.inner_size > 1) {
        const size_t work_amount = outer_size;

        auto ker = [&](const int ithr, const int nthr) {
            size_t start{0}, end{0};

            balance211(work_amount, nthr, ithr, start, end);

            size_t ou{0};
            nd_iterator_init(start, ou, outer_size);

            for (size_t iwork = start; iwork < end; ++iwork) {
                auto args = jit_softmax_call_s();
                args.channels = jpp.channels;
                args.work = jpp.inner_size;
                size_t off = data_d.off_l(ou * dim);
                args.src = src + off;
                args.dst = dst + off;

                (*kernel_)(&args);

                nd_iterator_step(ou, outer_size);
            }
        };

        parallel(0, work_amount, ker);
    } else {
        int ou_blocks = div_up(outer_size, jpp.outer_block);
        const size_t work_amount = ou_blocks;

        auto ker = [&](const int ithr, const int nthr) {
            size_t start{0}, end{0};

            balance211(work_amount, nthr, ithr, start, end);

            size_t oub{0};
            nd_iterator_init(start, oub, ou_blocks);

            for (size_t iwork = start; iwork < end; ++iwork) {
                size_t work = nstl::min(jpp.outer_block, outer_size - oub * jpp.outer_block);

                auto args = jit_softmax_call_s();
                args.channels = jpp.channels;
                args.work = work;
                size_t off = data_d.off_l(oub * jpp.outer_block * dim);
                args.src = src + off;
                args.dst = dst + off;

                (*kernel_)(&args);

                nd_iterator_step(oub, ou_blocks);
            }
        };

        parallel(0, work_amount, ker);
    }
}

template struct jit_uni_softmax_fwd_t<sse42>;
template struct jit_uni_softmax_fwd_t<avx2>;
template struct jit_uni_softmax_fwd_t<avx512_common>;

}
}
}
