/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#include "common/dnnl_thread.hpp"
#include "cpu/x64/jit_uni_fork_softmax.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace utils;

template <cpu_isa_t isa>
jit_uni_fork_softmax_fwd_t<isa>::jit_uni_fork_softmax_fwd_t(const pd_t *apd)
        : primitive_t(apd) {}

template <cpu_isa_t isa>
status_t jit_uni_fork_softmax_fwd_t<isa>::execute(const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const uint8_t*, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(uint8_t*, DNNL_ARG_DST);

    const memory_desc_wrapper data_d(pd()->src_md());

    const auto &jpp = pd()->jpp_;

    auto real_src_md = ctx.input(DNNL_ARG_SRC)->md();
    size_t outer_size = utils::array_product(real_src_md->dims, pd()->desc()->softmax_axis);

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
                args.src = src + off * jpp.dt_size;
                args.dst = dst + off * jpp.dt_size;

                (*kernel_)(&args);

                nd_iterator_step(ou, outer_size);
            }
        };

        parallel(0, ker);
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
                args.src = src + off * jpp.dt_size;
                args.dst = dst + off * jpp.dt_size;

                (*kernel_)(&args);

                nd_iterator_step(oub, ou_blocks);
            }
        };

        parallel(0, ker);
    }

    return status::success;
}

template struct jit_uni_fork_softmax_fwd_t<sse41>;
template struct jit_uni_fork_softmax_fwd_t<avx2>;
template struct jit_uni_fork_softmax_fwd_t<avx512_common>;

}
}
}
}
