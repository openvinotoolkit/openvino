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

#include "c_types_map.hpp"
#include "jit_generator.hpp"
#include "jit_uni_lrn.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <cpu_isa_t isa>
jit_uni_lrn_fwd_t<isa>::jit_uni_lrn_fwd_t(
    const pd_t *pd,
    const input_vector &inputs, const output_vector &outputs)
    : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd), ker_(nullptr)
    , ker_first_(nullptr), ker_last_(nullptr)
{
    using namespace alg_kind;

    const int C = conf_.C();
    const int H = conf_.H();
    const int W = conf_.W();
    const int ls = conf_.desc()->local_size;
    float A = conf_.desc()->lrn_alpha / ls;
    float K = conf_.desc()->lrn_k;

    auto pk = conf_.desc()->prop_kind;
    auto ak = conf_.desc()->alg_kind;
    auto dfmt = conf_.src_pd()->desc()->format;

    if (dfmt == nChw8c && ls == 5 && ak == lrn_across_channels) {
        ker_ = new jit_uni_lrn_fwd_kernel_f32<isa>(
                nchw8c_across(H, W, 0), A, K, pk);
        ker_first_ = new jit_uni_lrn_fwd_kernel_f32<isa>(
                nchw8c_across(H, W, -1), A, K, pk);
        ker_last_ = new jit_uni_lrn_fwd_kernel_f32<isa>(
                nchw8c_across(H, W, +1), A, K, pk);
    } else if (dfmt == nChw8c && ak == lrn_within_channel) {
        /* within channel, local_size (x) local_size */
        A /= ls; /* XXX: why? */
        ker_ = new jit_uni_lrn_fwd_kernel_f32<isa>(
                nchw8c_within(H, W, ls), A, K, pk);
    } else if (dfmt == nchw && ls == 5 && ak == lrn_across_channels) {
        ker_ = new jit_uni_lrn_fwd_kernel_f32<isa>(
                nchw_across(C, H*W, 0), A, K, pk);
        int remind = (H*W) % VECTOR_LENGTH;
        if (remind != 0) {
            ker_last_ = new jit_uni_lrn_fwd_kernel_f32<isa>(
                        nchw_across(C, H*W, remind), A, K, pk);
        }
    } else if (true /* XXX: why */) {
        ker_ = new jit_uni_lrn_fwd_kernel_f32<isa>(nhwc_across(C), A, K, pk);
    }
}

template <cpu_isa_t isa>
jit_uni_lrn_fwd_t<isa>::~jit_uni_lrn_fwd_t()
{ delete ker_; delete ker_first_; delete ker_last_; }

template <cpu_isa_t isa>
void jit_uni_lrn_fwd_t<isa>::execute_forward() {
    using namespace alg_kind;

    auto src = reinterpret_cast<const data_t*>(this->input_memory(0));
    auto dst = reinterpret_cast<data_t*>(this->memory(0));
    auto ws = reinterpret_cast<data_t*>(this->memory(1));

    const int N = conf_.MB();
    const int C = conf_.C();
    const int HW = conf_.H() * conf_.W();
    const int ls = conf_.desc()->local_size;

    auto ak = conf_.desc()->alg_kind;
    auto dfmt = conf_.src_pd()->desc()->format;

    if (dfmt == nChw8c && ls == 5 && ak == lrn_across_channels) {
        parallel_nd(N, C / VECTOR_LENGTH, [&](int n, int c8) {
            jit_args_fwd_t args;
            args.src = &src[n*HW*C + c8 * HW * VECTOR_LENGTH];
            args.dst = &dst[n*HW*C + c8 * HW * VECTOR_LENGTH];
            args.scratch = &ws[n*HW*C + c8 * HW * VECTOR_LENGTH];
            if (c8 == 0)
                (*ker_first_)(&args);
            else if (c8 == C / VECTOR_LENGTH - 1)
                (*ker_last_)(&args);
            else
                (*ker_)(&args);
        });
    }
    else if (dfmt == nChw8c && ak == lrn_within_channel) {
        parallel_nd(N, C / VECTOR_LENGTH, [&](int n, int c8) {
            jit_args_fwd_t args;
            args.src = &src[n*HW*C + c8 * HW * VECTOR_LENGTH];
            args.dst = &dst[n*HW*C + c8 * HW * VECTOR_LENGTH];
            args.scratch = &ws[n*HW*C + c8 * HW * VECTOR_LENGTH];
            (*ker_)(&args);
        });
    }
    else if (dfmt == nchw && ls == 5 && ak == lrn_across_channels) {
        parallel_nd(N, (HW + VECTOR_LENGTH - 1) / VECTOR_LENGTH,
            [&](int n, int hw8) {
            jit_args_fwd_t args;
            args.src = &src[n*HW*C + hw8 * VECTOR_LENGTH];
            args.dst = &dst[n*HW*C + hw8 * VECTOR_LENGTH];
            args.scratch = &ws[n*HW*C + hw8 * VECTOR_LENGTH];
            if ((hw8 + 1)*VECTOR_LENGTH > HW)
                (*ker_last_)(&args);
            else
                (*ker_)(&args);
        });
    }
    else { // nhwc
        parallel_nd(N, HW, [&](int n, int hw) {
            jit_args_fwd_t args;
            args.src = &src[n*HW*C + hw * C];
            args.dst = &dst[n*HW*C + hw * C];
            args.scratch = &ws[n*HW*C + hw * C];
            (*ker_)(&args);
        });
    }
}

template <cpu_isa_t isa>
status_t jit_uni_lrn_fwd_t<isa>::pd_t::init() {
    using namespace prop_kind;
    using namespace alg_kind;

    assert(engine()->kind() == engine_kind::cpu);

    if (!mayiuse(isa)) return unimplemented;

    const memory_desc_wrapper data_d(data_pd_.desc());
    bool ok = true
        && one_of(desc()->prop_kind, forward_training, forward_inference)
        && everyone_is(data_type::f32, desc()->data_desc.data_type)
        && !has_zero_dim_memory()
        && data_d.ndims() == 4
        && data_d.dims()[1] % VECTOR_LENGTH == 0
        && data_d.dims()[1] >= 2 * VECTOR_LENGTH
        && desc()->lrn_beta == 0.75
        && attr()->has_default_values();
    if (!ok) return unimplemented;

    if (desc_.prop_kind == forward_training) { ws_pd_ = data_pd_; }

    bool args_ok_across = true
        && desc()->alg_kind == lrn_across_channels
        && desc()->local_size == 5
        && one_of(data_d.format(), nChw8c, nchw, nhwc);

    const int jit_max_local_size = 5; // bigger size triggers too big code size
    bool args_ok_within = true
        && desc()->alg_kind == lrn_within_channel
        && desc()->local_size <= ( jit_max_local_size <= MAX_LOCAL_SIZE
                                 ? jit_max_local_size : MAX_LOCAL_SIZE)
        && data_d.dims()[2] >= desc()->local_size
        && data_d.dims()[3] >= desc()->local_size
        && one_of(data_d.format(), nChw8c);

    return args_ok_across || args_ok_within ? success : unimplemented;
}

template <cpu_isa_t isa>
jit_uni_lrn_bwd_t<isa>::jit_uni_lrn_bwd_t(const pd_t *pd,
    const input_vector &inputs, const output_vector &outputs)
    : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd)
    , ker_(nullptr), ker_first_(nullptr), ker_last_(nullptr)
{
    using namespace alg_kind;
    const int C = conf_.C();
    const int H = conf_.H();
    const int W = conf_.W();
    const int ls = conf_.desc()->local_size;
    float A = conf_.desc()->lrn_alpha / ls;
    float B = conf_.desc()->lrn_beta;

    int use_h_parallelizm = 0;// XXX
    if (C / VECTOR_LENGTH == 1) {
        ker_ = new jit_uni_lrn_bwd_kernel_f32<isa>(
            nchw8c_across(H, W, 3), A, B, use_h_parallelizm);
    }
    else {
        ker_ = new jit_uni_lrn_bwd_kernel_f32<isa>(
            nchw8c_across(H, W, 0), A, B, use_h_parallelizm);
        ker_first_ = new jit_uni_lrn_bwd_kernel_f32<isa>(
            nchw8c_across(H, W, -1), A, B, use_h_parallelizm);
        ker_last_ = new jit_uni_lrn_bwd_kernel_f32<isa>(
            nchw8c_across(H, W, +1), A, B, use_h_parallelizm);
    }
}

template <cpu_isa_t isa>
jit_uni_lrn_bwd_t<isa>::~jit_uni_lrn_bwd_t()
{
    delete ker_; delete ker_first_; delete ker_last_;
}

template <cpu_isa_t isa>
void jit_uni_lrn_bwd_t<isa>::execute_backward() {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto ws = reinterpret_cast<const data_t*>(this->input_memory(2));
    auto diff_src = reinterpret_cast<data_t*>(this->memory(0));

    const int N = conf_.MB();
    const int C = conf_.C();
    const int H = conf_.H();
    const int W = conf_.W();

    int use_h_parallelizm = 0; // XXX
    if (use_h_parallelizm) {
        parallel_nd(N, C / VECTOR_LENGTH, H, [&](int n, int c8, int h) {
            auto offset = n*C*H*W + c8*H*W*VECTOR_LENGTH
                + h*W*VECTOR_LENGTH;
            jit_args_bwd_t args;
            args.src = &src[offset];
            args.diff_dst = &diff_dst[offset];
            args.scratch = &ws[offset];
            args.diff_src = &diff_src[offset];
            if (C / VECTOR_LENGTH == 1)
                (*ker_)(&args);
            else if (c8 == 0)
                (*ker_first_)(&args);
            else if (c8 == C / VECTOR_LENGTH - 1)
                (*ker_last_)(&args);
            else
                (*ker_)(&args);
        });
    }
    else {
        parallel_nd(N, C / VECTOR_LENGTH, [&](int n, int c8) {
            auto offset = n*C*H*W + c8*H*W*VECTOR_LENGTH;
            jit_args_bwd_t args;
            args.src = &src[offset];
            args.diff_dst = &diff_dst[offset];
            args.scratch = &ws[offset];
            args.diff_src = &diff_src[offset];
            if (C / VECTOR_LENGTH == 1)
                (*ker_)(&args);
            else if (c8 == 0)
                (*ker_first_)(&args);
            else if (c8 == C / VECTOR_LENGTH - 1)
                (*ker_last_)(&args);
            else
                (*ker_)(&args);
        });
    }
}

template <cpu_isa_t isa>
status_t jit_uni_lrn_bwd_t<isa>::pd_t::init() {
    using namespace prop_kind;
    using namespace alg_kind;

    assert(engine()->kind() == engine_kind::cpu);

    if (!mayiuse(isa)) return unimplemented;

    const memory_desc_wrapper data_d(data_pd_.desc());
    bool ok = true
        && utils::one_of(desc()->prop_kind, backward, backward_data)
        && utils::everyone_is(data_type::f32, desc()->data_desc.data_type)
        && !has_zero_dim_memory()
        && data_d.ndims() == 4
        && data_d.dims()[1] % VECTOR_LENGTH == 0
        && desc()->lrn_beta == 0.75
        && attr()->has_default_values();
    if (!ok) return unimplemented;

    ws_pd_ = data_pd_;

    auto fwd_ws_d_ = hint_fwd_pd_->workspace_pd()->desc();
    bool ws_ok = true
        && fwd_ws_d_->ndims == data_d.ndims()
        && fwd_ws_d_->format == data_d.format()
        && fwd_ws_d_->data_type == data_d.data_type();
    if (!ws_ok) return unimplemented;

    bool args_ok_across = true
        && desc()->alg_kind == lrn_across_channels
        && desc()->local_size == 5
        && utils::one_of(data_d.format(), nChw8c);

    return args_ok_across ? success : unimplemented;
}

template struct jit_uni_lrn_fwd_t<sse42>;
template struct jit_uni_lrn_fwd_t<avx2>;
template struct jit_uni_lrn_bwd_t<avx2>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
