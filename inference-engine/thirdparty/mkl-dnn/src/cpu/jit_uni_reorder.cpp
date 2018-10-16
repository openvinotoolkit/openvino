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

#include <assert.h>

#include "c_types_map.hpp"
#include "memory_desc_wrapper.hpp"
#include "mkldnn_debug.h"
#include "nstl.hpp"
#include "type_helpers.hpp"

#include "cpu_primitive.hpp"
#include "cpu_reorder_pd.hpp"
#include "jit_uni_reorder.hpp"

#include "jit_generator.hpp"

// #define TR_DEBUG
#if defined(TR_DEBUG)
#define DEBUg(...) do { __VA_ARGS__ } while (0)
#else
#define DEBUg(...)
#endif
#define DEBUG(...) DEBUg(__VA_ARGS__)

using namespace Xbyak;
using namespace mkldnn::impl::types;

namespace mkldnn {
namespace impl {
namespace cpu {

namespace tr {

/** Minimal reasonable/desirable kernel size.
 * The constant might be used to determine how a problem should be split
 * between kernel and threading driver. */
const size_t ker_prb_size_min = 64;

/* kernel */
struct jit_uni_reorder_kernel_f32: public kernel_t, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_reorder_kernel_f32)

    enum {
        len_unroll_max = 256,
        ndims_jit_loop_max = 3,
    };

    struct simple_impl_desc_t {
        int ndims_full_unroll;
        int len_last_dim_unroll;
        int len_unroll;
    };

    static bool simple_impl_desc_init(const prb_t &prb,
            simple_impl_desc_t *desc) {
        const int ndims = prb.ndims;

        int ndims_full_unroll = 0;
        int len_last_dim_unroll = 1;
        int len_unroll = 1;

        for (int d = 0; d < ndims; ++d) {
            auto &node = prb.nodes[d];
            if (len_unroll * node.n <= len_unroll_max) {
                ndims_full_unroll++;
                len_unroll *= node.n;
            } else {
                len_last_dim_unroll = len_unroll_max / len_unroll;
                while (node.n % len_last_dim_unroll)
                    --len_last_dim_unroll;
                len_unroll *= len_last_dim_unroll;
                break;
            }
        }

        if (prb.ndims - ndims_full_unroll > ndims_jit_loop_max)
            return false;

        if (desc) {
            desc->ndims_full_unroll = ndims_full_unroll;
            desc->len_last_dim_unroll = len_last_dim_unroll;
            desc->len_unroll = len_unroll;
        }

        return true;
    }

    static bool applicable(const prb_t &p) {
        bool ok = true
            && utils::everyone_is(data_type::f32, p.itype, p.otype)
            && p.ndims > 0
            && utils::everyone_is(0, p.ioff, p.ooff) /* do we need this? */
            && utils::one_of(p.beta, 0.f, 1.f) /* anything else? */
            && simple_impl_desc_init(p, nullptr);
        if (!ok) return false;

        const ptrdiff_t max_stride = (1LL<<31) - 1;
        for (int d = 0; d < p.ndims; ++d) {
            const ptrdiff_t cms = max_stride / p.nodes[d].n;
            bool strides_ok = true
                && p.nodes[d].is < cms / (int)data_type_size(p.itype)
                && p.nodes[d].os < cms / (int)data_type_size(p.otype);
            if (!strides_ok) return false;
        }

        return true;
    }

    int n(int d) { assert(d < prb_.ndims); return (int)prb_.nodes[d].n; }
    int is(int d) { assert(d < prb_.ndims); return (int)prb_.nodes[d].is; }
    int os(int d) { assert(d < prb_.ndims); return (int)prb_.nodes[d].os; }

    Address i_addr(int i_off)
    { return ptr[reg_ptr_in + reg_off_in + i_off * itype_sz]; }

    Address o_addr(int o_off)
    { return ptr[reg_ptr_out + reg_off_out + o_off * otype_sz]; }

    void step(int off, int prev_i_off, int prev_o_off, int &i_off, int &o_off,
            int step_size = 1) {
        i_off = prev_i_off;
        o_off = prev_o_off;

        if (off == 0) return;

        int start_dim = 0, dims_prod = 1;
        for (; start_dim < prb_.ndims && dims_prod != step_size; ++start_dim)
            dims_prod *= n(start_dim);
        assert(start_dim < prb_.ndims);
        off /= step_size;

        for (int d = start_dim; d < prb_.ndims; ++d) {
            i_off += is(d);
            o_off += os(d);

            if (off % n(d)) break;

            i_off += - n(d) * is(d);
            o_off += - n(d) * os(d);
            off /= n(d);

            if (off == 0) break; /* FIXME: is it really required? */
        }
    }

    void tr8x8_avx2(int i_off, int o_off) {
        for (int i = 0; i < 8; i++)
            vmovups(Ymm(i), i_addr(i_off + i * 8));

        for (int i = 0; i < 8 / 2; i++) {
            vunpcklps(Ymm(8 + i), Ymm(2 * i), Ymm(2 * i + 1));
            vunpckhps(Ymm(i), Ymm(2 * i), Ymm(2 * i + 1));
        }

        const unsigned int lfloat = 0x44;
        const unsigned int ufloat = 0xee;
        for (int i = 0; i < 8 / 2; i++) {
            int j = i % 2 == 0 ? 8 + i : i - 1;
            vshufps(Ymm(8 / 2 + 2 * i), Ymm(j), Ymm(j + 1), lfloat);
            vshufps(Ymm(8 / 2 + 2 * i + 1), Ymm(j), Ymm(j + 1), ufloat);
        }

        const unsigned int lquad = 0x20;
        for (int i = 0; i < 8 / 2; i++)
            vperm2f128(Ymm(i), Ymm(8 / 2 + i), Ymm(8 + i), lquad);

        const unsigned int uquad = 0x31;
        for (int i = 8 / 2; i < 8; i++)
            vperm2f128(Ymm(i), Ymm(i), Ymm(8 / 2 + i), uquad);

        for (int i = 0; i < 8; i++)
            vmovups(o_addr(o_off + i * 8), Ymm(i));
    }

    bool process_unroll_tr8x8(int len) {
        bool can_do = true
            && mayiuse(avx2)
            && prb_.ndims >= 2
            && utils::everyone_is(8, n(0), n(1))
            && utils::everyone_is(1, os(0), is(1))
            && utils::everyone_is(8, os(1), is(0))
            && prb_.is_alpha == false
            && prb_.beta == 0.f;
        if (!can_do) return false;

        const int step_size = n(0) * n(1);
        int i_off = 0, o_off = 0;
        for (int off = 0; off < len; off += step_size) {
            step(off, i_off, o_off, i_off, o_off, step_size);
            tr8x8_avx2(i_off, o_off);
        }

        return true;
    }

    void process_unroll_generic(int len) {
        auto init = [&](int reg_unroll, int *i_off, int *o_off) {
            i_off[0] = o_off[0] = 0;
            for (int ur = 1; ur < reg_unroll; ++ur)
                step(ur, i_off[ur-1], o_off[ur-1], i_off[ur], o_off[ur]);
        };

        const int blk = 8;

        int i_off[blk] = {0};
        int o_off[blk] = {0};

        for (int off = 0; off < len; off += blk) {
            const int reg_unroll = nstl::min(off + blk, len) - off;

            /* compute offsets */
            if (off == 0) {
                init(reg_unroll, i_off, o_off);
            } else {
                step(off, i_off[blk-1], o_off[blk-1], i_off[0], o_off[0]);
                for (int ur = 1; ur < reg_unroll; ++ur)
                    step(off + ur, i_off[ur-1], o_off[ur-1], i_off[ur],
                            o_off[ur]);
            }

            for (int ur = 0; ur < reg_unroll; ++ur)
                movss(Xmm(ur), i_addr(i_off[ur]));

            /* check whether storing 4 values at once is possible */
            const bool can_store_xmm = true
                && os(0) == 1
                && n(0) % 4 == 0 /* TODO: relax to support [2, 2, ...] */
                && reg_unroll % 4 == 0;

            if (can_store_xmm) {
                /* gather 0th elements of each 4 xmms into one xmm */
                for (int ur = 0; ur < reg_unroll; ur += 2)
                    unpcklps(Xmm(ur), Xmm(ur + 1));
                for (int ur = 0; ur < reg_unroll; ur += 4)
                    unpcklpd(Xmm(ur), Xmm(ur + 2));

                /* xmm <-- alpha * xmm[:] */
                if (prb_.is_alpha)
                    for (int ur = 0; ur < reg_unroll; ur += 4)
                        mulps(Xmm(ur), xmm_alpha);

                /* dst <-- beta * dst + xmm[:] */
                assert(prb_.beta == 0.f || prb_.beta == 1.f);
                if (prb_.beta == 1.f) {
		    /* non VEX instructions do not support unaligned
                       memory for instructions other than
                       movups. Because register 1 is unused, we load
                       there and then call addps */
		  for (int ur = 0; ur < reg_unroll; ur += 4)
		      if (mayiuse(avx))
		          vaddps(Xmm(ur), o_addr(o_off[ur]));
		      else {
		          movups(Xmm(1), o_addr(o_off[ur]));
		          addps(Xmm(ur), Xmm(1));
		      }
		}

                for (int ur = 0; ur < reg_unroll; ur += 4)
                    movups(o_addr(o_off[ur]), Xmm(ur));
            } else {
                /* xmm[0] <-- alpha * xmm[0] */
                if (prb_.is_alpha)
                    for (int ur = 0; ur < reg_unroll; ++ur)
                        mulss(Xmm(ur), xmm_alpha);

                /* dst <-- beta * dst + xmm[0] */
                assert(prb_.beta == 0.f || prb_.beta == 1.f);
                if (prb_.beta == 1.f)
                    for (int ur = 0; ur < reg_unroll; ++ur)
                        addss(Xmm(ur), o_addr(o_off[ur]));

                for (int ur = 0; ur < reg_unroll; ++ur)
                    movss(o_addr(o_off[ur]), Xmm(ur));
            }
        }
    }

    void loop_begin(Label &l, Reg64 reg_cnt, int len) {
        mov(reg_cnt, len);
        L(l);
    }

    void loop_end(Label &l, Reg64 reg_cnt, int len, int i_step, int o_step) {
        add(reg_off_in, i_step * itype_sz);
        add(reg_off_out, o_step * otype_sz);
        dec(reg_cnt);
        jnz(l);

        sub(reg_off_in, len * i_step * itype_sz);
        sub(reg_off_out, len * o_step * otype_sz);
    }

    bool simple_impl() {
        simple_impl_desc_t d;
        if (!simple_impl_desc_init(prb_, &d)) return false;

        const int nfu = d.ndims_full_unroll;
        const int ldu = d.len_last_dim_unroll;
        const int n_jit_loops = prb_.ndims - d.ndims_full_unroll;
        assert(n_jit_loops <= ndims_jit_loop_max);

        xor_(reg_off_in, reg_off_in);
        xor_(reg_off_out, reg_off_out);

        Label l_loop[3];
        Reg64 reg_cnt[3] = {r15, r14, r13};

        if (n_jit_loops > 2)
            loop_begin(l_loop[2], reg_cnt[2], n(nfu + 2));

        if (n_jit_loops > 1)
            loop_begin(l_loop[1], reg_cnt[1], n(nfu + 1));

        if (n_jit_loops > 0)
            loop_begin(l_loop[0], reg_cnt[0], n(nfu + 0) / ldu);

        const bool optimized = false
            || process_unroll_tr8x8(d.len_unroll);
        if (!optimized)
            process_unroll_generic(d.len_unroll);

        if (n_jit_loops > 0)
            loop_end(l_loop[0], reg_cnt[0],
                    n(nfu + 0) / ldu, is(nfu + 0) * ldu, os(nfu + 0) * ldu);

        if (n_jit_loops > 1)
            loop_end(l_loop[1], reg_cnt[1],
                    n(nfu + 1), is(nfu + 1), os(nfu + 1));

        if (n_jit_loops > 2)
            loop_end(l_loop[2], reg_cnt[2],
                    n(nfu + 2), is(nfu + 2), os(nfu + 2));

        return true;
    }

    void impl() {
        if (simple_impl()) return;
        assert(!"no implementation available");
    }

    jit_uni_reorder_kernel_f32(const desc_t &desc)
        : kernel_t(desc), jit_generator() {
        itype_sz = data_type_size(prb_.itype);
        otype_sz = data_type_size(prb_.otype);

        preamble();
#       define PARAM(x) ptr[abi_param1 + offsetof(call_param_t, x)]
        if (prb_.is_alpha) {
            auto reg_ptr_alpha_tmp = reg_ptr_in;
            mov(reg_ptr_alpha_tmp, PARAM(scales));
            movups(xmm_alpha, ptr[reg_ptr_alpha_tmp]);
        }
        mov(reg_ptr_in, PARAM(in));
        mov(reg_ptr_out, PARAM(out));
#       undef PARAM
        impl();
        postamble();
        ker_ = (void (*)(const call_param_t *))getCode();
    }

private:
    int itype_sz;
    int otype_sz;

    Reg64 reg_ptr_in = rsi;
    Reg64 reg_ptr_out = rdx;

    Reg64 reg_off_in = r8;
    Reg64 reg_off_out = r9;

    Xmm xmm_alpha = xmm15;
};

status_t kernel_t::desc_init(kernel_t::desc_t &desc, const prb_t &prb,
        int ndims_ker_max) {
    desc.prb = prb;
    desc.prb.ioff = desc.prb.ooff = 0;

    if (ndims_ker_max > prb.ndims)
        return status::invalid_arguments;

    auto ndims_ker_max_f = [&]() {
        size_t cur_size = 1;
        for (int d = 0; d < prb.ndims; cur_size *= prb.nodes[d++].n)
            if (cur_size >= ker_prb_size_min) return d;
        return prb.ndims;
    };

    if (ndims_ker_max <= 0)
        ndims_ker_max = ndims_ker_max_f();

    /* traverse through kernel implementations */
    /* TODO: find a better way to do that... */
    desc.id = 0;
    for (int ndims_ker = ndims_ker_max; ndims_ker > 0; --ndims_ker) {
        desc.prb.ndims = ndims_ker;
        if (jit_uni_reorder_kernel_f32::applicable(desc.prb))
            return status::success;
    }

    return status::unimplemented;
}

kernel_t *kernel_t::create(const kernel_t::desc_t &desc) {
    switch (desc.id) {
    case 0: return new jit_uni_reorder_kernel_f32(desc);
    default: assert(!"unknown kernel id"); return nullptr;
    }

    return nullptr;
}

}

static void prb_block_for_cache(tr::prb_t &prb) {
    if (prb.nodes[0].is % 64 == 0 && prb.nodes[0].n > 16) {
        /** an attempt to use caches more efficient and
         * address the 4K-aliasing issue */
        /* TODO: improve the logic around here */
        int j = 1;
        for (; j < prb.ndims && prb.nodes[j].is != 1; ++j);

        if (j == 1 || j == prb.ndims) return;
        if (prb.nodes[j].n > 16 && prb.nodes[j].n % 16 == 0)
            prb_node_split(prb, j, 16);

        prb_node_move(prb, j, 1);
        DEBUG({ printf("cache: "); prb_dump(prb); });
    }
}

/** finds the maximum number of dimension the kernel should process and
 * optionally splits one of the dimension to achieve better balance between
 * parallel driver and the kernel. */
static void prb_thread_kernel_balance(tr::prb_t &prb, int &ndims_ker_max) {
    size_t sz_total = 1;
    for (int d = 0; d < prb.ndims; ++d)
        sz_total *= prb.nodes[d].n;

    /* sz_drv_min is the minimal size for the parallel
     * driver required for good parallelization */
    const size_t sz_drv_min = nstl::min<size_t>(1024,
            utils::div_up(sz_total, 1024));

    /* kdims -- # of dimensions processed by a kernel
     * sz_ker_cur -- product of the dimension processed by a kernel
     * sz_drv_cur -- product of the dimension processed by a driver */

    int kdims = prb.ndims;
    size_t sz_drv_cur = 1;
    for (; kdims > 1 && sz_drv_cur < sz_drv_min; --kdims)
        sz_drv_cur *= prb.nodes[kdims - 1].n;

    size_t sz_ker_cur = 1;
    for (int d = 0; d < kdims; ++d)
        sz_ker_cur *= prb.nodes[d].n;

    /* Initially kdims is chosen so that sz_drv_cur >= sz_drv_min.
     *
     * It might happen that for chosen kdims the sz_ker_cur is too small
     * (less than tr::ker_prb_size_min). In that case try to split the
     * innermost driver dimension into two, to increase sz_ker_cur. */
    bool want_split = true
        && kdims < prb.ndims
        && sz_ker_cur < tr::ker_prb_size_min
        && sz_drv_cur > sz_drv_min;
    if (want_split) {
        /* sz_want_borrow is the minimal sz, so that:
         *  o) sz_ker_cur * sz_want_borrow >= tr::ker_prb_size_min
         *  o) current innermost driver dimension is divisible by
         *     sz_want_borrow (so that we can evenly split that
         *     dimension into two)
         *
         *  In the worst case the minimal sz_want_borrow is equal
         *  to the innermost driver dimension itself. In that case
         *  we will sacrifice it in favor of kernel (is it fine?). */
        size_t sz_want_borrow
            = utils::div_up(tr::ker_prb_size_min, sz_ker_cur);
        for (; prb.nodes[kdims].n % sz_want_borrow; ++sz_want_borrow);
        if (sz_want_borrow != prb.nodes[kdims].n)
            prb_node_split(prb, kdims, sz_want_borrow);
        kdims += 1;
        DEBUG({ printf("split: "); prb_dump(prb);
                printf("ndims_ker_max = %d\n", kdims); });
    }

    ndims_ker_max = kdims;
}

struct jit_uni_reorder_t : public cpu_primitive_t {
    struct pd_t : public cpu_reorder_pd_t {
        pd_t(const cpu_memory_pd_t *input_pd, const cpu_memory_pd_t *output_pd,
                const primitive_attr_t *attr)
            : cpu_reorder_pd_t(input_pd, output_pd, attr) {}

        DECLARE_COMMON_PD_T("jit:uni", jit_uni_reorder_t);

        static status_t create(reorder_pd_t **reorder_pd,
                const memory_pd_t *input_pd, const memory_pd_t *output_pd,
                const primitive_attr_t *attr) {
            const memory_desc_t *imd = input_pd->desc();
            const memory_desc_t *omd = output_pd->desc();

            bool args_ok = true
                && imd->data_type == data_type::f32
                && omd->data_type == data_type::f32;
            if (!args_ok)
                return impl::status::unimplemented;

            auto prb = tr::prb_t();

            status_t prb_init_status = prb_init(prb, *imd, *omd, attr);
            if (prb_init_status != success) return prb_init_status;

            DEBUG({ printf("init : "); prb_dump(prb); });
            prb_normalize(prb);
            DEBUG({ printf("norm : "); prb_dump(prb); });
            prb_simplify(prb);
            DEBUG({ printf("smpl : "); prb_dump(prb); });

            prb_block_for_cache(prb);

            int ndims_ker_max;
            prb_thread_kernel_balance(prb, ndims_ker_max);

            tr::kernel_t::desc_t ker_desc;
            status_t ker_init_status
                = tr::kernel_t::desc_init(ker_desc, prb, ndims_ker_max);
            if (ker_init_status != status::success) return ker_init_status;

            const int ndims_driver = prb.ndims - ker_desc.prb.ndims;
            if (ndims_driver > jit_uni_reorder_t::ndims_driver_max)
                return status::unimplemented;

            DEBUG({ printf("ker  : "); prb_dump(ker_desc.prb); });

            auto _pd = new pd_t((const cpu_memory_pd_t *)input_pd,
                    (const cpu_memory_pd_t *)output_pd, attr);
            if (_pd == nullptr) return out_of_memory;
            if (_pd->init() != success) { delete _pd; return unimplemented; }
            _pd->prb_ = prb;
            _pd->ker_desc_ = ker_desc;
            return safe_ptr_assign<reorder_pd_t>(*reorder_pd, _pd);
        }

        tr::prb_t prb_;
        tr::kernel_t::desc_t ker_desc_;
    };

    jit_uni_reorder_t(const pd_t *pd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd) {
        kernel_ = tr::kernel_t::create(conf_.ker_desc_);
        assert(kernel_);
    }
    ~jit_uni_reorder_t() { delete kernel_; }

    void omp_driver_0d(int off, const float *in, float *out,
            const float *scales) {
        tr::call_param_t c{in, out, scales};
        (*kernel_)(&c);
    }

    void omp_driver_1d(int off, const float *in, float *out,
            const float *scales) {
        tr::node_t *ns = conf_.prb_.nodes + off;
#       pragma omp parallel for
        for (ptrdiff_t d0 = 0; d0 < (ptrdiff_t)ns[0].n; ++d0) {
            auto c = tr::call_param_t();
            c.in = in + d0 * ns[0].is;
            c.out = out + d0 * ns[0].os;
            c.scales = scales;
            (*kernel_)(&c);
        }
    }

    void omp_driver_2d(int off, const float *in, float *out,
            const float *scales) {
        tr::node_t *ns = conf_.prb_.nodes + off;
#       pragma omp parallel for collapse(2)
        for (ptrdiff_t d1 = 0; d1 < (ptrdiff_t)ns[1].n; ++d1) {
        for (ptrdiff_t d0 = 0; d0 < (ptrdiff_t)ns[0].n; ++d0) {
            auto c = tr::call_param_t();
            c.in = in + d0 * ns[0].is + d1 * ns[1].is;
            c.out = out + d0 * ns[0].os + d1 * ns[1].os;
            c.scales = scales;
            (*kernel_)(&c);
        }
        }
    }

    void omp_driver_3d(int off, const float *in, float *out,
            const float *scales) {
        tr::node_t *ns = conf_.prb_.nodes + off;
#       pragma omp parallel for collapse(3)
        for (ptrdiff_t d2 = 0; d2 < (ptrdiff_t)ns[2].n; ++d2) {
        for (ptrdiff_t d1 = 0; d1 < (ptrdiff_t)ns[1].n; ++d1) {
        for (ptrdiff_t d0 = 0; d0 < (ptrdiff_t)ns[0].n; ++d0) {
            auto c = tr::call_param_t();
            c.in = in + d0 * ns[0].is + d1 * ns[1].is + d2 * ns[2].is;
            c.out = out + d0 * ns[0].os + d1 * ns[1].os + d2 * ns[2].os;
            c.scales = scales;
            (*kernel_)(&c);
        }
        }
        }
    }

    void omp_driver_4d(int off, const float *in, float *out,
            const float *scales) {
        tr::node_t *ns = conf_.prb_.nodes + off;
#       pragma omp parallel for collapse(4)
        for (ptrdiff_t d3 = 0; d3 < (ptrdiff_t)ns[3].n; ++d3) {
        for (ptrdiff_t d2 = 0; d2 < (ptrdiff_t)ns[2].n; ++d2) {
        for (ptrdiff_t d1 = 0; d1 < (ptrdiff_t)ns[1].n; ++d1) {
        for (ptrdiff_t d0 = 0; d0 < (ptrdiff_t)ns[0].n; ++d0) {
            auto c = tr::call_param_t();
            c.in = in + d0 * ns[0].is + d1 * ns[1].is + d2 * ns[2].is
                + d3 * ns[3].is;
            c.out = out + d0 * ns[0].os + d1 * ns[1].os + d2 * ns[2].os
                + d3 * ns[3].os;
            c.scales = scales;
            (*kernel_)(&c);
        }
        }
        }
        }
    }

    void omp_driver(const float *in, float *out, const float *scales) {
        in += conf_.prb_.ioff;
        out += conf_.prb_.ooff;

        DEBUG({ printf("prb : "); tr::prb_dump(conf_.prb_); });
        DEBUG({ printf("ker : "); tr::prb_dump(conf_.ker_desc_.prb); });

        int ndims = conf_.prb_.ndims;
        int ndims_ker = conf_.ker_desc_.prb.ndims;
        assert(ndims - ndims_ker <= ndims_driver_max);

        switch (ndims - ndims_ker) {
        case 0: omp_driver_0d(ndims_ker, in, out, scales); break;
        case 1: omp_driver_1d(ndims_ker, in, out, scales); break;
        case 2: omp_driver_2d(ndims_ker, in, out, scales); break;
        case 3: omp_driver_3d(ndims_ker, in, out, scales); break;
        case 4: omp_driver_4d(ndims_ker, in, out, scales); break;
        default: assert(!"unimplemented");
        }
    }

    virtual void execute(event_t *e) {
        auto in = reinterpret_cast<const float *>(input_memory(0));
        auto out = reinterpret_cast<float *>(memory());

        omp_driver(in, out, conf_.attr()->output_scales_.scales_);

        e->set_state(event_t::ready);
    }

    enum { ndims_driver_max = 4 };

private:
    pd_t conf_;
    tr::kernel_t *kernel_;
};

status_t jit_uni_reorder_create(reorder_pd_t **reorder_pd,
        const memory_pd_t *input_pd, const memory_pd_t *output_pd,
        const primitive_attr_t *attr) {
    return jit_uni_reorder_t::pd_t::create(reorder_pd, input_pd, output_pd,
            attr);
}

}
}
}
