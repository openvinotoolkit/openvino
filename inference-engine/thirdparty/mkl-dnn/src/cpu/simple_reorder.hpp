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

#ifndef CPU_SIMPLE_REORDER_HPP
#define CPU_SIMPLE_REORDER_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "math_utils.hpp"
#include "mkldnn_thread.hpp"
#include "utils.hpp"

#include "cpu_reorder_pd.hpp"
#include "cpu_primitive.hpp"

#include "simple_q10n.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::data_type;

using namespace mkldnn::impl::utils;
using math::saturate;

template<impl::data_type_t type>
using data_t = typename prec_traits<type>::type;

namespace fmt_order {
    const bool keep = true;
    const bool reverse = false;
    const bool any = keep;
}

namespace spec {
struct direct_copy {};
struct direct_copy_except_dim_0 {};
struct reference {};
}

#define SIMPLE_REORDER_TEMPL_DECL \
    impl::data_type_t type_i, impl::memory_format_t fmt_i, \
    impl::data_type_t type_o, impl::memory_format_t fmt_o, bool order_keep
#define SIMPLE_REORDER_TEMPL_CALL \
    type_i, fmt_i, type_o, fmt_o, order_keep

#define DECLARE_COMMON_PARAMS() \
        const memory_desc_wrapper &input_d = pd->input_pd(); \
        const memory_desc_wrapper &output_d = pd->output_pd(); \
        const float alpha = pd->alpha(); MAYBE_UNUSED(alpha); \
        const float beta = pd->beta(); MAYBE_UNUSED(beta);

/* specific reorders: common template */
template <SIMPLE_REORDER_TEMPL_DECL, typename spec = void>
struct simple_reorder_impl {};

namespace {
bool simple_fmt_check(bool order_keep, impl::memory_format_t fmt_i,
        impl::memory_format_t fmt_o, const memory_desc_wrapper &input_d,
        const memory_desc_wrapper &output_d) {
    return input_d.format() == (order_keep ? fmt_i : fmt_o)
        && output_d.format() == (order_keep ? fmt_o : fmt_i);
}
bool simple_attr_check(const primitive_attr_t *attr, bool many_scales_support) {
    if (many_scales_support)
        return true;
    return utils::implication(attr, attr->output_scales_.mask_ == 0);
}
#define SIMPLE_IS_APPLICABLE(many_scales_support) \
    static bool is_applicable(const memory_desc_wrapper &input_d, \
            const memory_desc_wrapper &output_d, const primitive_attr_t *attr) \
    { \
        return simple_fmt_check(order_keep, fmt_i, fmt_o, input_d, output_d) \
            && simple_attr_check(attr, many_scales_support); \
    }
}

/* specific reorders: implementation */

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
    typename utils::enable_if<
        fmt_i == nchw && (fmt_o == nChw8c || fmt_o == nChw16c)
    >::type>
{
    SIMPLE_IS_APPLICABLE(false);

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        const auto &nchw_d = order_keep ? input_d : output_d;
        const auto &dims = input_d.dims();
        constexpr int blksize = fmt_o == nChw8c ? 8 : 16;

        auto ker = [&](const data_t<type_i> *i, data_t<type_o> *o, int blk_proc) {
            if (alpha == 1.0 && beta == 0.0) {
                for (int w = 0; w < dims[3]; ++w) {
                    for (int c = 0; c < blk_proc; ++c) {
                        const auto nchw_off =
                            c * nchw_d.blocking_desc().strides[0][1] + w;
                        if (order_keep) {
                            o[w * blksize + c] = data_t<type_o>(i[nchw_off]);
                        } else {
                            o[nchw_off] = data_t<type_o>(i[w * blksize + c]);
                        }
                    }
                }
            } else {
                for (int w = 0; w < dims[3]; ++w) {
                    for (int c = 0; c < blk_proc; ++c) {
                        const auto nchw_off =
                            c * nchw_d.blocking_desc().strides[0][1] + w;
                        if (order_keep) {
                            o[w * blksize + c] = data_t<type_o>(
                                alpha * i[nchw_off]
                                + (beta ? beta * o[w * blksize + c] : 0));
                        } else {
                            o[nchw_off] = data_t<type_o>(
                                alpha * i[w * blksize + c]
                                + (beta ? beta * o[nchw_off] : 0));
                        }
                    }
                }
            }
        };

        const int CB = (dims[1] - 1) / blksize + 1;
        int blktile  = (dims[1] - 1) % blksize + 1;

        parallel_nd(dims[0], CB, dims[2], [&](int n, int C, int h) {
            constexpr int i_c_mult = order_keep ? blksize : 1;
            constexpr int o_c_mult = order_keep ? 1 : blksize;
            auto i = &input[input_d.blk_off(n, i_c_mult * C, h)];
            auto o = &output[output_d.blk_off(n, o_c_mult * C, h)];
            ker(i, o, C < CB-1 ? blksize : blktile);
        });

        return success;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
    typename utils::enable_if<
        fmt_i == ncdhw && (fmt_o == nCdhw16c)
    >::type>
{
    SIMPLE_IS_APPLICABLE(false);

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        const auto &ncdhw_d = order_keep ? input_d : output_d;
        const auto &dims = input_d.dims();
        constexpr int blksize = 16;

        auto ker = [&](const data_t<type_i> *i, data_t<type_o> *o) {
            if (alpha == 1.0 && beta == 0.0) {
                for (int w = 0; w < dims[4]; ++w) {
                    for (int c = 0; c < blksize; ++c) {
                        const auto ncdhw_off =
                            c * ncdhw_d.blocking_desc().strides[0][1] + w; //to check
                        if (order_keep) {
                            o[w * blksize + c] = data_t<type_o>(i[ncdhw_off]);
                        } else {
                            o[ncdhw_off] = data_t<type_o>(i[w * blksize + c]);
                        }
                    }
                }
            } else {
                for (int w = 0; w < dims[4]; ++w) {
                    for (int c = 0; c < blksize; ++c) {
                        const auto ncdhw_off =
                            c * ncdhw_d.blocking_desc().strides[0][1] + w; //to check
                        if (order_keep) {
                            o[w * blksize + c] = data_t<type_o>(
                                alpha * i[ncdhw_off]
                                + (beta ? beta * o[w * blksize + c] : 0));
                        } else {
                            o[ncdhw_off] = data_t<type_o>(
                                alpha * i[w * blksize + c]
                                + (beta ? beta * o[ncdhw_off] : 0));
                        }
                    }
                }
            }
        };
        parallel_nd(dims[0], dims[1] / blksize, dims[2],
            [&](int n, int C, int d) {
            for (int h = 0; h < dims[3]; ++h) {
                constexpr int i_c_mult = order_keep ? blksize : 1;
                constexpr int o_c_mult = order_keep ? 1 : blksize;
                auto i = &input[input_d.blk_off(n, i_c_mult * C, d, h)];
                auto o = &output[output_d.blk_off(n, o_c_mult * C, d, h)];
                ker(i, o);
            }
        });
        return success;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
    typename utils::enable_if<
        fmt_i == nhwc && (fmt_o == nChw8c || fmt_o == nChw16c)
    >::type>
{
    SIMPLE_IS_APPLICABLE(false);

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        const auto &dims = input_d.dims();
        constexpr int blksize = fmt_o == nChw8c ? 8 : 16;
        const auto is = input_d.blocking_desc().strides[0];
        const auto os = output_d.blocking_desc().strides[0];

        round_mode_t rmode = pd->attr()->round_mode_;

        auto ker = [&](const data_t<type_i> *i, data_t<type_o> *o) {
            if (order_keep) {
                if (alpha == 1.0 && beta == 0.0) {
#                   pragma unroll
                    for (int C = 0; C < dims[1] / blksize; ++C) {
                        PRAGMA_OMP_SIMD()
                        for (int c = 0; c < blksize; ++c) {
                            o[C * os[1] + c] = qz_a1b0<data_t<type_i>,
                                data_t<type_o>>()(i[C * blksize + c], rmode);
                        }
                    }
                } else if (alpha == 1.0) {
#                   pragma unroll
                    for (int C = 0; C < dims[1] / blksize; ++C) {
                        PRAGMA_OMP_SIMD()
                        for (int c = 0; c < blksize; ++c) {
                            o[C * os[1] + c] = qz_a1<data_t<type_i>,
                                data_t<type_o>>()(i[C * blksize + c],
                                o[C * os[1] + c], beta, rmode);
                        }
                    }
                } else if (beta == 0.0) {
#                   pragma unroll
                    for (int C = 0; C < dims[1] / blksize; ++C) {
                        PRAGMA_OMP_SIMD()
                        for (int c = 0; c < blksize; ++c) {
                            o[C * os[1] + c] = qz_b0<data_t<type_i>,
                                data_t<type_o>>()(i[C * blksize + c], alpha, rmode);
                        }
                    }
                } else {
#                   pragma unroll
                    for (int C = 0; C < dims[1] / blksize; ++C) {
                        PRAGMA_OMP_SIMD()
                        for (int c = 0; c < blksize; ++c) {
                            o[C * os[1] + c] = qz<data_t<type_i>,
                                data_t<type_o>>()(i[C * blksize + c],
                                o[C * os[1] + c], alpha, beta, rmode);
                        }
                    }
                }
            } else {
                if (alpha == 1.0 && beta == 0.0) {
#                   pragma unroll
                    for (int C = 0; C < dims[1] / blksize; ++C) {
                        PRAGMA_OMP_SIMD()
                        for (int c = 0; c < blksize; ++c) {
                            o[C * blksize + c] = qz_a1b0<data_t<type_i>,
                                data_t<type_o>>()(i[C * is[1] + c], rmode);
                        }
                    }
                } else if (alpha == 1.0) {
#                   pragma unroll
                    for (int C = 0; C < dims[1] / blksize; ++C) {
                        PRAGMA_OMP_SIMD()
                        for (int c = 0; c < blksize; ++c) {
                            o[C * blksize + c] = qz_a1<data_t<type_i>,
                                data_t<type_o>>()(i[C * is[1] + c],
                                o[C * blksize + c], beta, rmode);
                        }
                    }
                } else if (beta == 0.0) {
#                   pragma unroll
                    for (int C = 0; C < dims[1] / blksize; ++C) {
                        PRAGMA_OMP_SIMD()
                        for (int c = 0; c < blksize; ++c) {
                            o[C * blksize + c] = qz_b0<data_t<type_i>,
                                data_t<type_o>>()(i[C * is[1] + c], alpha, rmode);
                        }
                    }
                } else {
#                   pragma unroll
                    for (int C = 0; C < dims[1] / blksize; ++C) {
                        PRAGMA_OMP_SIMD()
                        for (int c = 0; c < blksize; ++c) {
                            o[C * blksize + c] = qz<data_t<type_i>,
                                data_t<type_o>>()(i[C * is[1] + c],
                               o[C * blksize + c], alpha, beta, rmode);
                        }
                    }
                }
            }
        };

        parallel_nd(dims[0], dims[2], dims[3], [&](int n, int h, int w) {
            auto i = &input[input_d.blk_off(n, 0, h, w)];
            auto o = &output[output_d.blk_off(n, 0, h, w)];
            ker(i, o);
        });

        return success;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
    typename utils::enable_if<fmt_i == chwn
    && (fmt_o == nChw8c || fmt_o == nChw16c)>::type>
{
    SIMPLE_IS_APPLICABLE(false);

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        const auto &dims = input_d.dims();
        const auto i_st = input_d.blocking_desc().strides[0];
        const auto o_st = output_d.blocking_desc().strides[0];

        constexpr int blksize = fmt_o == nChw8c ? 8 : 16;
        constexpr int tsize = 16;

        constexpr int i_mult = order_keep ? blksize : 1;
        constexpr int o_mult = order_keep ? 1 : blksize;

        const auto ci_mult = order_keep ? i_st[1] : 1;
        const auto co_mult = order_keep ? 1 : o_st[1];

        auto ker = [&](const data_t<type_i> *i, data_t<type_o> *o,
                const int nsize) {
            if (alpha == 1.0 && beta == 0) {
                PRAGMA_OMP_SIMD(collapse(2))
                for (int n = 0; n < nsize; n++) {
                    for (int c = 0; c < blksize; ++c) {
                        o[n * o_st[0] + c * co_mult] =
                            data_t<type_o>(i[n * i_st[0] + c * ci_mult]);
                    }
                }
            } else {
                PRAGMA_OMP_SIMD(collapse(2))
                for (int n = 0; n < nsize; n++) {
                    for (int c = 0; c < blksize; ++c) {
                        o[n * o_st[0] + c * co_mult] = data_t<type_o>(
                            alpha * i[n * i_st[0] + c * ci_mult]
                            + (beta ? beta * o[n * o_st[0] + c * co_mult] : 0));
                    }
                }
            }
        };

        parallel_nd(dims[1] / blksize, dims[2], div_up(dims[0], tsize), dims[3],
            [&](int C, int h, int n_blk, int w) {
            int n = n_blk * tsize;
            const int nsize =
                n + tsize > dims[0] ? dims[0] - n : tsize;
            auto i = &input[n * i_st[0] + C * i_mult * i_st[1]
                + h * i_st[2] + w * i_st[3]];
            auto o = &output[n * o_st[0] + C * o_mult * o_st[1]
                + h * o_st[2] + w * o_st[3]];
            ker(i, o, nsize);
        });

        return success;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
    typename utils::enable_if<fmt_i == nChw8c && fmt_o == nChw16c>::type>
{
    SIMPLE_IS_APPLICABLE(false);

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        const auto &dims = input_d.dims();

        constexpr int blksize_16c = 16;
        constexpr int blksize_8c = 8;
        constexpr int ic_mult = order_keep ? 2 : 1;
        constexpr int oc_mult = order_keep ? 1 : 2;

        const auto stride_8c = order_keep ? input_d.blocking_desc().strides[0]
            : output_d.blocking_desc().strides[0];

        auto ker = [&](const data_t<type_i> *i, data_t<type_o> *o, int blk_proc) {
            if (alpha == 1.0 && beta == 0.0) {
                for (int blk = 0; blk < blk_proc; ++blk){
                    const int i_blk = order_keep ? blk * (int)stride_8c[1]
                        : blk * blksize_8c;
                    const int o_blk = order_keep ? blk * blksize_8c
                        : blk * (int)stride_8c[1];
                    for (int c = 0; c < blksize_8c; ++c) {
                        o[o_blk + c] = i[i_blk + c];
                    }
                }
            } else {
                for (int blk = 0; blk < 2; ++blk) {
                    const int i_blk = order_keep ? blk * (int)stride_8c[1]
                        : blk * blksize_8c;
                    const int o_blk = order_keep ? blk * blksize_8c
                        : blk * (int)stride_8c[1];
                    for (int c = 0; c < blk_proc; ++c) {
                        o[o_blk + c] = data_t<type_o>(
                            alpha * i[i_blk + c]
                            + (beta ? beta * o[o_blk + c] : 0));
                    }
                }
            }
        };

        const int CB = (dims[1] - 1) / blksize_16c + 1;
        const int blktile_16  = ((dims[1] - 1) % blksize_16c + 1);
        int blktile  = ((blktile_16 - 1) / blksize_8c + 1);

        parallel_nd(dims[0], CB, dims[2], dims[3],
            [&](int n, int C, int h, int w) {
            auto i = &input[input_d.blk_off(n, C * ic_mult, h, w)];
            auto o = &output[output_d.blk_off(n, C * oc_mult, h, w)];
            ker(i,o, C < CB-1 ? 2 : blktile );

        });

        return success;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
    typename utils::enable_if<fmt_i == nchw && fmt_o == nhwc>::type>
{
    SIMPLE_IS_APPLICABLE(false);

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        const auto &dims = input_d.dims();
        const auto is = input_d.blocking_desc().strides[0];
        const auto os = output_d.blocking_desc().strides[0];

        round_mode_t rmode = pd->attr()->round_mode_;

        auto ker = [&](const data_t<type_i> *i, data_t<type_o> *o) {
            if (order_keep) {
                if (alpha == 1.0 && beta == 0.0) {
#                   pragma unroll
                    for (int w = 0; w < dims[3]; ++w) {
                        PRAGMA_OMP_SIMD()
                        for (int c = 0; c < dims[1]; ++c) {
                            o[w * os[3] + c] = qz_a1b0<data_t<type_i>,
                                data_t<type_o>>()(i[c * is[1] + w], rmode);
                        }
                    }
                } else if (alpha == 1.0) {
#                   pragma unroll
                    for (int w = 0; w < dims[3]; ++w) {
                        PRAGMA_OMP_SIMD()
                        for (int c = 0; c < dims[1]; ++c) {
                            o[w * os[3] + c] = qz_a1<data_t<type_i>,
                                data_t<type_o>>()(i[c * is[1] + w],
                                 o[w * os[3] + c], beta, rmode);
                        }
                    }
                } else if (beta == 0.0) {
#                   pragma unroll
                    for (int w = 0; w < dims[3]; ++w) {
                        PRAGMA_OMP_SIMD()
                        for (int c = 0; c < dims[1]; ++c) {
                            o[w * os[3] + c] = qz_b0<data_t<type_i>,
                                data_t<type_o>>()(i[c * is[1] + w], alpha, rmode);
                        }
                    }
                } else {
#                   pragma unroll
                    for (int w = 0; w < dims[3]; ++w) {
                        PRAGMA_OMP_SIMD()
                        for (int c = 0; c < dims[1]; ++c) {
                            o[w * os[3] + c] = qz<data_t<type_i>,
                                data_t<type_o>>()(i[c * is[1] + w],
                                o[w * os[3] + c], alpha, beta, rmode);
                        }
                    }
                }
            } else {
                if (alpha == 1.0 && beta == 0.0) {
#                   pragma unroll
                    for (int w = 0; w < dims[3]; ++w) {
                        PRAGMA_OMP_SIMD()
                        for (int c = 0; c < dims[1]; ++c) {
                            o[c * os[1] + w] = qz_a1b0<data_t<type_i>,
                                data_t<type_o>>()(i[w * is[3] + c], rmode);
                        }
                    }
                } else if (alpha == 1.0) {
#                   pragma unroll
                    for (int w = 0; w < dims[3]; ++w) {
                        PRAGMA_OMP_SIMD()
                        for (int c = 0; c < dims[1]; ++c) {
                            o[c * os[1] + w] = qz_a1<data_t<type_i>,
                                data_t<type_o>>()(i[w * is[3] + c],
                                o[c * os[1] + w], beta, rmode);
                        }
                    }
                } else if (beta == 0.0) {
#                   pragma unroll
                    for (int w = 0; w < dims[3]; ++w) {
                        PRAGMA_OMP_SIMD()
                        for (int c = 0; c < dims[1]; ++c) {
                            o[c * os[1] + w] = qz_b0<data_t<type_i>,
                                data_t<type_o>>()(i[w * is[3] + c], alpha, rmode);
                        }
                    }
                } else {
#                   pragma unroll
                    for (int w = 0; w < dims[3]; ++w) {
                        PRAGMA_OMP_SIMD()
                        for (int c = 0; c < dims[1]; ++c) {
                            o[c * os[1] + w] = qz<data_t<type_i>,
                                data_t<type_o>>()(i[w * is[3] + c],
                                o[c * os[1] + w], alpha, beta, rmode);
                        }
                    }
                }
            }
        };

        parallel_nd(dims[0], dims[2], [&](int n, int h) {
            auto i = &input[input_d.blk_off(n, 0, h)];
            auto o = &output[output_d.blk_off(n, 0, h)];
            ker(i, o);
        });

        return success;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
    typename utils::enable_if<fmt_i == hwio && fmt_o == oihw>::type>
{
    SIMPLE_IS_APPLICABLE(false);

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        const auto &dims = input_d.dims();
        const auto is = input_d.blocking_desc().strides[0];
        const auto os = output_d.blocking_desc().strides[0];

        auto ker = [&](const data_t<type_i> *i, data_t<type_o> *o) {
            if (alpha == 1.0 && beta == 0) {
                for (int oc = 0; oc < dims[0]; ++oc) {
                    for (int kw = 0; kw < dims[3]; ++kw) {
                        if (order_keep) {
                            o[oc * os[0] + kw] = data_t<type_o>(i[kw*is[3]+oc]);
                        } else {
                            o[kw * os[3] + oc] = data_t<type_o>(i[oc*is[0]+kw]);
                        }
                    }
                }
            } else {
                for (int oc = 0; oc < dims[0]; ++oc) {
                    for (int kw = 0; kw < dims[3]; ++kw) {
                        const auto dst_off = order_keep ? oc * os[0] + kw :
                                                          kw * os[3] + oc;
                        const auto src_off = order_keep ? kw * is[3] + oc :
                                                          oc * is[0] + kw;
                        o[dst_off] = data_t<type_o>(alpha * i[src_off]
                                     + (beta ? beta * o[dst_off] : 0));
                    }
                }
            }
        };

        parallel_nd(dims[1], dims[2], [&](int ic, int kh) {
            auto i = &input[input_d.blk_off(0, ic, kh)];
            auto o = &output[output_d.blk_off(0, ic, kh)];
            ker(i, o);
        });

        return success;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
    typename utils::enable_if<fmt_i == nchw && fmt_o == chwn>::type>
{
    SIMPLE_IS_APPLICABLE(false);

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        const auto &dims = input_d.dims();

        constexpr int tsize = 16;

        const auto istrides = input_d.blocking_desc().strides[0];
        const auto ostrides = output_d.blocking_desc().strides[0];
        const auto CHW = dims[1] * dims[2] * dims[3];

        auto ker = [&](const data_t<type_i> *i, data_t<type_o> *o,
                const int nrows, const int ncols) {
            if (alpha == 1.0 && beta == 0) {
                PRAGMA_OMP_SIMD(collapse(2))
                for (int row = 0; row < nrows; ++row) {
                    for (int col = 0; col < ncols; ++col) {
                        const auto o_idx = row * ostrides[0]
                            + col * ostrides[3];
                        const auto i_idx = row * istrides[0]
                            + col * istrides[3];
                        o[o_idx] = data_t<type_o>(i[i_idx]);
                    }
                }
            } else {
                PRAGMA_OMP_SIMD(collapse(2))
                for (int row = 0; row < nrows; ++row) {
                    for (int col = 0; col < ncols; ++col) {
                        const auto o_idx = row * ostrides[0]
                            + col * ostrides[3];
                        const auto i_idx = row * istrides[0]
                            + col * istrides[3];
                        o[o_idx] = data_t<type_o>(alpha * i[i_idx]
                            + (beta ? beta * o[o_idx] : 0));
                    }
                }
            }
        };

        parallel_nd(div_up(dims[0], tsize), div_up(CHW, tsize),
            [&](int r_blk, int c_blk) {
            int r = r_blk * tsize;
            int c = c_blk * tsize;
            const int nrows =
                r + tsize > dims[0] ? dims[0] - r : tsize;
            const int ncols = c + tsize > CHW ? CHW - c : tsize;
            auto i = &input[r * istrides[0] + c * istrides[3]];
            auto o = &output[r * ostrides[0] + c * ostrides[3]];
            ker(i, o, nrows, ncols);
        });

        return success;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
    typename utils::enable_if<fmt_i == hwio
    && (fmt_o == Ohwi8o || fmt_o == Ohwi16o)>::type>
{
    SIMPLE_IS_APPLICABLE(false);

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        const auto &dims = input_d.dims();
        const auto is = input_d.blocking_desc().strides[0];
        const auto os = output_d.blocking_desc().strides[0];

        constexpr int blksize = fmt_o == Ohwi8o ? 8 : 16;

        auto ker = [&](const data_t<type_i> *i, data_t<type_o> *o) {
            if (alpha == 1.0 && beta == 0) {
                PRAGMA_OMP_SIMD(collapse(2))
                for (int O = 0; O < dims[0] / blksize; ++O) {
                    for (int oc = 0; oc < blksize; ++oc) {
                        if (order_keep) {
                            o[O * os[0] + oc] =
                                data_t<type_o>(i[O * blksize + oc]);
                        } else {
                            o[O * blksize + oc] =
                                data_t<type_o>(i[O * is[0] + oc]);
                        }
                    }
                }
            } else {
                PRAGMA_OMP_SIMD(collapse(2))
                for (int O = 0; O < dims[0] / blksize; ++O) {
                    for (int oc = 0; oc < blksize; ++oc) {
                        const auto dst_off = order_keep ? O * os[0] + oc :
                                                          O * blksize + oc;
                        const auto src_off = order_keep ? O * blksize + oc :
                                                          O * is[0] + oc;
                        o[dst_off] = data_t<type_o>(alpha * i[src_off]
                                     + (beta ? beta * o[dst_off] : 0));
                    }
                }
            }
        };

        parallel_nd(dims[2], dims[3], dims[1],
            [&](int h, int w, int ic) {
            auto i = &input[input_d.blk_off(0, ic, h, w)];
            auto o = &output[output_d.blk_off(0, ic, h, w)];
            ker(i, o);
        });

        return success;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
    typename utils::enable_if<
        (fmt_i == goihw && (fmt_o == gOIhw8i8o || fmt_o == gOIhw16i16o))
        || ((fmt_i == oihw || fmt_i == ihwo)
                && (fmt_o == OIhw8i8o || fmt_o == OIhw16i16o))
    >::type>
{
    SIMPLE_IS_APPLICABLE(false);

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        constexpr bool w_groups = fmt_i == goihw;

        const auto &_g_oihw_d = order_keep ? input_d : output_d;
        const auto &dims = input_d.dims();
        constexpr int blksize =
            (fmt_o == OIhw8i8o || fmt_o == gOIhw8i8o) ? 8 : 16;

        auto ker = [&](const data_t<type_i> *i, data_t<type_o> *o) {
            if (alpha == 1.0 && beta == 0.0) {
                for (int ic = 0; ic < blksize; ++ic) {
                for (int oc = 0; oc < blksize; ++oc) {
                    const auto _g_oihw_off =
                        oc * _g_oihw_d.blocking_desc().strides[0][w_groups + 0]
                        + ic * _g_oihw_d.blocking_desc().strides[0]
                            [w_groups + 1];
                    if (order_keep) {
                        o[ic * blksize + oc] = data_t<type_o>(i[_g_oihw_off]);
                    } else {
                        o[_g_oihw_off] = data_t<type_o>(i[ic * blksize + oc]);
                    }
                }
                }
            } else {
                for (int ic = 0; ic < blksize; ++ic) {
                for (int oc = 0; oc < blksize; ++oc) {
                    const auto _g_oihw_off =
                        oc * _g_oihw_d.blocking_desc().strides[0][w_groups + 0]
                        + ic * _g_oihw_d.blocking_desc().strides[0]
                            [w_groups + 1];
                    if (order_keep) {
                        o[ic * blksize + oc] =
                            data_t<type_o>(alpha * i[_g_oihw_off]
                            + (beta ? beta * o[ic * blksize + oc] : 0));
                    } else {
                        o[_g_oihw_off] =
                            data_t<type_o>(alpha * i[ic * blksize + oc]
                            + (beta ? beta * o[_g_oihw_off] : 0));
                    }
                }
                }
            }
        };

        const int _G = w_groups ? dims[0] : 1;

        parallel_nd(_G, dims[w_groups + 0] / blksize,
            dims[w_groups + 1] / blksize, dims[w_groups + 2],
            dims[w_groups + 3], [&](int g, int O, int I, int h, int w) {
            constexpr int i_mult = order_keep ? blksize : 1;
            constexpr int o_mult = order_keep ? 1 : blksize;
            auto i = &input[input_d.blk_off<!w_groups>(g,
                    i_mult * O, i_mult * I, h, w)];
            auto o = &output[output_d.blk_off<!w_groups>(
                    g, o_mult * O, o_mult * I, h, w)];
            ker(i, o);
        });

        return success;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
    typename utils::enable_if<
        (fmt_i == goidhw && (fmt_o == gOIdhw16i16o))
        || ((fmt_i == oidhw) && (fmt_o == OIdhw16i16o))
    >::type>
{
    SIMPLE_IS_APPLICABLE(false);

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        constexpr bool w_groups = fmt_i == goidhw;

        const auto &_g_oihw_d = order_keep ? input_d : output_d;
        const auto &dims = input_d.dims();
        constexpr int blksize = 16;

        auto ker = [&](const data_t<type_i> *i, data_t<type_o> *o) {
            if (alpha == 1.0 && beta == 0.0) {
                for (int ic = 0; ic < blksize; ++ic) {
                for (int oc = 0; oc < blksize; ++oc) {
                    const auto _g_oihw_off =
                        oc * _g_oihw_d.blocking_desc().strides[0][w_groups + 0]
                        + ic * _g_oihw_d.blocking_desc().strides[0]
                            [w_groups + 1];
                    if (order_keep) {
                        o[ic * blksize + oc] = data_t<type_o>(i[_g_oihw_off]);
                    } else {
                        o[_g_oihw_off] = data_t<type_o>(i[ic * blksize + oc]);
                    }
                }
                }
            } else {
                for (int ic = 0; ic < blksize; ++ic) {
                for (int oc = 0; oc < blksize; ++oc) {
                    const auto _g_oihw_off =
                        oc * _g_oihw_d.blocking_desc().strides[0][w_groups + 0]
                        + ic * _g_oihw_d.blocking_desc().strides[0]
                            [w_groups + 1];
                    if (order_keep) {
                        o[ic * blksize + oc] =
                            data_t<type_o>(alpha * i[_g_oihw_off]
                            + (beta ? beta * o[ic * blksize + oc] : 0));
                    } else {
                        o[_g_oihw_off] =
                            data_t<type_o>(alpha * i[ic * blksize + oc]
                            + (beta ? beta * o[_g_oihw_off] : 0));
                    }
                }
                }
            }
        };

        const int _G = w_groups ? dims[0] : 1;
        parallel_nd(_G, dims[w_groups + 0] / blksize,
            dims[w_groups + 1] / blksize, dims[w_groups + 2],
            dims[w_groups + 3], [&](int g, int O, int I, int d, int h) {
            for (int w = 0; w < dims[w_groups + 4]; ++w) {
                    constexpr int i_mult = order_keep ? blksize : 1;
                    constexpr int o_mult = order_keep ? 1 : blksize;
                    auto i = &input[input_d.blk_off<!w_groups>(g,
                            i_mult * O, i_mult * I, d, h, w)];
                    auto o = &output[output_d.blk_off<!w_groups>(
                            g, o_mult * O, o_mult * I, d, h, w)];
                    ker(i, o);
            }
        });

        return success;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
    typename utils::enable_if<
        (fmt_i == goihw && (fmt_o == gOIhw8o8i || fmt_o == gOIhw16o16i
                            || fmt_o == gIOhw16o16i))
        || (fmt_i == oihw && (fmt_o == OIhw8o8i || fmt_o == OIhw16o16i
                            || fmt_o == IOhw16o16i))
    >::type>
{
    SIMPLE_IS_APPLICABLE(false);

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        constexpr bool w_groups = fmt_i == goihw;

        const auto &_g_oihw_d = order_keep ? input_d : output_d;
        const auto &dims = input_d.dims();
        constexpr int blksize =
            (fmt_o == OIhw8o8i || fmt_o == gOIhw8o8i) ? 8 : 16;

        auto ker = [&](const data_t<type_i> *i, data_t<type_o> *o) {
            if (alpha == 1.0 && beta == 0.0) {
                for (int oc = 0; oc < blksize; ++oc) {
                for (int ic = 0; ic < blksize; ++ic) {
                    const auto _g_oihw_off =
                        oc * _g_oihw_d.blocking_desc().strides[0][w_groups + 0]
                        + ic * _g_oihw_d.blocking_desc().strides[0]
                            [w_groups + 1];
                    if (order_keep) {
                        o[oc * blksize + ic] = data_t<type_o>(i[_g_oihw_off]);
                    } else {
                        o[_g_oihw_off] = data_t<type_o>(i[oc * blksize + ic]);
                    }
                }
                }
            } else {
                for (int oc = 0; oc < blksize; ++oc) {
                for (int ic = 0; ic < blksize; ++ic) {
                    const auto _g_oihw_off =
                        oc * _g_oihw_d.blocking_desc().strides[0][w_groups + 0]
                        + ic * _g_oihw_d.blocking_desc().strides[0]
                            [w_groups + 1];
                    if (order_keep) {
                        o[oc * blksize + ic] =
                            data_t<type_o>(alpha * i[_g_oihw_off]
                            + (beta ? beta * o[oc * blksize + ic] : 0));
                    } else {
                        o[_g_oihw_off] =
                            data_t<type_o>(alpha * i[oc * blksize + ic]
                            + (beta ? beta * o[_g_oihw_off] : 0));
                    }
                }
                }
            }
        };

        const int _G = w_groups ? dims[0] : 1;

        parallel_nd(_G, dims[w_groups + 0] / blksize,
            dims[w_groups + 1] / blksize, dims[w_groups + 2],
            dims[w_groups + 3], [&](int g, int O, int I, int h, int w) {
            constexpr int i_mult = order_keep ? blksize : 1;
            constexpr int o_mult = order_keep ? 1 : blksize;
            auto i = &input[input_d.blk_off<!w_groups>(g,
                    i_mult * O, i_mult * I, h, w)];
            auto o = &output[output_d.blk_off<!w_groups>(
                    g, o_mult * O, o_mult * I, h, w)];
            ker(i, o);
        });

        return success;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
    typename utils::enable_if<
        (fmt_i == goidhw && fmt_o == gOIdhw16o16i)
        || (fmt_i == oidhw && fmt_o == OIdhw16o16i)
    >::type>
{
    SIMPLE_IS_APPLICABLE(false);

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        constexpr bool w_groups = fmt_i == goidhw;

        const auto &_g_oidhw_d = order_keep ? input_d : output_d;
        const auto &dims = input_d.dims();
        constexpr int blksize = 16;

        auto ker = [&](const data_t<type_i> *i, data_t<type_o> *o) {
            if (alpha == 1.0 && beta == 0.0) {
                for (int oc = 0; oc < blksize; ++oc) {
                for (int ic = 0; ic < blksize; ++ic) {
                    const auto _g_oidhw_off =
                        oc * _g_oidhw_d.blocking_desc().strides[0][w_groups + 0]
                        + ic * _g_oidhw_d.blocking_desc().strides[0]
                            [w_groups + 1];
                    if (order_keep) {
                        o[oc * blksize + ic] = data_t<type_o>(i[_g_oidhw_off]);
                    } else {
                        o[_g_oidhw_off] = data_t<type_o>(i[oc * blksize + ic]);
                    }
                }
                }
            } else {
                for (int oc = 0; oc < blksize; ++oc) {
                for (int ic = 0; ic < blksize; ++ic) {
                    const auto _g_oidhw_off =
                        oc * _g_oidhw_d.blocking_desc().strides[0][w_groups + 0]
                        + ic * _g_oidhw_d.blocking_desc().strides[0]
                            [w_groups + 1];
                    if (order_keep) {
                        o[oc * blksize + ic] =
                            data_t<type_o>(alpha * i[_g_oidhw_off]
                            + (beta ? beta * o[oc * blksize + ic] : 0));
                    } else {
                        o[_g_oidhw_off] =
                            data_t<type_o>(alpha * i[oc * blksize + ic]
                            + (beta ? beta * o[_g_oidhw_off] : 0));
                    }
                }
                }
            }
        };

        const int _G = w_groups ? dims[0] : 1;

        parallel_nd(_G, dims[w_groups + 0] / blksize,
            dims[w_groups + 1] / blksize, dims[w_groups + 2],
            dims[w_groups + 3], [&](int g, int O, int I, int d, int h) {
            for (int w = 0; w < dims[w_groups + 4]; ++w) {
                constexpr int i_mult = order_keep ? blksize : 1;
                constexpr int o_mult = order_keep ? 1 : blksize;
                auto i = &input[input_d.blk_off<!w_groups>(g,
                        i_mult * O, i_mult * I, d, h, w)];
                auto o = &output[output_d.blk_off<!w_groups>(
                        g, o_mult * O, o_mult * I, d, h, w)];
                ker(i, o);
            }
        });

        return success;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
    typename utils::enable_if<
        (fmt_i == goihw && fmt_o == gOihw16o)
        || (fmt_i == oihw && fmt_o == Oihw16o)
    >::type>
{
    SIMPLE_IS_APPLICABLE(false);

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        constexpr bool w_groups = fmt_i == goihw;

        const auto &_g_oihw_d = order_keep ? input_d : output_d;
        const auto strd_oc = _g_oihw_d.blocking_desc().strides[0][w_groups];
        const auto &dims = input_d.dims();
        const int blksize = 16;

        const int _G = w_groups ? dims[0] : 1;
        constexpr int i_mult = order_keep ? blksize : 1;
        constexpr int o_mult = order_keep ? 1 : blksize;

        parallel_nd(_G, dims[w_groups + 0] / blksize, dims[w_groups + 1],
            dims[w_groups + 2], dims[w_groups + 3],
            [&](int g, int O, int i, int h, int w) {
            auto inp = &input [input_d.blk_off<!w_groups>(g,
                    i_mult * O, i, h, w)];
            auto out = &output[output_d.blk_off<!w_groups>(g,
                    o_mult * O, i, h, w)];
            if (alpha == 1.0 && beta == 0.0) {
                for (int oc = 0; oc < blksize; ++oc) {
                    const auto off = oc * strd_oc;
                    if (order_keep) {
                        out[oc] = data_t<type_o>(inp[off]);
                    } else {
                        out[off] = data_t<type_o>(inp[oc]);
                    }
                }
            } else {
                for (int oc = 0; oc < blksize; ++oc) {
                    const auto off = oc * strd_oc;
                    if (order_keep) {
                        out[oc] = data_t<type_o>(alpha * inp[off] + (beta
                                    ? beta * out[oc] : 0));
                    } else {
                        out[off] = data_t<type_o>(alpha * inp[oc] + (beta
                                    ? beta * out[off] : 0));
                    }
                }
            }
        });

        return success;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
    typename utils::enable_if<
        (fmt_i == goidhw && fmt_o == gOidhw16o)
        || (fmt_i == oidhw && fmt_o == Oidhw16o)
    >::type>
{
    SIMPLE_IS_APPLICABLE(false);

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        constexpr bool w_groups = fmt_i == goidhw;

        const auto &_g_oihw_d = order_keep ? input_d : output_d;
        const auto strd_oc = _g_oihw_d.blocking_desc().strides[0][w_groups];
        const auto &dims = input_d.dims();
        const int blksize = 16;

        const int _G = w_groups ? dims[0] : 1;
        constexpr int i_mult = order_keep ? blksize : 1;
        constexpr int o_mult = order_keep ? 1 : blksize;

        parallel_nd(_G, dims[w_groups + 0] / blksize, dims[w_groups + 1],
            dims[w_groups + 2], dims[w_groups + 3], dims[w_groups + 4],
            [&](int g, int O, int i, int d, int h, int w) {
                auto inp = &input [input_d.blk_off<!w_groups>(g,
                        i_mult * O, i, d, h, w)];
                auto out = &output[output_d.blk_off<!w_groups>(g,
                        o_mult * O, i, d, h, w)];
                if (alpha == 1.0 && beta == 0.0) {
                    for (int oc = 0; oc < blksize; ++oc) {
                        const auto off = oc * strd_oc;
                        if (order_keep) {
                            out[oc] = data_t<type_o>(inp[off]);
                        } else {
                            out[off] = data_t<type_o>(inp[oc]);
                        }
                    }
                } else {
                    for (int oc = 0; oc < blksize; ++oc) {
                        const auto off = oc * strd_oc;
                        if (order_keep) {
                            out[oc] = data_t<type_o>(
                                    alpha * inp[off] + (beta
                                        ? beta * out[oc] : 0));
                        } else {
                            out[off] = data_t<type_o>(
                                    alpha * inp[oc] + (beta
                                        ? beta * out[off] : 0));
                        }
                    }
                }
        });

        return success;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
    typename utils::enable_if<(fmt_i == goihw && fmt_o == Goihw8g) ||
                              (fmt_i == goihw && fmt_o == Goihw16g)>::type>
{
    SIMPLE_IS_APPLICABLE(false);

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        constexpr bool w_groups = fmt_i == goihw;

        const auto &_goihw_d = order_keep ? input_d : output_d;
        const auto &dims = input_d.dims();
        const int blksize = fmt_o == Goihw8g ? 8 : 16;

        const int NG = dims[0];
        constexpr int i_mult = order_keep ? blksize : 1;
        constexpr int o_mult = order_keep ? 1 : blksize;

        parallel_nd(NG / blksize, dims[1], dims[2], dims[3], dims[4],
            [&](int G, int oc, int ic, int h, int w) {
            auto i = &input[input_d.blk_off<!w_groups>(
                   G * i_mult, oc, ic, h, w)];
            auto o = &output[output_d.blk_off<!w_groups>(
                   G * o_mult, oc, ic, h, w)];
            if (alpha == 1.0 && beta == 0.0) {
                for (int g = 0; g < blksize; ++g) {
                    const auto _goihw_off = g *
                        _goihw_d.blocking_desc().strides[0][0];
                    if (order_keep) {
                        o[g] = data_t<type_o>(i[_goihw_off]);
                    } else {
                        o[_goihw_off] = data_t<type_o>(i[g]);
                    }
                }
            } else {
                for (int g = 0; g < blksize; ++g) {
                    const auto _goihw_off = g *
                        _goihw_d.blocking_desc().strides[0][0];
                    if (order_keep) {
                        o[g] = data_t<type_o>(alpha * i[_goihw_off] +
                             (beta ? beta * o[g] : 0));
                   } else {
                        o[_goihw_off] = data_t<type_o>(alpha *
                             i[g] + (beta ? beta * o[_goihw_off] : 0));
                   }
               }
           }
        });

        return success;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
    typename utils::enable_if<
        fmt_i == hwio && (fmt_o == OIhw8i8o || fmt_o == OIhw16i16o)
    >::type>
{
    SIMPLE_IS_APPLICABLE(false);

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        const auto &_hwio_d = order_keep ? input_d : output_d;
        const auto &dims = input_d.dims();
        constexpr int blksize = fmt_o == OIhw8i8o ? 8 : 16;
        const auto _hwio_st = _hwio_d.blocking_desc().strides[0];

        auto ker = [&](const data_t<type_i> *i, data_t<type_o> *o) {
            if (alpha == 1.0 && beta == 0.0) {
                PRAGMA_OMP_SIMD(collapse(2))
                for (int ic = 0; ic < blksize; ++ic) {
                    for (int oc = 0; oc < blksize; ++oc) {
                        if (order_keep) {
                            o[ic * blksize + oc] =
                                data_t<type_o>(i[oc + ic * _hwio_st[1]]);
                        } else {
                            o[oc + ic * _hwio_st[1]] =
                                data_t<type_o>(i[ic * blksize + oc]);
                        }
                    }
                }
            } else {
                PRAGMA_OMP_SIMD(collapse(2))
                for (int ic = 0; ic < blksize; ++ic) {
                    for (int oc = 0; oc < blksize; ++oc) {
                        const auto dst_off = order_keep ? ic * blksize + oc :
                                                          ic * _hwio_st[1] + oc;
                        const auto src_off = order_keep ? ic * _hwio_st[1] + oc :
                                                          ic * blksize + oc;
                        o[dst_off] = data_t<type_o>(alpha * i[src_off]
                                     + (beta ? beta * o[dst_off] : 0));
                    }
                }
            }
        };

        parallel_nd(dims[2], dims[3], dims[0] / blksize, dims[1] / blksize,
            [&](int h, int w, int O, int I) {
            constexpr int i_mult = order_keep ? blksize : 1;
            constexpr int o_mult = order_keep ? 1 : blksize;
            auto i = &input[input_d.blk_off(
                    i_mult * O, i_mult * I, h, w)];
            auto o = &output[output_d.blk_off(
                    o_mult * O, o_mult * I, h, w)];
            ker(i, o);
        });

        return success;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
    typename utils::enable_if<
          (fmt_i == goihw && (fmt_o == gOIhw4i16o4i || fmt_o == gOIhw8i16o2i))
       || (fmt_i == oihw && (fmt_o == OIhw4i16o4i || fmt_o == OIhw8i16o2i))
    >::type>
{
    SIMPLE_IS_APPLICABLE(false);

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        static constexpr bool w_groups = fmt_i == goihw;
        int sblk = fmt_o == OIhw4i16o4i || fmt_o == gOIhw4i16o4i ? 4 : 2;

        constexpr int is_3d = false;

        const auto &_g_oihw_d = order_keep ? input_d : output_d;
        const auto &dims = input_d.dims();
        const auto &pdims = order_keep
            ? output_d.blocking_desc().padding_dims
            : input_d.blocking_desc().padding_dims;

        const int blksize = 16;
        const int G = w_groups ? dims[0] : 1;
        const int OC = dims[w_groups + 0];
        const int NB_OC = pdims[w_groups + 0] / blksize;
        const int IC = dims[w_groups + 1];
        const int NB_IC = pdims[w_groups + 1] / blksize;
        //const int D = is_3d ? dims[w_groups + 2] : 1;
        const int H = dims[w_groups + 2 + is_3d];
        const int W = dims[w_groups + 3 + is_3d];

        auto index = [&](const int ic, const int oc) {
            return ((ic / sblk) * blksize * sblk + sblk * oc + ic % sblk);
        };

        auto ker = [&](const data_t<type_i> *i, data_t<type_o> *o,
                const int oc_block, const int ic_block) {
            if (alpha == 1.0 && beta == 0.0) {
                for (int ic = 0; ic < ic_block; ++ic) {
                for (int oc = 0; oc < oc_block; ++oc) {
                    const auto _g_oihw_off =
                        oc * _g_oihw_d.blocking_desc().strides[0][w_groups + 0]
                      + ic * _g_oihw_d.blocking_desc().strides[0][w_groups + 1];
                    if (order_keep) {
                        o[index(ic, oc)] =
                            data_t<type_o>(i[_g_oihw_off]);
                    } else {
                        o[_g_oihw_off] =
                            data_t<type_o>(i[index(ic, oc)]);
                    }
                }
                }
            } else {
                for (int ic = 0; ic < ic_block; ++ic) {
                for (int oc = 0; oc < oc_block; ++oc) {
                    const auto _g_oihw_off =
                        oc * _g_oihw_d.blocking_desc().strides[0][w_groups + 0]
                      + ic * _g_oihw_d.blocking_desc().strides[0][w_groups + 1];
                    if (order_keep) {
                        o[index(ic, oc)] = data_t<type_o>(
                            alpha * i[_g_oihw_off]
                            + (beta ? beta * o[index(ic, oc)] : 0));
                    } else {
                        o[_g_oihw_off] = data_t<type_o>(
                            alpha * i[index(ic, oc)]
                            + (beta ? beta * o[_g_oihw_off] : 0));
                    }
                }
                }
            }
        };

        constexpr int i_mult = order_keep ? blksize : 1;
        constexpr int o_mult = order_keep ? 1 : blksize;

        parallel_nd(G, NB_OC, NB_IC, H, W,
            [&](int g, int O, int I, int h, int w) {
            auto i = &input[input_d.blk_off<!w_groups>(g,
                    i_mult * O, i_mult * I, h, w)];
            auto o = &output[output_d.blk_off<!w_groups>(
                    g, o_mult * O, o_mult * I, h, w)];
            const int oc_block = nstl::min(blksize, OC - O * blksize);
            const int ic_block = nstl::min(blksize, IC - I * blksize);
            ker(i, o, oc_block, ic_block);
        });
        return success;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
    typename utils::enable_if<
        (fmt_i == gOIhw8i16o2i && fmt_o == gOIhw8o16i2o)
        || (fmt_i == OIhw8i16o2i && fmt_o == OIhw8o16i2o)
    >::type>
{
    SIMPLE_IS_APPLICABLE(false);

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        static constexpr bool w_groups = fmt_i == gOIhw8i16o2i;

        const auto &dims = input_d.dims();
        const int blksize = 16;

        auto index_src = [&](const int ic, const int oc) {
            return ((ic / 2) * blksize * 2 + 2 * oc + ic % 2);
        };
        auto index_dst = [&](const int ic, const int oc) {
            return ((oc / 2) * blksize * 2 + 2 * ic + oc % 2);
        };

        auto ker = [&](const data_t<type_i> *i, data_t<type_o> *o) -> void {
            if (alpha == 1.0 && beta == 0.0) {
                for (int ic = 0; ic < blksize; ++ic) {
                    for (int oc = 0; oc < blksize; ++oc) {
                        o[index_dst(ic,oc)] = data_t<type_o>(i[index_src(ic,oc)]);
                    }
                }
            } else {
                for (int ic = 0; ic < blksize; ++ic) {
                    for (int oc = 0; oc < blksize; ++oc) {
                        o[index_dst(ic,oc)] = data_t<type_o>(
                                alpha * i[index_src(ic,oc)]
                                + (beta ? beta * o[index_dst(ic,oc)] : 0));
                    }
                }
            }
        };

        const int _G = w_groups ? dims[0] : 1;

        parallel_nd(_G, dims[w_groups + 0] / blksize,
            dims[w_groups + 1] / blksize, dims[w_groups + 2],
            dims[w_groups + 3], [&](int g, int o, int i, int h, int w) {
            auto i_ptr = &input[input_d.blk_off<!w_groups>(g, o, i, h, w)];
            auto o_ptr = &output[output_d.blk_off<!w_groups>(g, o, i, h, w)];
            ker(i_ptr, o_ptr);
        });

        return success;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
    typename utils::enable_if<
        (fmt_i == gOIhw8i8o && fmt_o == gOIhw8o8i)
        || (fmt_i == OIhw8i8o && fmt_o == OIhw8o8i)
        || (fmt_i == gOIhw16i16o && fmt_o == gOIhw16o16i)
        || (fmt_i == OIhw16i16o && fmt_o == OIhw16o16i)
        || (fmt_i == gOIhw16i16o && fmt_o == gIOhw16o16i)
        || (fmt_i == OIhw16i16o && fmt_o == IOhw16o16i)
    >::type>
{
    SIMPLE_IS_APPLICABLE(false);

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        constexpr bool w_groups = (fmt_i == gOIhw8i8o || fmt_i == gOIhw16i16o);

        const auto &dims = input_d.dims();
        constexpr int blksize =
            (fmt_i == OIhw8i8o || fmt_i == gOIhw8i8o) ? 8 : 16;

        auto ker = [&](const data_t<type_i> *i, data_t<type_o> *o) {
            for (int ic = 0; ic < blksize; ++ic) {
                for (int oc = 0; oc < blksize; ++oc) {
                    const int o_idx = ic * blksize + oc;
                    const int i_idx = oc * blksize + ic;
                    o[o_idx] = (alpha == 1.0 && beta == 0.0)
                        ? data_t<type_o>(i[i_idx])
                        : data_t<type_o>(alpha * i[i_idx]
                            + (beta ? beta * o[o_idx] : 0));
                }
            }
        };

        const int _G = w_groups ? dims[0] : 1;

        parallel_nd(_G, dims[w_groups + 0] / blksize,
            dims[w_groups + 1] / blksize, dims[w_groups + 2],
            dims[w_groups + 3], [&](int g, int o, int i, int h, int w) {
            auto i_ptr = &input[input_d.blk_off<!w_groups>(g,
                    o, i, h, w)];
            auto o_ptr = &output[output_d.blk_off<!w_groups>(g,
                    o, i, h, w)];
            ker(i_ptr, o_ptr);
        });

        return success;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
    typename utils::enable_if<
        (fmt_i == gOIdhw16i16o && fmt_o == gOIdhw16o16i)
        || (fmt_i == OIdhw16i16o && fmt_o == OIdhw16o16i)
    >::type>
{
    SIMPLE_IS_APPLICABLE(false);

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        constexpr bool w_groups = fmt_i == gOIdhw16i16o;

        const auto &dims = input_d.dims();
        constexpr int blksize = 16;

        auto ker = [&](const data_t<type_i> *i, data_t<type_o> *o) {
            for (int ic = 0; ic < blksize; ++ic) {
                for (int oc = 0; oc < blksize; ++oc) {
                    const int o_idx = ic * blksize + oc;
                    const int i_idx = oc * blksize + ic;
                    o[o_idx] = (alpha == 1.0 && beta == 0.0)
                        ? data_t<type_o>(i[i_idx])
                        : data_t<type_o>(alpha * i[i_idx]
                            + (beta ? beta * o[o_idx] : 0));
                }
            }
        };

        const int _G = w_groups ? dims[0] : 1;

        parallel_nd(_G, dims[w_groups + 0] / blksize,
            dims[w_groups + 1] / blksize, dims[w_groups + 2],
            dims[w_groups + 3], [&](int g, int o, int i, int d, int h) {
            for (int w = 0; w < dims[w_groups + 4]; ++w) {
                auto i_ptr = &input[input_d.blk_off<!w_groups>(g,
                        o, i, d, h, w)];
                auto o_ptr = &output[output_d.blk_off<!w_groups>(g,
                        o, i, d, h, w)];
                ker(i_ptr, o_ptr);
            }
        });

        return success;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
    typename utils::enable_if<
        (fmt_i == Oihw16o && fmt_o == Ohwi16o)
        || (fmt_i == gOihw16o && fmt_o == gOhwi16o)
    >::type>
{
    SIMPLE_IS_APPLICABLE(false);

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        const bool w_groups = fmt_i == gOihw16o;

        const auto &dims = input_d.dims();
        const int blksize = 16;

        const int _G = w_groups ? dims[0] : 1;

        parallel_nd(_G, dims[w_groups + 0] / blksize, dims[w_groups + 1],
            dims[w_groups + 2], dims[w_groups + 3],
            [&](int g, int o, int i, int h, int w) {
            auto i_ptr = &input[input_d.blk_off<!w_groups>(g,
                    o, i, h, w)];
            auto o_ptr = &output[output_d.blk_off<!w_groups>(g,
                    o, i, h, w)];
            for (int oc = 0; oc < blksize; ++oc) {
                o_ptr[oc] = (alpha == 1.0 && beta == 0.0)
                    ? data_t<type_o>(i_ptr[oc])
                    : data_t<type_o>(alpha * i_ptr[oc]
                        + (beta ? beta * o_ptr[oc] : 0));
            }
        });

        return success;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
    typename utils::enable_if<
        (fmt_i == Oidhw16o && fmt_o == Odhwi16o)
        || (fmt_i == gOidhw16o && fmt_o == gOdhwi16o)
    >::type>
{
    SIMPLE_IS_APPLICABLE(false);

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        const bool w_groups = fmt_i == gOidhw16o;

        const auto &dims = input_d.dims();
        const int blksize = 16;

        const int _G = w_groups ? dims[0] : 1;

        parallel_nd(_G, dims[w_groups + 0] / blksize, dims[w_groups + 1],
            dims[w_groups + 2], dims[w_groups + 3], dims[w_groups + 4],
            [&](int g, int o, int i, int d, int h, int w) {
            auto i_ptr = &input[input_d.blk_off<!w_groups>(g,
                    o, i, d, h, w)];
            auto o_ptr = &output[output_d.blk_off<!w_groups>(g,
                    o, i, d, h, w)];
            for (int oc = 0; oc < blksize; ++oc) {
                o_ptr[oc] = (alpha == 1.0 && beta == 0.0)
                    ? data_t<type_o>(i_ptr[oc])
                    : data_t<type_o>(alpha * i_ptr[oc]
                        + (beta ? beta * o_ptr[oc] : 0));
            }
        });

        return success;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
    typename utils::enable_if<
        fmt_i == any
        && (fmt_o == nChw8c || fmt_o == nChw16c || fmt_o == nCdhw16c)
    >::type>
{
    static bool is_applicable(const memory_desc_wrapper &input_d,
        const memory_desc_wrapper &output_d, const primitive_attr_t *attr) {
        return order_keep
            ? output_d.format() == fmt_o && utils::one_of(input_d.format(),
                    nchw, nhwc, chwn, ncdhw, ndhwc)
            : input_d.format() == fmt_o && utils::one_of(output_d.format(),
                    nchw, nhwc, chwn, ncdhw, ndhwc);
    }

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        constexpr int is_3d = fmt_o == nCdhw16c;
        constexpr int blksize = fmt_o == nChw8c ? 8 : 16;

        const auto &flat_d = order_keep ? input_d : output_d;
        const auto &dims = input_d.dims();
        const auto &pdims = order_keep
            ? output_d.blocking_desc().padding_dims
            : input_d.blocking_desc().padding_dims;

        const int C = dims[1];
        const int D = is_3d ? dims[2] : 1;
        const int H = dims[2 + is_3d];
        const int W = dims[3 + is_3d];

        auto ker = [&](const data_t<type_i> *i, data_t<type_o> *o,
            const int c_block) {
            if (alpha == 1.0 && beta == 0.0) {
                for (int w = 0; w < W; ++w)
                for (int c = 0; c < c_block; ++c) {
                    const ptrdiff_t flat_off = 0
                        + c * flat_d.blocking_desc().strides[0][1]
                        + w * flat_d.blocking_desc().strides[0][3 + is_3d];
                    if (order_keep) {
                        o[w * blksize + c] = data_t<type_o>(i[flat_off]);
                    } else {
                        o[flat_off] = data_t<type_o>(i[w * blksize + c]);
                    }
                }
            } else {
                for (int w = 0; w < W; ++w)
                for (int c = 0; c < c_block; ++c) {
                    const ptrdiff_t flat_off = 0
                        + c * flat_d.blocking_desc().strides[0][1]
                        + w * flat_d.blocking_desc().strides[0][3 + is_3d];
                    if (order_keep) {
                        o[w * blksize + c] = data_t<type_o>(
                            alpha * i[flat_off]
                            + (beta ? beta * o[w * blksize + c] : 0));
                    } else {
                        o[flat_off] = data_t<type_o>(
                            alpha * i[w * blksize + c]
                            + (beta ? beta * o[flat_off] : 0));
                    }
                }
            }
        };

        constexpr int i_c_mult = order_keep ? blksize : 1;
        constexpr int o_c_mult = order_keep ? 1 : blksize;

        parallel_nd(dims[0], pdims[1] / blksize, D, H,
            [&](int n, int nb_c, int d, int h) {
            auto i = &input[is_3d
                ? input_d.blk_off(n, i_c_mult * nb_c, d, h)
                : input_d.blk_off(n, i_c_mult * nb_c, h)];
            auto o = &output[is_3d
                ? output_d.blk_off(n, o_c_mult * nb_c, d, h)
                : output_d.blk_off(n, o_c_mult * nb_c, h)];
            const int c_block = nstl::min(blksize, C - nb_c * blksize);
            ker(i, o, c_block);
        });

        return success;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
    typename utils::enable_if<fmt_i == any &&
    (false
     || fmt_o == OIhw16i16o || fmt_o == gOIhw16i16o
     || fmt_o == OIdhw16i16o || fmt_o == gOIdhw16i16o
     || fmt_o == OIhw16o16i || fmt_o == gOIhw16o16i
     || fmt_o == OIdhw16o16i || fmt_o == gOIdhw16o16i
     || fmt_o == IOhw16o16i || fmt_o == gIOhw16o16i
     )>::type>
{
    static bool is_applicable(const memory_desc_wrapper &input_d,
        const memory_desc_wrapper &output_d, const primitive_attr_t *attr) {
        return order_keep
            ? output_d.format() == fmt_o && utils::one_of(input_d.format(),
                    oihw, ihwo, hwio, goihw, hwigo, dhwio, oidhw, goidhw)
            : input_d.format() == fmt_o &&  utils::one_of(output_d.format(),
                    oihw, ihwo, hwio, goihw, hwigo, dhwio, oidhw, goidhw);
    }

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        static constexpr bool w_groups = false
            || fmt_o == gOIhw16i16o || fmt_o == gOIdhw16i16o
            || fmt_o == gOIhw16o16i || fmt_o == gOIdhw16o16i
            || fmt_o == gIOhw16o16i;

        constexpr int is_3d = false
            || fmt_o == OIdhw16i16o || fmt_o == gOIdhw16i16o
            || fmt_o == OIdhw16o16i || fmt_o == gOIdhw16o16i;

        const auto &flat_d = order_keep ? input_d : output_d;
        const auto &dims = input_d.dims();
        const auto &pdims = order_keep
            ? output_d.blocking_desc().padding_dims
            : input_d.blocking_desc().padding_dims;

        constexpr int blksize = 16;
        const int G = w_groups ? dims[0] : 1;
        const int OC = dims[w_groups + 0];
        const int NB_OC = pdims[w_groups + 0] / blksize;
        const int IC = dims[w_groups + 1];
        const int NB_IC = pdims[w_groups + 1] / blksize;
        const int D = is_3d ? dims[w_groups + 2] : 1;
        const int H = dims[w_groups + 2 + is_3d];
        const int W = dims[w_groups + 3 + is_3d];

        auto index = [&](const int ic, const int oc) {
            if (fmt_o == OIhw16i16o || fmt_o == gOIhw16i16o ||
                    fmt_o == OIdhw16i16o || fmt_o == gOIdhw16i16o)
                return ic * blksize + oc;
            else
                return oc * blksize + ic;
        };

        auto ker = [&](const data_t<type_i> *i, data_t<type_o> *o,
            const int oc_block, const int ic_block) {
            if (alpha == 1.0 && beta == 0.0) {
                for (int oc = 0; oc < oc_block; ++oc)
                for (int ic = 0; ic < ic_block; ++ic) {
                    const ptrdiff_t flat_off = 0
                        + oc * flat_d.blocking_desc().strides[0][w_groups + 0]
                        + ic * flat_d.blocking_desc().strides[0][w_groups + 1];
                    if (order_keep) {
                        o[index(ic, oc)] = data_t<type_o>(i[flat_off]);
                    } else {
                        o[flat_off] = data_t<type_o>(i[index(ic, oc)]);
                    }
                }
            } else {
                for (int oc = 0; oc < oc_block; ++oc)
                for (int ic = 0; ic < ic_block; ++ic) {
                    const ptrdiff_t flat_off = 0
                        + oc * flat_d.blocking_desc().strides[0][w_groups + 0]
                        + ic * flat_d.blocking_desc().strides[0][w_groups + 1];
                    if (order_keep) {
                        o[index(ic, oc)] = data_t<type_o>(alpha * i[flat_off]
                                + (beta ? beta * o[index(ic, oc)] : 0));
                    } else {
                        o[flat_off] = data_t<type_o>(alpha * i[index(ic, oc)]
                                + (beta ? beta * o[flat_off] : 0));
                    }
                }
            }
        };

        constexpr int i_mult = order_keep ? blksize : 1;
        constexpr int o_mult = order_keep ? 1 : blksize;

        parallel_nd(G, NB_OC, NB_IC, D, H, W,
            [&](int g, int nb_oc, int nb_ic, int d, int h, int w) {
            auto i = &input[is_3d
                ? input_d.blk_off<!w_groups>(g, i_mult * nb_oc, i_mult * nb_ic, d, h, w)
                : input_d.blk_off<!w_groups>(g, i_mult * nb_oc, i_mult * nb_ic, h, w)];
            auto o = &output[is_3d
                ? output_d.blk_off<!w_groups>(g, o_mult * nb_oc, o_mult * nb_ic, d, h, w)
                : output_d.blk_off<!w_groups>(g, o_mult * nb_oc, o_mult * nb_ic, h, w)];
            const int oc_block = nstl::min(blksize, OC - nb_oc * blksize);
            const int ic_block = nstl::min(blksize, IC - nb_ic * blksize);
            ker(i, o, oc_block, ic_block);
        });

        return success;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
    typename utils::enable_if<fmt_i == any &&
    (false
     || fmt_o == Oihw16o || fmt_o == Ohwi16o || fmt_o == Oidhw16o
     || fmt_o == Odhwi16o
     || fmt_o == gOihw16o || fmt_o == gOhwi16o || fmt_o == gOidhw16o
     || fmt_o == gOdhwi16o
     )>::type>
{
    static bool is_applicable(const memory_desc_wrapper &input_d,
        const memory_desc_wrapper &output_d, const primitive_attr_t *attr) {
        return order_keep
            ? output_d.format() == fmt_o && utils::one_of(input_d.format(),
                    oihw, ihwo, hwio, goihw, hwigo, dhwio, oidhw, goidhw)
            : input_d.format() == fmt_o && utils::one_of(output_d.format(),
                    oihw, ihwo, hwio, goihw, hwigo, dhwio, oidhw, goidhw);
    }

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        constexpr int blksize = 16;

        static constexpr bool w_groups = fmt_o == gOihw16o || fmt_o == gOhwi16o
            || fmt_o == gOidhw16o || fmt_o == gOdhwi16o;
        constexpr int is_3d = false
            || fmt_o == gOidhw16o || fmt_o == Oidhw16o
            || fmt_o == gOdhwi16o || fmt_o == Odhwi16o;

        const auto &flat_d = order_keep ? input_d : output_d;
        const auto &dims = input_d.dims();
        const auto &pdims = order_keep
            ? output_d.blocking_desc().padding_dims
            : input_d.blocking_desc().padding_dims;

        const int G = w_groups ? dims[0] : 1;
        const int OC = dims[w_groups + 0];
        const int IC = dims[w_groups + 1];
        const int D = is_3d ? dims[w_groups + 2] : 1;
        const int H = dims[w_groups + 2 + is_3d];
        const int W = dims[w_groups + 3 + is_3d];

        constexpr int i_mult = order_keep ? blksize : 1;
        constexpr int o_mult = order_keep ? 1 : blksize;
        const auto strd_oc = flat_d.blocking_desc().strides[0][w_groups];

        parallel_nd(G, pdims[w_groups + 0] / blksize, IC, D, H, W,
            [&](int g, int nb_oc, int ic, int d, int h, int w) {
            auto inp = &input[is_3d
                ? input_d.blk_off<!w_groups>(g, i_mult * nb_oc, ic, d, h, w)
                : input_d.blk_off<!w_groups>(g, i_mult * nb_oc, ic, h, w)];
            auto out = &output[is_3d
                ? output_d.blk_off<!w_groups>(g, o_mult * nb_oc, ic, d, h, w)
                : output_d.blk_off<!w_groups>(g, o_mult * nb_oc, ic, h, w)];
            const int oc_block = nstl::min(blksize, OC - nb_oc * blksize);
            if (alpha == 1.0 && beta == 0.0) {
                for (int oc = 0; oc < oc_block; ++oc) {
                    const auto off = oc * strd_oc;
                    if (order_keep) {
                        out[oc] = data_t<type_o>(inp[off]);
                    } else {
                        out[off] = data_t<type_o>(inp[oc]);
                    }
                }
            } else {
                for (int oc = 0; oc < oc_block; ++oc) {
                    const auto off = oc * strd_oc;
                    if (order_keep) {
                        out[oc] = data_t<type_o>(alpha * inp[off]
                                + (beta ? beta * out[oc] : 0));
                    } else {
                        out[off] = data_t<type_o>(alpha * inp[oc]
                                + (beta ? beta * out[off] : 0));
                    }
                }
            }
        });

        return success;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
    typename utils::enable_if<
        fmt_i == any && fmt_o == any && order_keep == fmt_order::any,
    spec::direct_copy>::type>
{
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d, const primitive_attr_t *attr) {
        /* FIXME: is the formula correct? */
        return input_d.similar_to(output_d, true, false, 0)
            && input_d.is_dense() && output_d.is_dense()
            && simple_attr_check(attr, false);
    }

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        assert(input_d.is_dense());

        input += input_d.blk_off(0);
        output += output_d.blk_off(0);

        const size_t nelems = input_d.nelems();

        constexpr int block_size = 16;
        const auto num_blocks = nelems / block_size;
        const auto rem_elems = nelems % block_size;

        parallel(0, [&](const int ithr, const int nthr) {
            size_t start{0}, end{0};
            balance211(num_blocks, nthr, ithr, start, end);
            start = start * block_size;
            end = end * block_size;
            round_mode_t rmode = pd->attr()->round_mode_;

            if (alpha == 1.0 && beta == 0.0) {
                PRAGMA_OMP_SIMD()
                for (size_t e = start; e < end; ++e) {
                    output[e] = qz_a1b0<data_t<type_i>, data_t<type_o>>()
                                (input[e], rmode);
                }
            } else if (alpha == 1.0) {
                PRAGMA_OMP_SIMD()
                for (size_t e = start; e < end; ++e) {
                    output[e] = qz_a1<data_t<type_i>, data_t<type_o>>()
                                (input[e], output[e], beta, rmode);
                }
            } else if (beta == 0.0) {
                PRAGMA_OMP_SIMD()
                for (size_t e = start; e < end; ++e) {
                    output[e] = qz_b0<data_t<type_i>, data_t<type_o>>()
                                (input[e], alpha, rmode);
                }
            } else {
                PRAGMA_OMP_SIMD()
                for (size_t e = start; e < end; ++e) {
                    output[e] = qz<data_t<type_i>, data_t<type_o>>()
                                (input[e], output[e], alpha, beta, rmode);
                }
            }

            if (rem_elems != 0 && ithr == nthr - 1){
                if (alpha == 1.0 && beta == 0.0) {
                    PRAGMA_OMP_SIMD()
                    for (size_t e = nelems - rem_elems; e < nelems; ++e) {
                        output[e] = qz_a1b0<data_t<type_i>,
                            data_t<type_o>>()(input[e], rmode);
                    }
                } else if (alpha == 1.0) {
                    PRAGMA_OMP_SIMD()
                    for (size_t e = nelems - rem_elems; e < nelems; ++e) {
                        output[e] = qz_a1<data_t<type_i>,
                            data_t<type_o>>()(input[e], output[e], beta, rmode);
                    }
                } else if (beta == 0.0) {
                    PRAGMA_OMP_SIMD()
                    for (size_t e = nelems - rem_elems; e < nelems; ++e) {
                        output[e] = qz_b0<data_t<type_i>,
                            data_t<type_o>>()(input[e], alpha, rmode);
                    }
                } else {
                    PRAGMA_OMP_SIMD()
                    for (size_t e = nelems - rem_elems; e < nelems; ++e) {
                        output[e] = qz<data_t<type_i>, data_t<type_o>>()
                                    (input[e], output[e], alpha, beta, rmode);
                   }
               }
            }
        });
        return success;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
    typename utils::enable_if<
        fmt_i == any && fmt_o == any && order_keep == fmt_order::any,
    spec::direct_copy_except_dim_0>::type>
{
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d, const primitive_attr_t *attr) {
        auto is_dense_no_0 = [](const memory_desc_wrapper &data_d) {
            return nelems_no_dim_0(data_d) == _size_no_dim_0(data_d);
        };
        /* FIXME: is the formula correct? */
        return input_d.similar_to(output_d, true, false, 1)
            && is_dense_no_0(input_d) && is_dense_no_0(output_d)
            && simple_attr_check(attr, false);
    }

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        input += input_d.blk_off(0);
        output += output_d.blk_off(0);

        const int N = input_d.dims()[0];
        const size_t is = input_d.blocking_desc().strides[0][0];
        const size_t os = output_d.blocking_desc().strides[0][0];
        const size_t nelems_no_d0 = nelems_no_dim_0(input_d);
        const size_t work_amount = N * nelems_no_d0;

        if (alpha == 1.0 && beta == 0.0) {
            parallel(0, [&](const int ithr, const int nthr) {
                size_t n{0}, dim1_s{0};
                size_t start{0}, end{0};
                balance211(work_amount, nthr, ithr, start, end);
                nd_iterator_init(start, n, N, dim1_s, nelems_no_d0);
                while(start < end) {
                    size_t work_rem = end - start;
                    size_t dim1_e =
                        dim1_s + work_rem > nelems_no_d0 ? nelems_no_d0
                        : dim1_s + work_rem;
                    PRAGMA_OMP_SIMD()
                    for (size_t e = dim1_s; e < dim1_e; ++e){
                        output[os * n + e] = data_t<type_o>(input[is * n + e]);
                    }
                    nd_iterator_jump(start, end, n, N, dim1_s, nelems_no_d0);
                }
            });
        } else {
            parallel(0, [&](const int ithr, const int nthr) {
                size_t n{0}, dim1_s{0};
                size_t start{0}, end{0};
                balance211(work_amount, nthr, ithr, start, end);
                nd_iterator_init(start, n, N, dim1_s, nelems_no_d0);
                while(start < end) {
                    size_t work_rem = end - start;
                    size_t dim1_e =
                        dim1_s + work_rem > nelems_no_d0 ? nelems_no_d0
                        : dim1_s + work_rem;
                    PRAGMA_OMP_SIMD()
                    for (size_t e = dim1_s; e < dim1_e; ++e){
                        output[os * n + e] = data_t<type_o>(
                                alpha * input[is * n + e]
                                + beta * output[os * n + e]);
                    }
                    nd_iterator_jump(start, end, n, N, dim1_s, nelems_no_d0);
                }
            });
        }

        return success;
    }

private:
    static size_t nelems_no_dim_0(const memory_desc_wrapper &data_d) {
        const int ndims = data_d.ndims();
        if (ndims <= 1) return 1;
        return utils::array_product(data_d.dims() + 1, data_d.ndims() - 1);
    }

    static size_t _size_no_dim_0(const memory_desc_wrapper &data_d) {
        size_t max_size = 0;
        auto &blk = data_d.blocking_desc();
        for (int d = 1; d < data_d.ndims(); ++d) {
            auto block = blk.block_dims[d];
            max_size = nstl::max(max_size,
                    size_t(size_t(blk.padding_dims[d] / block)
                        * blk.strides[0][d]));
            if (block > 1)
                max_size = nstl::max(max_size,
                        size_t(block * blk.strides[1][d]));
        }
        return max_size;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
    typename utils::enable_if<
        fmt_i == any && fmt_o == any && order_keep == fmt_order::any,
    spec::reference>::type>
{
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d, const primitive_attr_t *attr) {
        /* supported smask: 0x0...011..10...0,
         * i.e. 1 should be contiguous */
        int smask = attr ? attr->output_scales_.mask_ : 0;
        for (; smask > 0 && !(smask & 0x1); smask >>= 1);
        for (; smask > 0 && smask & 0x1; smask >>= 1);
        return true
            && input_d.is_blocking_desc()
            && output_d.is_blocking_desc()
            && smask == 0;
    }

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        const size_t nelems = input_d.nelems();

        int ndims_start = 0, ndims_mask = 0;
        int smask = pd->attr()->output_scales_.mask_;
        for (; smask > 0 && !(smask & 0x1); smask >>= 1) ++ndims_start;
        for (; smask > 0 && smask & 0x1; smask >>= 1) ++ndims_mask;
        assert(smask == 0);

        const ptrdiff_t D_start
            = utils::array_product(input_d.dims(), ndims_start);
        const ptrdiff_t D_mask
            = utils::array_product(input_d.dims() + ndims_start, ndims_mask);
        const ptrdiff_t D_rest = nelems / D_start / D_mask;

        const float *scales = pd->attr()->output_scales_.scales_;

        parallel_nd(D_start, D_mask, D_rest,
            [&](ptrdiff_t ds, ptrdiff_t dm, ptrdiff_t dr) {
            const float scale = scales[dm];

            const size_t e = (ds * D_mask + dm) * D_rest + dr;
            float i = (float)input[input_d.off_l(e)];
            auto &o = output[output_d.off_l(e)];

            i = scale * i + (beta ? beta * (float)o : 0);
            if (type_o != f32) {
                switch (pd->attr()->round_mode_) {
                case round_mode::down: i = floorf(i); break;
                case round_mode::nearest: i = nearbyintf(i); break;
                }
                o = saturate<data_t<type_o>>(i);
            } else {
                o = (data_t<type_o>)i;
            }
        });

        return success;
    }
};


/* high level class declaration */

template <SIMPLE_REORDER_TEMPL_DECL, typename spec = void>
struct simple_reorder_t: public cpu_primitive_t {
    struct pd_t: public cpu_reorder_pd_t {
        pd_t(const cpu_memory_pd_t *input_pd, const cpu_memory_pd_t *output_pd,
                const primitive_attr_t *attr)
            : cpu_reorder_pd_t(input_pd, output_pd, attr) {}

        DECLARE_COMMON_PD_T("simple:any", simple_reorder_t);

        static status_t create(reorder_pd_t **reorder_pd,
                const memory_pd_t *input_pd, const memory_pd_t *output_pd,
                const primitive_attr_t *attr) {
            assert(input_pd->engine()->kind() == engine_kind::cpu);
            assert(output_pd->engine()->kind() == engine_kind::cpu);
            bool args_ok = true
                && input_pd->desc()->data_type == type_i
                && output_pd->desc()->data_type == type_o
                && simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL, spec>::
                is_applicable(input_pd->desc(), output_pd->desc(), attr);
            if (!args_ok)
                return invalid_arguments;

            auto _pd = new pd_t((const cpu_memory_pd_t *)input_pd,
                    (const cpu_memory_pd_t *)output_pd, attr);
            if (_pd == nullptr) return out_of_memory;
            if (_pd->init() != success) { delete _pd; return unimplemented; }
            return safe_ptr_assign<reorder_pd_t>(*reorder_pd, _pd);
        }
    };

    simple_reorder_t(const pd_t *pd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd) {}

    virtual void execute(event_t *e) {
        auto input = reinterpret_cast<const data_t<type_i> *>(
                this->input_memory(0));
        auto output = reinterpret_cast<data_t<type_o> *>(this->memory());
        simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL, spec>::execute(
                &conf_, input, output);
        e->set_state(event_t::ready);
    }

private:
    pd_t conf_;
};

#undef SIMPLE_REORDER_TEMPL_DECL
#undef SIMPLE_REORDER_TEMPL_CALL

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
