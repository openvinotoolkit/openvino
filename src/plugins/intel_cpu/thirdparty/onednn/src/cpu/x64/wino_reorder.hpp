/*******************************************************************************
* Copyright 2017-2021 Intel Corporation
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

#ifndef CPU_X64_WINO_REORDER_HPP
#define CPU_X64_WINO_REORDER_HPP

#include "common/dnnl_thread.hpp"
#include "common/primitive.hpp"
#include "common/primitive_desc.hpp"

#include "cpu/reorder/cpu_reorder_pd.hpp"
#include "cpu/simple_q10n.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <data_type_t type_i, data_type_t type_o>
struct wino_reorder_t : public primitive_t {
    struct pd_t : public cpu_reorder_pd_t {
        using cpu_reorder_pd_t::cpu_reorder_pd_t;

        DECLARE_COMMON_PD_T("wino_reorder", wino_reorder_t);

        status_t init(
                engine_t *engine, engine_t *src_engine, engine_t *dst_engine) {
            status_t status
                    = cpu_reorder_pd_t::init(engine, src_engine, dst_engine);
            if (status != status::success) return status;

            bool ok = attr()->has_default_values(
                    primitive_attr_t::skip_mask_t::oscale
                    | primitive_attr_t::skip_mask_t::post_ops);
            if (!ok) return status::unimplemented;

            init_scratchpad();

            return status::success;
        }

    private:
        static status_t create(reorder_pd_t **reorder_pd, engine_t *engine,
                const primitive_attr_t *attr, engine_t *src_engine,
                const memory_desc_t *src_md, engine_t *dst_engine,
                const memory_desc_t *dst_md) {
            const memory_desc_wrapper id(src_md), od(dst_md);
            bool args_ok = true && id.data_type() == type_i
                    && od.data_type() == type_o
                    && od.format_kind() == format_kind::wino
                    && utils::one_of(od.wino_desc().wino_format,
                            dnnl_wino_wei_aaOIoi, dnnl_wino_wei_aaOio,
                            dnnl_wino_wei_aaOBiOo, dnnl_wino_wei_OBaaIBOIio)
                    && (id.matches_tag(utils::pick(id.ndims() - 4,
                                format_tag::oihw, format_tag::goihw))
                            || id.matches_tag(utils::pick(id.ndims() - 4,
                                    format_tag::hwio, format_tag::hwigo)));
            if (!args_ok) return status::invalid_arguments;

            auto _pd = new pd_t(attr, src_engine->kind(), src_md,
                    dst_engine->kind(), dst_md);
            if (_pd == nullptr) return status::out_of_memory;
            if (_pd->init(engine, src_engine, dst_engine) != status::success) {
                delete _pd;
                return status::unimplemented;
            }
            _pd->init_scratchpad_md();
            return safe_ptr_assign(*reorder_pd, _pd);
        }

        void init_scratchpad() {
            const auto &wino_desc = memory_desc_wrapper(dst_md()).wino_desc();
            const int nb_oc = wino_desc.oc / wino_desc.oc_block;
            const int work_amount = nb_oc * wino_desc.ic;
            const int thread_count
                    = nstl::min(dnnl_get_max_threads(), work_amount);
            const size_t transform_space_size = static_cast<size_t>(wino_desc.r)
                    * wino_desc.alpha * wino_desc.oc_block * thread_count;
            const size_t plain_size = static_cast<size_t>(wino_desc.alpha)
                    * wino_desc.alpha * wino_desc.oc * wino_desc.ic;

            using namespace memory_tracking::names;
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.template book<in_data_t>(
                    key_reorder_wino_transform_space, transform_space_size);
            scratchpad.template book<out_data_t>(
                    key_reorder_wino_plain, plain_size);
        }
        friend dnnl::impl::impl_list_item_t;
    };

    wino_reorder_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        const memory_desc_wrapper src_d(pd()->src_md());
        const memory_desc_wrapper dst_d(pd()->dst_md());

        r_ = dst_d.wino_desc().r;
        w_alpha_ = dst_d.wino_desc().alpha;
        wino_format_ = dst_d.wino_desc().wino_format;

        const auto &in_dims = src_d.dims();
        int groups;
        int groups_offset;
        if (src_d.ndims() == 5) {
            groups = in_dims[0];
            groups_offset = 1;
        } else {
            groups = 1;
            groups_offset = 0;
        }
        assert(groups == 1); // groups are not supported now
        MAYBE_UNUSED(groups);

        or_oc_ = in_dims[0 + groups_offset];
        or_ic_ = in_dims[1 + groups_offset];
        kh_ = in_dims[2 + groups_offset];
        kw_ = in_dims[3 + groups_offset];

        oc_ = dst_d.wino_desc().oc;
        ic_ = dst_d.wino_desc().ic;
        oc_block_ = dst_d.wino_desc().oc_block;
        ic_block_ = dst_d.wino_desc().ic_block;
        assert(oc_ % oc_block_ == 0 && ic_ % ic_block_ == 0);
        nb_oc_ = oc_ / oc_block_;
        nb_ic_ = ic_ / ic_block_;
        ic2_block_ = 1;
        if (wino_format_ == dnnl_wino_wei_OBaaIBOIio)
            ic2_block_ = dst_d.wino_desc().ic2_block;
        oc2_block_ = dst_d.wino_desc().oc2_block;
        assert(nb_ic_ % ic2_block_ == 0 && nb_oc_ % oc2_block_ == 0);

        adj_scale_ = dst_d.wino_desc().adj_scale;

        size_wino_wei_ = w_alpha_ * w_alpha_ * oc_ * ic_;
        work_amount_ = ic_ * nb_oc_;
        thread_count_ = nstl::min(dnnl_get_max_threads(), work_amount_);
        size_wspace_thr_ = r_ * w_alpha_ * oc_block_;

        return status::success;
    }

private:
    typedef typename prec_traits<type_i>::type in_data_t;
    typedef typename prec_traits<type_o>::type out_data_t;
    const int unsign_val_in_wino_domain_ = 5;

    void transform(out_data_t *__restrict tmp_wei,
            const in_data_t *__restrict input,
            in_data_t *__restrict wspace) const {
        const memory_desc_wrapper src_d(pd()->src_md());

        const int smask = pd()->attr()->output_scales_.mask_;
        const int ndims_mask = math::ilog2q(smask + 1);
        const size_t D_mask = utils::array_product(src_d.dims(), ndims_mask);
        const float *__restrict scales = pd()->attr()->output_scales_.scales_;
        assert(D_mask == 1 || D_mask == static_cast<size_t>(oc_));

        /* transform weights to winograd domain */
        const float G_2x2_3x3[4][3] = {{1.0, 0.0, 0.0}, {0.5, 0.5, 0.5},
                {0.5, -0.5, 0.5}, {0.0, 0.0, 1.0}};

        const float G_4x4_3x3[6][3] = {{1.13777777777778f, 0.f, 0.f},
                {-0.688403361344538f, -0.430252100840336f, -0.26890756302521f},
                {-0.688403361344538f, 0.430252100840336f, -0.26890756302521f},
                {0.119514472455649f, 0.179271708683473f, 0.26890756302521f},
                {0.119514472455649f, -0.179271708683473f, 0.26890756302521f},
                {0.f, 0.f, 1.f}};

        float *__restrict g;
        if (utils::one_of(wino_format_, dnnl_wino_wei_aaOIoi,
                    dnnl_wino_wei_aaOio, dnnl_wino_wei_aaOBiOo))
            g = (float *)G_2x2_3x3;
        else if (wino_format_ == dnnl_wino_wei_OBaaIBOIio)
            g = (float *)G_4x4_3x3;
        else {
            assert(!"Unknown winograd weights target layout");
            return;
        }

        const bool has_oihw_format = false
                || src_d.matches_tag(format_tag::oihw)
                || src_d.matches_tag(format_tag::goihw);

        const int Z = oc_ * ic_;
        const int or_ioc_ = or_ic_ * or_oc_;
        assert(r_ == kh_ && r_ == kw_);

        parallel_nd_ext(
                0, ic_, nb_oc_, [&](int ithr, int nthr, dim_t iic, dim_t ob) {
                    if (ithr >= work_amount_) return;

                    const in_data_t *__restrict _inp = has_oihw_format
                            ? input
                                    + (ob * oc_block_ * or_ic_ + iic) * kh_
                                            * kw_
                            : input + iic * or_oc_ + ob * oc_block_;
                    out_data_t *__restrict _out
                            = tmp_wei + (iic * nb_oc_ + ob) * oc_block_;

                    in_data_t *__restrict wspace_thr
                            = wspace + ithr * size_wspace_thr_;

                    std::memset(wspace_thr, 0.f,
                            size_wspace_thr_ * sizeof(in_data_t));

                    if (has_oihw_format) {
                        for_(int ih = 0; ih < r_; ++ih)
                        for_(int j = 0; j < w_alpha_; ++j)
                        for (int ioc = 0; ioc < oc_block_; ++ioc) {
                            PRAGMA_OMP_SIMD()
                            for (int iw = 0; iw < r_; ++iw) {
                                const int inp_oc = ob * oc_block_ + ioc;
                                const int inp_ic = iic;
                                in_data_t inp_v
                                        = (inp_ic < or_ic_ && inp_oc < or_oc_)
                                        ? _inp[ioc * or_ic_ * kh_ * kw_
                                                + ih * kw_ + iw]
                                        : 0.f;
                                wspace_thr[(ih * w_alpha_ + j) * oc_block_
                                        + ioc]
                                        += inp_v * g[j * r_ + iw];
                            }
                        }
                    } else { // hwio format case
                        for_(int ih = 0; ih < r_; ++ih)
                        for_(int j = 0; j < w_alpha_; ++j)
                        for (int iw = 0; iw < kw_; ++iw) {
                            const float g_multiplier = g[j * r_ + iw];
                            const in_data_t *__restrict inp_base
                                    = _inp + or_ioc_ * (iw + ih * kw_);
                            in_data_t *__restrict wspace_base = wspace_thr
                                    + (ih * w_alpha_ + j) * oc_block_;

                            PRAGMA_OMP_SIMD()
                            for (int ioc = 0; ioc < oc_block_; ++ioc) {
                                const int inp_oc = ob * oc_block_ + ioc;
                                const int inp_ic = iic;
                                in_data_t inp_v
                                        = (inp_ic < or_ic_ && inp_oc < or_oc_)
                                        ? inp_base[ioc]
                                        : 0.f;

                                wspace_base[ioc] += inp_v * g_multiplier;
                            }
                        }
                    }

                    for_(int i = 0; i < w_alpha_; ++i)
                    for_(int j = 0; j < w_alpha_; ++j)
                    for (int ioc = 0; ioc < oc_block_; ++ioc) {
                        float res = 0;
                        PRAGMA_OMP_SIMD(reduction(+ : res))
                        for (int k = 0; k < r_; ++k)
                            res += g[i * r_ + k]
                                    * wspace_thr[(k * w_alpha_ + j) * oc_block_
                                            + ioc];
                        if (type_o == data_type::s8) {
                            const float scale = (D_mask == 1)
                                    ? scales[0]
                                    : scales[ob * oc_block_ + ioc];
                            _out[(i * w_alpha_ + j) * Z + ioc]
                                    = qz_b0<in_data_t, out_data_t>()(
                                            static_cast<in_data_t>(res),
                                            scale * adj_scale_);
                        } else {
                            _out[(i * w_alpha_ + j) * Z + ioc]
                                    = static_cast<out_data_t>(res);
                        }
                    }
                });
    }

    void reorder_to_aaOIoi(out_data_t *__restrict output,
            const out_data_t *__restrict tmp_wei) const {
        int32_t *__restrict dst_bias = nullptr;
        if (type_o == data_type::s8) {
            const auto bias_shift = sizeof(out_data_t) * size_wino_wei_;
            const size_t bias_size = w_alpha_ * w_alpha_ * oc_;

            dst_bias = reinterpret_cast<int32_t *>(output + bias_shift);
            utils::array_set(dst_bias, 0, bias_size);
        }
        int index = 0;
        for_(int u_h = 0; u_h < w_alpha_; u_h++)
        for (int u_w = 0; u_w < w_alpha_; u_w++) {
            for_nd(0, 1, nb_oc_, oc_block_, [&](dim_t ob, dim_t o) {
                const int u_h_shift = u_h * w_alpha_ * ic_ * oc_;
                const int u_w_shift = u_w * ic_ * oc_;
                const int u_h_shift_b = u_h * w_alpha_ * oc_;
                const int u_w_shift_b = u_w * oc_;
                const int oc_block_shift = ob * oc_block_ * ic_ + o * ic_block_;
                for_(int ib = 0; ib < nb_ic_; ib++)
                for (int i = 0; i < ic_block_; i++) {
                    const int _i = ib * ic_block_;
                    const int _o = ob * oc_block_;
                    const int ic_shift = (_i + i) * oc_;
                    const int oc_shift = (_o + o);
                    const int ic_block_shift = ib * oc_block_ * ic_block_ + i;
                    const int src_offset
                            = u_h_shift + u_w_shift + ic_shift + oc_shift;
                    const int dst_offset = u_h_shift + u_w_shift
                            + oc_block_shift + ic_block_shift;

                    output[dst_offset] = tmp_wei[src_offset];
                    if (type_o == data_type::s8) {
                        const int bias_offset
                                = u_h_shift_b + u_w_shift_b + oc_shift;
                        if (index != unsign_val_in_wino_domain_)
                            dst_bias[bias_offset] -= (128
                                    * static_cast<int32_t>(output[dst_offset]));
                        else
                            dst_bias[bias_offset] = 0;
                    }
                }
            });
            index++;
        }
    }

    void reorder_to_aaOio(out_data_t *__restrict output,
            const out_data_t *__restrict tmp_wei) const {
        parallel_nd(w_alpha_, w_alpha_, nb_oc_,
                [&](dim_t u_h, dim_t u_w, dim_t ob) {
                    for_(int ib = 0; ib < nb_ic_; ib++)
                    for_(int i = 0; i < ic_block_; i++)
                    for (int o = 0; o < oc_block_; o++) {
                        const int src_offset = u_h * w_alpha_ * ic_ * oc_
                                + u_w * ic_ * oc_ + (ib * ic_block_ + i) * oc_
                                + (ob * oc_block_ + o);

                        const int dst_offset = u_h * w_alpha_ * nb_oc_ * nb_ic_
                                        * ic_block_ * oc_block_
                                + u_w * nb_oc_ * nb_ic_ * ic_block_ * oc_block_
                                + ob * nb_ic_ * ic_block_ * oc_block_
                                + ib * ic_block_ * oc_block_ + i * oc_block_
                                + o;
                        output[dst_offset] = tmp_wei[src_offset];
                    }
                });
    }

    void reorder_to_aaOBiOo(out_data_t *__restrict output,
            const out_data_t *__restrict tmp_wei) const {
        const int oc_chunks = nb_oc_ / oc2_block_;
        parallel_nd(w_alpha_, w_alpha_, oc_chunks,
                [&](dim_t u_h, dim_t u_w, dim_t occ) {
                    for (int ib = 0; ib < nb_ic_; ib++) {
                        out_data_t *__restrict wei_ptr = output
                                + (((u_h * w_alpha_ + u_w) * oc_chunks + occ)
                                                  * nb_ic_
                                          + ib)
                                        * oc2_block_ * ic_block_ * oc_block_;
                        int wei_offset = 0;
                        for_(int i = 0; i < ic_block_; i++)
                        for (int ob2 = 0; ob2 < oc2_block_; ob2++) {
                            for (int o = 0; o < oc_block_; o++) {
                                const int icp = ib * ic_block_ + i;
                                const int ocp = occ * oc2_block_ * oc_block_
                                        + ob2 * oc_block_ + o;

                                const int src_offset
                                        = u_h * w_alpha_ * ic_ * oc_
                                        + u_w * ic_ * oc_ + icp * oc_ + ocp;
                                wei_ptr[wei_offset + o] = tmp_wei[src_offset];
                            }
                            wei_offset += oc_block_;
                        }
                    }
                });
    }

    void reorder_to_OBaaIBOIio(out_data_t *__restrict output,
            const out_data_t *__restrict tmp_wei) const {
        const int ic_chunks = nb_ic_ / ic2_block_;
        const int oc_chunks = nb_oc_ / oc2_block_;
        parallel_nd(oc_chunks, w_alpha_, w_alpha_,
                [&](dim_t occ, dim_t u_h, dim_t u_w) {
                    for_(int icc = 0; icc < ic_chunks; icc++)
                    for (int ob = 0; ob < oc2_block_; ob++) {
                        const int ocp = (occ * oc2_block_ + ob) * oc_block_;
                        for_(int ib = 0; ib < ic2_block_; ib++)
                        for (int i = 0; i < ic_block_; i++) {
                            const int icp
                                    = (icc * ic2_block_ + ib) * ic_block_ + i;

                            const int src_offset = u_h * w_alpha_ * ic_ * oc_
                                    + u_w * ic_ * oc_ + icp * oc_ + ocp;
                            const int wei_offset
                                    = ((((((occ * w_alpha_ + u_h) * w_alpha_
                                                  + u_w) * ic_chunks
                                                 + icc) * oc2_block_
                                                + ob) * ic2_block_
                                               + ib) * ic_block_
                                              + i)
                                    * oc_block_;
                            for (int o = 0; o < oc_block_; o++)
                                output[wei_offset + o]
                                        = tmp_wei[src_offset + o];
                        }
                    }
                });
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        auto input = CTX_IN_MEM(const in_data_t *, DNNL_ARG_FROM);
        auto output = CTX_OUT_MEM(out_data_t *, DNNL_ARG_TO);

        auto wspace = (in_data_t * __restrict) ctx.get_scratchpad_grantor()
                              .template get<void>(memory_tracking::names::
                                              key_reorder_wino_transform_space);
        auto tmp_wei = (out_data_t * __restrict) ctx.get_scratchpad_grantor()
                               .template get<void>(memory_tracking::names::
                                               key_reorder_wino_plain);

        transform(tmp_wei, input, wspace);

        /* reorder to winograd domain */
        switch (wino_format_) {
            case dnnl_wino_wei_aaOIoi:
                reorder_to_aaOIoi(output, tmp_wei);
                break;
            case dnnl_wino_wei_aaOio: reorder_to_aaOio(output, tmp_wei); break;
            case dnnl_wino_wei_aaOBiOo:
                reorder_to_aaOBiOo(output, tmp_wei);
                break;
            case dnnl_wino_wei_OBaaIBOIio:
                reorder_to_OBaaIBOIio(output, tmp_wei);
                break;
            default: assert(!"Unknown wino format"); break;
        }

        return status::success;
    }

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    int r_, w_alpha_;
    dim_t ic_, oc_, or_ic_, or_oc_, kh_, kw_;
    dim_t oc_block_, ic_block_, oc2_block_, ic2_block_;
    float adj_scale_;
    dim_t nb_oc_, nb_ic_;
    dnnl_wino_memory_format_t wino_format_;
    int size_wino_wei_;
    int size_wspace_thr_;
    int thread_count_;
    int work_amount_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
