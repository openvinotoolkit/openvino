/*******************************************************************************
* Copyright 2016-2020 Intel Corporation
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

/* [todo] antonvor:
 * This file contains the old plugin behavior in order to fix performance
 * problems after upgrading to OneDNN v1.6. This kernel is executed only on
 * machines with avx2 instruction set support and in the case of a fused
 * convolution. Remove after problems are fixed.
*/

#ifndef CPU_X64_JIT_AVX2_1X1_CONVOLUTION_WITH_DW_CONV_HPP
#define CPU_X64_JIT_AVX2_1X1_CONVOLUTION_WITH_DW_CONV_HPP

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/primitive_hashing.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/dw_convolution_utils.hpp"
#include "cpu/platform.hpp"

#include "cpu/x64/cpu_reducer.hpp"
#include "cpu/x64/jit_avx2_1x1_conv_kernel_f32.hpp"
#include "cpu/x64/jit_uni_1x1_conv_utils.hpp"
#include "cpu/x64/jit_uni_dw_convolution.hpp"

#include "cpu/x64/jit_avx2_1x1_conv_kernel_f32_old.hpp"
#include "cpu/x64/jit_uni_dw_conv_row_f32.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct jit_avx2_1x1_convolution_with_dw_conv_fwd_t : public primitive_t {
    // TODO: (Roma) Code duplication duplication! Remove with templates
    //              (maybe...)!
    struct pd_t : public cpu_convolution_fwd_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
             const typename pd_t::base_class *hint_fwd_pd)
                : cpu_convolution_fwd_pd_t(adesc, attr, hint_fwd_pd)
                , jcp_(), jcp_dw_(), rtus_() {}

        pd_t(const pd_t &other) : cpu_convolution_fwd_pd_t(other) {
            if (copy(other) != status::success) is_initialized_ = false;
        }

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit_1x1_with_dw_conv:", avx2, ""),
                            jit_avx2_1x1_convolution_with_dw_conv_fwd_t);

        status_t init(engine_t *engine) {
            using namespace prop_kind;
            assert(engine->kind() == engine_kind::cpu);
            bool ok = true
                      && this->set_default_formats()
                      && utils::one_of(this->desc()->prop_kind, forward_training,
                                       forward_inference)
                      && utils::one_of(this->desc()->alg_kind,
                                       alg_kind::convolution_auto,
                                       alg_kind::convolution_direct)
                      && !this->has_zero_dim_memory()
                      && utils::everyone_is(data_type::f32,
                                            this->desc()->src_desc.data_type,
                                            this->desc()->weights_desc.data_type,
                                            this->desc()->dst_desc.data_type)
                      && IMPLICATION(this->with_bias(),
                                     data_type::f32 == this->desc()->bias_desc.data_type);
            if (!ok) return status::unimplemented;

            const convolution_desc_t *conv_d = this->desc();
            const memory_desc_t *src_d = src_md();
            rtus_prepare(this, conv_d, src_d, dst_md(), weights_md());

            status_t sts_1x1 = jit_avx2_1x1_conv_kernel_f32_old::init_conf(
                    jcp_, *conv_d, *src_d, *weights_md(), *dst_md(), *attr());
            if (sts_1x1 != status::success) return sts_1x1;

            if (jcp_.with_dw_conv) {
                status_t sts_dw = jit_uni_dw_conv_row_f32<avx2>::init_conf(jcp_, jcp_dw_, *this->attr());
                if (sts_dw != status::success) return sts_dw;
            } else {
                return status::unimplemented;
            }

            auto scratchpad = scratchpad_registry().registrar();
            jit_avx2_1x1_conv_kernel_f32_old::init_scratchpad(scratchpad, jcp_, jcp_dw_);

            rtus_prepare_space_info(this, scratchpad, dnnl_get_max_threads());

            return status::success;
        }

        const memory_desc_t *dst_md(int index = 0) const override {
            return &dst_md_;
        }

        const memory_desc_t *arg_md(int index = 0) const override {
            return convolution_fwd_pd_t::arg_md(index);
        }

        arg_usage_t arg_usage(int arg) const override {
            return convolution_fwd_pd_t::arg_usage(arg);
        }

        jit_1x1_conv_conf_t jcp_;
        jit_conv_conf_t jcp_dw_;
        reduce_to_unit_stride_t rtus_;

    protected:
        bool set_default_formats() {
            using namespace format_tag;

            auto dat_tag = utils::pick(ndims() - 3, nCw8c, nChw8c, nCdhw8c);
            auto wei_tag = with_groups()
                           ? utils::pick(ndims() - 3, gOIw8i8o, gOIhw8i8o, gOIdhw8i8o)
                           : utils::pick(ndims() - 3, OIw8i8o, OIhw8i8o, OIdhw8i8o);

            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }

        status_t copy(const pd_t &other) {
            jcp_ = other.jcp_;
            rtus_ = other.rtus_;
            jcp_dw_ = other.jcp_dw_;

            return status::success;
        }
    };

    template <cpu_isa_t isa, typename conv_t>
    friend status_t init_rtus_driver(conv_t *self);

    jit_avx2_1x1_convolution_with_dw_conv_fwd_t(const pd_t *apd) : primitive_t(apd),
        kernel_old_(nullptr), rtus_driver_(nullptr) {
        kernel_old_ = new jit_avx2_1x1_conv_kernel_f32_old(pd()->jcp_, pd()->jcp_dw_, *pd()->attr());
        init_rtus_driver<avx2>(this);

        if (pd()->jcp_.with_dw_conv) {
            kernel_dw_ = new jit_uni_dw_conv_row_f32<avx2>(pd()->jcp_dw_, *pd()->attr(), pd()->jcp_dw_.ch_block);
        }
    }

    status_t init(engine_t *engine) override {
        CHECK(kernel_old_->create_kernel());
        if (kernel_dw_)
            CHECK(kernel_dw_->create_kernel());
        return status::success;
    }

    ~jit_avx2_1x1_convolution_with_dw_conv_fwd_t() {
        delete kernel_old_;
        delete rtus_driver_;
        delete kernel_dw_;
    }

    typedef typename prec_traits<data_type::f32>::type data_t;

    status_t execute(const exec_ctx_t &ctx) const override {
        execute_forward(ctx);
        return status::success;
    }

private:
    void execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    jit_avx2_1x1_conv_kernel_f32_old *kernel_old_;
    jit_uni_dw_conv_row_f32<avx2> *kernel_dw_;
    rtus_driver_t<avx2> *rtus_driver_;

};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
