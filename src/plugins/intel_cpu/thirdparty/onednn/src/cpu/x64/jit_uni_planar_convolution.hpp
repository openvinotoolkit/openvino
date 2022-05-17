/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
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

#ifndef CPU_X64_JIT_UNI_PLANAR_CONVOLUTION_HPP
#define CPU_X64_JIT_UNI_PLANAR_CONVOLUTION_HPP

#include "jit_primitive_conf.hpp"
#include "jit_uni_planar_conv_kernel_f32.hpp"

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"

#include "cpu/cpu_convolution_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <cpu_isa_t isa>
struct _jit_uni_planar_convolution_fwd_t: public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
             const typename pd_t::base_class *hint_fwd_pd)
                : cpu_convolution_fwd_pd_t(adesc, attr, hint_fwd_pd), jcp_() {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit_planar:", isa, ""),
                _jit_uni_planar_convolution_fwd_t<isa>);

        status_t init(engine_t *engine) {
            bool ok = true
                && is_fwd()
                && set_default_alg_kind(alg_kind::convolution_direct)
                && !this->has_zero_dim_memory()
                && utils::everyone_is(data_type::f32,
                        this->desc()->src_desc.data_type,
                        this->desc()->weights_desc.data_type,
                        this->desc()->dst_desc.data_type)
                && IMPLICATION(this->with_bias(), data_type::f32 == this->desc()->bias_desc.data_type)
                && attr()->has_default_values(primitive_attr_t::skip_mask_t::post_ops);
            if (!ok) return status::unimplemented;

            status_t sts = jit_uni_planar_conv_fwd_kernel_f32<isa>::init_conf(jcp_, *desc(), src_md_, weights_md_, dst_md_, bias_md_, *attr());

            return sts;
        }

        jit_conv_conf_t jcp_;
    };

    _jit_uni_planar_convolution_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    typedef typename prec_traits<data_type::f32>::type data_t;

    status_t init(engine_t *engine) override {
        CHECK(safe_ptr_assign(kernel_, new jit_uni_planar_conv_fwd_kernel_f32<isa>(pd()->jcp_, *pd()->attr())));
        return kernel_->create_kernel();
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        execute_forward(ctx);
        return status::success;
    }

private:
    void execute_forward(const exec_ctx_t &ctx) const;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<jit_uni_planar_conv_fwd_kernel_f32<isa>> kernel_;
};

using jit_avx512_common_planar_convolution_fwd_t = _jit_uni_planar_convolution_fwd_t<avx512_common>;
using jit_avx2_planar_convolution_fwd_t = _jit_uni_planar_convolution_fwd_t<avx2>;

}
}
}
}

#endif
