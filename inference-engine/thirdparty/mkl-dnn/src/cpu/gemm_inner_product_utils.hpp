/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef CPU_GEMM_INNER_PRODUCT_UTILS_HPP
#define CPU_GEMM_INNER_PRODUCT_UTILS_HPP

#include "c_types_map.hpp"
#include "cpu_inner_product_pd.hpp"
#include "cpu_engine.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"
#include "jit_generator.hpp"
#include "jit_uni_eltwise.hpp"
#include "ref_eltwise.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

namespace inner_product_utils {

template <impl::data_type_t acc_type, impl::data_type_t dst_type>
class pp_kernel_t : jit_generator
{
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(gemm_x8s8s32x_inner_product_fwd_t::pp_kernel);
    pp_kernel_t(const cpu_inner_product_fwd_pd_t *pd);
    ~pp_kernel_t() {
        if (do_eltwise_) {
            delete eltwise_injector_;
            delete ref_eltwise_;
        }
    }

    typedef typename prec_traits<acc_type>::type acc_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;

    void operator()(dst_data_t *dst, const acc_data_t *acc, const char *bias,
            const float *scales, size_t start, size_t end);

private:
    void generate();

    struct ker_args {
        dst_data_t *dst;
        const acc_data_t *acc;
        const char *bias;
        const float *scales;
        float nslope;
        size_t len;
        size_t oc_offset;
    };
    void (*ker_)(const ker_args *args);
    jit_uni_eltwise_injector_f32<avx512_common> *eltwise_injector_;
    ref_eltwise_scalar_fwd_t *ref_eltwise_;

    size_t OC_;
    data_type_t bias_data_type_;
    size_t bias_data_type_size_;
    size_t scale_idx_mult_;
    round_mode_t rmode_;
    bool do_bias_;
    bool do_eltwise_;
    post_ops_t::entry_t::eltwise_t eltwise_;
};

}

}
}
}

#endif
