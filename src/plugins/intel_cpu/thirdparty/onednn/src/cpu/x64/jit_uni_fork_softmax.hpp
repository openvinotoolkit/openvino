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

#ifndef CPU_X64_JIT_UNI_FORK_SOFTMAX_HPP
#define CPU_X64_JIT_UNI_FORK_SOFTMAX_HPP

#include <assert.h>
#include <memory>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_softmax_pd.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_uni_fork_softmax_kernel_f32.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <cpu_isa_t isa>
struct jit_uni_fork_softmax_fwd_t : public primitive_t {
    struct pd_t : public cpu_softmax_fwd_pd_t {
        using cpu_softmax_fwd_pd_t::cpu_softmax_fwd_pd_t;

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit:", isa, ""),
                jit_uni_fork_softmax_fwd_t<isa>);

        status_t init(engine_t *engine) {
            const memory_desc_wrapper src_d(src_md());
            const memory_desc_wrapper dst_d(dst_md());
            auto data_type = src_d.data_type();

            auto ndims = desc_.data_desc.ndims;
            auto dims = desc_.data_desc.dims;
            auto axis = desc_.softmax_axis;

            size_t inner_size = utils::array_product(dims + axis + 1, ndims - axis - 1);

            format_tag_t dat_tag = utils::pick(ndims - 3, format_tag::ncw, format_tag::nchw, format_tag::ncdhw);

            // TODO: disabled because of failed test (case: for axis == 0, batch == 2). Needs to be debugged.
            if (ndims == 3)
                return status::unimplemented;

            using namespace data_type;
            bool ok = src_d == dst_d && mayiuse(isa) && is_fwd()
                      && !has_zero_dim_memory()
                      && utils::one_of(data_type, f32, bf16)
                      && attr()->has_default_values()
                      && src_d.is_dense(true)
                      && src_d.matches_one_of_tag(dat_tag) == dat_tag
                      && inner_size > 1;
            if (!ok) return status::unimplemented;

            return jit_uni_fork_softmax_kernel_f32<isa>::init_conf(jpp_, desc_, src_md(), dst_md());
        }
        jit_softmax_conf_t jpp_;
    };

    jit_uni_fork_softmax_fwd_t(const pd_t *apd);

    status_t init(engine_t *engine) override {
        CHECK(safe_ptr_assign(kernel_, new jit_uni_fork_softmax_kernel_f32<isa>(pd()->jpp_)));
        return kernel_->create_kernel();
    }

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::unique_ptr<jit_uni_fork_softmax_kernel_f32<isa>> kernel_;
};

}
}
}
}

#endif
