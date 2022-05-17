/*******************************************************************************
* Copyright 2021 Arm Ltd. and affiliates
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

#ifndef CPU_AARCH64_ACL_ELTWISE_HPP
#define CPU_AARCH64_ACL_ELTWISE_HPP

#include "cpu/cpu_eltwise_pd.hpp"

#include "cpu/aarch64/acl_eltwise_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct acl_eltwise_resource_t : public resource_t {
    acl_eltwise_resource_t()
        : acl_eltwise_obj_(utils::make_unique<acl_eltwise_obj_t>()) {}

    status_t configure(const acl_eltwise_conf_t &aep) {
        if (!acl_eltwise_obj_) return status::out_of_memory;

        // Init Compute Library tensors based on info from descriptor
        acl_eltwise_obj_->src_tensor.allocator()->init(aep.src_info);
        acl_eltwise_obj_->dst_tensor.allocator()->init(aep.dst_info);

        // clang-format off
        acl_eltwise_obj_->act.configure(
            &acl_eltwise_obj_->src_tensor,
            &acl_eltwise_obj_->dst_tensor,
            aep.act_info);
        // clang-format on

        return status::success;
    }

    acl_eltwise_obj_t &get_acl_obj() const { return *acl_eltwise_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(acl_eltwise_resource_t);

private:
    std::unique_ptr<acl_eltwise_obj_t> acl_eltwise_obj_;
}; // acl_eltwise_resource_t

template <data_type_t data_type>
struct acl_eltwise_fwd_t : public primitive_t {
    struct pd_t : public cpu_eltwise_fwd_pd_t {
        using cpu_eltwise_fwd_pd_t::cpu_eltwise_fwd_pd_t;
        pd_t(const eltwise_desc_t *adesc, const primitive_attr_t *attr,
                const eltwise_fwd_pd_t *hint_fwd_pd)
            : cpu_eltwise_fwd_pd_t(adesc, attr, hint_fwd_pd), aep_() {}

        DECLARE_COMMON_PD_T("eltwise:acl", acl_eltwise_fwd_t);

        status_t init(engine_t *engine) {
            using namespace utils;
            const auto &po = attr()->post_ops_;

            bool ok = is_fwd() && data_type == desc()->data_desc.data_type
                    && !has_zero_dim_memory() && attr()->has_default_values()
                    && po.len() == 0;
            if (!ok) return status::unimplemented;

            auto conf_status = acl_eltwise_utils::init_conf_eltwise(
                    aep_, data_md_, *desc(), *attr());
            if (conf_status != status::success) return status::unimplemented;

            acl_common_utils::acl_thread_bind();

            return status::success;
        }

        acl_eltwise_conf_t aep_;
    };

    acl_eltwise_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    using data_t = typename prec_traits<data_type>::type;

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        auto r = utils::make_unique<acl_eltwise_resource_t>();
        if (!r) return status::out_of_memory;

        // Configure the resource based on information from primitive descriptor
        auto st = r->configure(pd()->aep_);
        if (st == status::success) { mapper.add(this, std::move(r)); }

        return st;
    }

private:
    // execute_forward has to be const thus mutability of mtx
    mutable std::mutex mtx;
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
}; // acl_eltwise_fwd_t

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_ACL_ELTWISE_HPP
