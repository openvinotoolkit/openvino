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

#ifndef CPU_AARCH64_ACL_INNER_PRODUCT_HPP
#define CPU_AARCH64_ACL_INNER_PRODUCT_HPP

#include "cpu/cpu_inner_product_pd.hpp"

#include "cpu/aarch64/acl_inner_product_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct acl_ip_resource_t : public resource_t {
    acl_ip_resource_t() : acl_ip_obj_(utils::make_unique<acl_ip_obj_t>()) {}

    status_t configure(const acl_ip_conf_t &aip) {
        if (!acl_ip_obj_) return status::out_of_memory;

        // Init Compute Library tensors based on info from descriptor
        acl_ip_obj_->src_tensor.allocator()->init(aip.src_info);
        acl_ip_obj_->wei_tensor.allocator()->init(aip.wei_info);
        acl_ip_obj_->dst_tensor.allocator()->init(aip.dst_info);
        acl_ip_obj_->bia_tensor.allocator()->init(aip.bia_info);
        if (aip.with_sum) {
            acl_ip_obj_->dst_acc_tensor.allocator()->init(aip.dst_info);
        }

        // clang-format off
        acl_ip_obj_->fc.configure(
            &acl_ip_obj_->src_tensor,
            &acl_ip_obj_->wei_tensor,
            aip.with_bias ? &acl_ip_obj_->bia_tensor : nullptr,
            aip.with_sum ? &acl_ip_obj_->dst_acc_tensor : &acl_ip_obj_->dst_tensor,
            aip.fc_info);
        // clang-format on
        if (aip.with_sum) {
            acl_ip_obj_->add.configure(&acl_ip_obj_->dst_tensor,
                    &acl_ip_obj_->dst_acc_tensor, &acl_ip_obj_->dst_tensor,
                    arm_compute::ConvertPolicy::SATURATE);
            acl_ip_obj_->dst_acc_tensor.allocator()->allocate();
        }

        return status::success;
    }

    acl_ip_obj_t &get_acl_obj() const { return *acl_ip_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(acl_ip_resource_t);

private:
    std::unique_ptr<acl_ip_obj_t> acl_ip_obj_;
}; // acl_ip_resource_t

struct acl_inner_product_fwd_t : public primitive_t {
    struct pd_t : public cpu_inner_product_fwd_pd_t {
        using cpu_inner_product_fwd_pd_t::cpu_inner_product_fwd_pd_t;

        DECLARE_COMMON_PD_T("inner_product:acl", acl_inner_product_fwd_t);

        status_t init(engine_t *engine) {
            using namespace utils;

            const bool ok = is_fwd() && !has_zero_dim_memory()
                    && expect_data_types(data_type::f32, data_type::f32,
                            data_type::f32, data_type::f32, data_type::f32)
                    && attr()->has_default_values(
                            primitive_attr_t::skip_mask_t::post_ops,
                            data_type::f32)
                    && (set_default_params() == status::success)
                    && post_ops_ok();

            if (!ok) return status::unimplemented;

            auto conf_status = acl_inner_product_utils::init_conf_ip(aip_,
                    src_md_, weights_md_, dst_md_, bias_md_, *desc(), *attr());
            // conf_status here can be either status::success or status::unimplemented
            if (conf_status != status::success) return conf_status;

            acl_common_utils::acl_thread_bind();

            return status::success;
        }

        acl_ip_conf_t aip_;

    protected:
        bool post_ops_ok() const {
            auto const &po = attr()->post_ops_;
            // "true" here stands for eltwise.scale == 1.f check
            auto is_eltwise
                    = [&](int idx) { return po.entry_[idx].is_eltwise(true); };
            auto is_sum = [&](int idx) { return po.entry_[idx].is_sum(); };

            bool eltwise_ok = false;
            // Compute Library supports here only one eltwise post-op or sum
            if (po.len() == 1 && is_eltwise(0)) {
                const auto act_type = po.entry_[0].eltwise.alg;
                eltwise_ok = acl_common_utils::acl_act_ok(act_type);
            }

            return eltwise_ok || (po.len() == 1 && is_sum(0))
                    || (po.len() == 0);
        }
    }; // pd_t

    acl_inner_product_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        auto r = utils::make_unique<acl_ip_resource_t>();
        if (!r) return status::out_of_memory;

        // Configure the resource based on information from primitive descriptor
        auto st = r->configure(pd()->aip_);
        if (st == status::success) { mapper.add(this, std::move(r)); }

        return st;
    }

    using data_t = typename prec_traits<data_type::f32>::type;

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    //To guard the const execute_forward, the mutex must be 'mutable'
    mutable std::mutex mtx;
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
}; // acl_inner_product_fwd_t

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_ACL_INNER_PRODUCT_HPP
