/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
* Copyright 2020 Codeplay Software Limited
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

#ifndef GPU_NVIDIA_CUDNN_POOLING_HPP
#define GPU_NVIDIA_CUDNN_POOLING_HPP

#include "common/c_types_map.hpp"
#include "common/pooling_pd.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "gpu/nvidia/cudnn_pooling_impl.hpp"
#include "gpu/nvidia/sycl_cuda_engine.hpp"
#include "gpu/nvidia/sycl_cuda_stream.hpp"
#include "gpu/nvidia/sycl_cuda_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

struct cudnn_pooling_common_t {
    template <typename pd_t>
    void init_ws(const pd_t *pd, memory_desc_t &ws_md) {
        bool is_fwd = pd->is_fwd();
        memory_desc_wrapper src_wrap(is_fwd ? pd->src_md() : pd->diff_src_md());
        memory_desc_wrapper dst_wrap(is_fwd ? pd->dst_md() : pd->diff_dst_md());

        const auto src_size = src_wrap.nelems();
        const auto dst_size = dst_wrap.nelems();
        const dims_t ws_size = {(dim_t)(src_size + dst_size)};

        dnnl_memory_desc_init_by_tag(
                &ws_md, 1, ws_size, src_wrap.data_type(), format_tag::x);
    }

    status_t init_mem_by_tag(format_tag_t tag, memory_desc_t &md) {
        if (tag == format_tag::undef) { return status::unimplemented; }
        CHECK(memory_desc_init_by_tag(md, tag));
        return status::success;
    }

    format_tag_t get_tag(const memory_desc_t &md) const {
        using namespace format_tag;
        auto tag = memory_desc_matches_one_of_tag(md, ab, abc, abcd,
                abcde, // NCHW derivatives
                ba, bca, bcda, bcdea, cba, cdba,
                cdeba, // IO and spatial derivatives
                acb, acdb, acdeb, // NHWC derivatives
                aBcd16b, aBcde16b, aBcd8b, aBcde8b, aBcd4b,
                aBcde4b); // blocked layouts
        return tag;
    }
};

struct cudnn_pooling_fwd_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public pooling_fwd_pd_t, public cudnn_pooling_common_t {
        using pooling_fwd_pd_t::pooling_fwd_pd_t;

        DECLARE_COMMON_PD_T("cuda:cudnn:any", cudnn_pooling_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using namespace prop_kind;
            using namespace alg_kind;
            using namespace format_tag;

            assert(engine->kind() == engine_kind::gpu);
            auto src_dt = src_md()->data_type;

            bool ok = true && is_fwd();
            ok = ok && set_default_params() == status::success;
            ok = ok
                    && utils::one_of(desc()->prop_kind, forward_training,
                            forward_inference);
            ok = ok
                    && utils::one_of(desc()->alg_kind, pooling_max,
                            pooling_avg_include_padding,
                            pooling_avg_exclude_padding);
            ok = ok && utils::one_of(src_dt, s8, f16, f32);
            ok = ok
                    && IMPLICATION(utils::one_of(src_dt, f16),
                            desc()->prop_kind == forward_inference);
            ok = ok
                    && IMPLICATION(
                            src_dt == s8, desc()->accum_data_type == s32);
            ok = ok && attr()->has_default_values();
            ok = ok && blocking_ok();
            if (!ok) return status::unimplemented;

            bool is_training = desc_.prop_kind == forward_training;
            if (is_training) init_ws(this, ws_md_);

            if (has_zero_dim_memory()) return status::success;

            pooling_impl_.reset(new cudnn_pooling_fwd_impl_t());
            return pooling_impl_->init(this);
        }

        bool blocking_ok() const {
            if (!utils::one_of(src_md()->data_type, data_type::s8)
                    && src_md()->format_desc.blocking.inner_nblks > 0)
                return false;

            if (src_md()->format_desc.blocking.inner_nblks > 1) return false;

            if (utils::one_of(src_md()->data_type, data_type::s8)
                    && src_md()->format_desc.blocking.inner_nblks == 1) {
                return memory_desc_matches_nchw_vect_c(src_md())
                        && memory_desc_matches_nchw_vect_c(dst_md());
            }

            return true;
        }

        std::shared_ptr<cudnn_pooling_impl_base_t> pooling_impl_;
    };

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

struct cudnn_pooling_bwd_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public pooling_bwd_pd_t, public cudnn_pooling_common_t {
        using pooling_bwd_pd_t::pooling_bwd_pd_t;

        DECLARE_COMMON_PD_T("cuda:cudnn:any", cudnn_pooling_bwd_t);

        status_t init(engine_t *engine) {
            using namespace prop_kind;
            using namespace alg_kind;
            using namespace format_tag;
            assert(engine->kind() == engine_kind::gpu);

            bool ok = true && !is_fwd()
                    && set_default_params() == status::success
                    && desc()->prop_kind == backward_data
                    && utils::one_of(desc()->alg_kind, pooling_max,
                            pooling_avg_include_padding,
                            pooling_avg_exclude_padding)
                    && (utils::everyone_is(data_type::f32,
                                diff_dst_md()->data_type,
                                diff_src_md()->data_type)
                            || utils::everyone_is(data_type::f16,
                                    diff_dst_md()->data_type,
                                    diff_src_md()->data_type))
                    && attr()->has_default_values() && no_blocking();
            if (!ok) return status::unimplemented;

            init_mem_by_tag(get_tag(diff_dst_md_), diff_src_md_);

            init_ws(this, ws_md_);
            if (!compare_ws(hint_fwd_pd_)) return status::unimplemented;

            if (has_zero_dim_memory()) { return status::success; };

            pooling_impl_.reset(new cudnn_pooling_bwd_impl_t());
            return pooling_impl_->init(this);
        }

        bool no_blocking() const {
            return diff_src_md()->format_desc.blocking.inner_nblks
                    + diff_dst_md()->format_desc.blocking.inner_nblks
                    == 0;
        }

        std::shared_ptr<cudnn_pooling_impl_base_t> pooling_impl_;
    };

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
