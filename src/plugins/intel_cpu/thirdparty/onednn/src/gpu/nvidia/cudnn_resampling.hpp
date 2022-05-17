/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef GPU_NVIDIA_CUDNN_RESAMPLING_HPP
#define GPU_NVIDIA_CUDNN_RESAMPLING_HPP

#include <cudnn.h>
#include <CL/sycl.hpp>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/resampling_pd.hpp"
#include "common/type_helpers.hpp"
#include "gpu/nvidia/sycl_cuda_engine.hpp"
#include "gpu/nvidia/sycl_cuda_scoped_context.hpp"
#include "gpu/nvidia/sycl_cuda_stream.hpp"
#include "gpu/nvidia/sycl_cuda_utils.hpp"

#include "gpu/nvidia/cudnn_resampling_impl.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

struct cudnn_resampling_pd_base_t {
protected:
    status_t init_mem_by_tag(format_tag_t tag, memory_desc_t &md) {
        if (tag == format_tag::undef) return status::unimplemented;
        CHECK(memory_desc_init_by_tag(md, tag));
        return status::success;
    }
};

struct cudnn_resampling_base_t : public primitive_t {
protected:
    using primitive_t::primitive_t;
    template <typename data_t>
    struct theta_t {
        data_t s0_, i_, tx_;
        data_t j_, s1_, ty_;
        theta_t(data_t s0, data_t i, data_t tx, data_t j, data_t s1, data_t ty)
            : s0_(s0), i_(i), tx_(tx), j_(j), s1_(s1), ty_(ty) {}
    };

    cl::sycl::buffer<uint8_t, 1> &buffer(memory_storage_t *mem_storage) {
        return utils::downcast<sycl::sycl_buffer_memory_storage_t *>(
                mem_storage)
                ->buffer();
    }
    cl::sycl::buffer<uint8_t, 1> &buffer(memory_storage_t *mem_storage) const {
        return utils::downcast<sycl::sycl_buffer_memory_storage_t *>(
                mem_storage)
                ->buffer();
    }
    template <typename data_t, typename pd_t>
    status_t prepare_coordinate_grid(engine_t *engine, const pd_t *pd) {
        using io = cudnn_resampling_impl_base_t::io;
        int ndims = pd->resampling_impl_->ndims();
        data_t OW = pd->resampling_impl_->dims_[io::dst][ndims - 1],
               IW = pd->resampling_impl_->dims_[io::src][ndims - 1],
               OH = pd->resampling_impl_->dims_[io::dst][ndims - 2],
               IH = pd->resampling_impl_->dims_[io::src][ndims - 2];
        // cudnn uses the normalized value between -1<=(xsi, ysi)<= 1 for
        // building the grid. Therefore, scaling parameter for tau_theta must be
        // adjusted for computing the normalized value per grid.
        data_t w = 1;
        if (IW != 1 && IW != OW) w = IW * (OW - 1) / (OW * (IW - 1));

        data_t h = 1;
        if (IH != 1 && IH != OH) h = IH * (OH - 1) / (OH * (IH - 1));

        // the taue of theta size is fixed in cudnn
        int tau_thea_size = 2 * 3;
        auto theta_size = pd->MB();
        auto tau_theta = theta_t<data_t> {w, 0.f, 0.f, 0.f, h, 0.f};
        std::vector<theta_t<data_t>> theta_data(theta_size, tau_theta);

        auto grid_size = pd->MB() * pd->OH() * pd->OW() * 2;
        auto sycl_engine = utils::downcast<sycl_cuda_engine_t *>(engine);

        auto theta_size_in_byte = tau_thea_size * theta_size * sizeof(data_t);
        auto grid_size_in_byte = grid_size * sizeof(data_t);

        memory_storage_t *mem_grid_ptr;
        CHECK(sycl_engine->create_memory_storage(&mem_grid_ptr,
                memory_flags_t::alloc, grid_size_in_byte, nullptr));
        grid_storage_.reset(mem_grid_ptr);

        memory_storage_t *mem_theta_ptr;
        CHECK(sycl_engine->create_memory_storage(&mem_theta_ptr,
                memory_flags_t::alloc, theta_size_in_byte, nullptr));
        theta_storage_.reset(mem_theta_ptr);

        stream_t *service_stream;
        CHECK(sycl_engine->get_service_stream(service_stream));

        auto cuda_stream
                = utils::downcast<sycl_cuda_stream_t *>(service_stream);
        auto event = copy(cuda_stream->queue(),
                reinterpret_cast<uint8_t *>(theta_data.data()),
                buffer(theta_storage_.get()));
        auto &st_desc_ = pd->resampling_impl_->st_desc_;
        cuda_stream->interop_task([&](cl::sycl::handler &cgh) {
            cgh.depends_on(event);
            auto theta_acc
                    = buffer(theta_storage_.get())
                              .get_access<cl::sycl::access::mode::read>(cgh);
            auto grid_acc
                    = buffer(grid_storage_.get())
                              .get_access<cl::sycl::access::mode::write>(cgh);

            cgh.interop_task([=](const cl::sycl::interop_handler &ih) {
                // scoped context will make sure the top of the stack context is
                // the engine context while creating the cublas handle.
                auto &s_engine = *utils::downcast<sycl_cuda_engine_t *>(engine);
                cuda_sycl_scoped_context_handler_t sc(s_engine);
                auto handle = cuda_stream->get_cudnn_handle();
                auto theta = sc.memory<void *>(ih, theta_acc);
                auto grid = sc.memory<void *>(ih, grid_acc);
                CUDNN_EXECUTE_FUNC(cudnnSpatialTfGridGeneratorForward, handle,
                        st_desc_, theta, grid);
            });
        });

        // cudnn requires the grid data to be normalized between (-1, -1) <=
        // (xsi, ysi) <= (1,1) when the value is outside of the boundary, cudnn
        // assume the values are 0, while oneDNN uses the boundary values. So we
        // clamp the outside of the boundary values to the boundary,. This will
        // fix the upsampling issue.
        std::vector<data_t> unbound_raw_grid(grid_size);
        auto event2 = copy(cuda_stream->queue(), buffer(grid_storage_.get()),
                reinterpret_cast<uint8_t *>(unbound_raw_grid.data()));
        event2.wait();
        for (int i = 0; i < grid_size; i++) {
            if (std::fabs(unbound_raw_grid[i]) > 1)
                unbound_raw_grid[i] = unbound_raw_grid[i]
                        / (std::fabs(unbound_raw_grid[i]));
        }

        auto event3 = copy(cuda_stream->queue(),
                reinterpret_cast<uint8_t *>(unbound_raw_grid.data()),
                buffer(grid_storage_.get()));
        event3.wait();
        return status::success;
    }
    std::unique_ptr<memory_storage_t> grid_storage_;
    std::unique_ptr<memory_storage_t> theta_storage_;
};

struct cudnn_resampling_fwd_t : public cudnn_resampling_base_t {
    using cudnn_resampling_base_t::cudnn_resampling_base_t;
    struct pd_t : public resampling_fwd_pd_t,
                  public cudnn_resampling_pd_base_t {
        using cudnn_resampling_pd_base_t::cudnn_resampling_pd_base_t;
        using resampling_fwd_pd_t::resampling_fwd_pd_t;
        DECLARE_COMMON_PD_T("cuda:cudnn:any", cudnn_resampling_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using namespace format_tag;

            assert(engine->kind() == engine_kind::gpu);

            bool ok = desc()->alg_kind == alg_kind::resampling_linear
                    && is_fwd() && utils::one_of(src_md()->data_type, f32, f16)
                    && src_md()->data_type == dst_md()->data_type
                    && set_default_params() == status::success
                    && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            // src must have a tag and src must follow the same tag
            format_tag_t dat_tag = memory_desc_matches_one_of_tag(
                    *src_md(), ncw, nchw, nwc, nhwc);
            if (dat_tag == format_tag::undef) return status::unimplemented;
            if (!memory_desc_matches_tag(*dst_md(), dat_tag)) {
                return status::unimplemented;
            }

            resampling_impl_.reset(new cudnn_resampling_fwd_impl_t());
            return resampling_impl_->init(this);
        }

        std::shared_ptr<cudnn_resampling_impl_base_t> resampling_impl_;
    };

    status_t init(engine_t *engine) override {
        status_t status;
        auto wrap = memory_desc_wrapper(pd()->src_md());
        switch (wrap.data_type()) {
            case data_type::f32:
                status = prepare_coordinate_grid<float>(engine, pd());
                break;
            case data_type::f16:
                status = prepare_coordinate_grid<float16_t>(engine, pd());
                break;
            default: status = status::unimplemented;
        }
        return status;
    }

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

struct cudnn_resampling_bwd_t : public cudnn_resampling_base_t {
    using cudnn_resampling_base_t::cudnn_resampling_base_t;
    struct pd_t : public resampling_bwd_pd_t,
                  public cudnn_resampling_pd_base_t {
        using cudnn_resampling_pd_base_t::cudnn_resampling_pd_base_t;
        using resampling_bwd_pd_t::resampling_bwd_pd_t;
        DECLARE_COMMON_PD_T("cuda:cudnn:any", cudnn_resampling_bwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using namespace format_tag;

            assert(engine->kind() == engine_kind::gpu);
            bool ok = desc()->alg_kind == alg_kind::resampling_linear
                    && !is_fwd() && utils::one_of(diff_src_md()->data_type, f32)
                    && diff_src_md()->data_type == diff_dst_md()->data_type
                    && set_default_params() == status::success
                    && attr()->has_default_values();
            if (!ok) return status::unimplemented;
            // dst must have a tag and src must follow the same tag
            format_tag_t dat_tag = memory_desc_matches_one_of_tag(
                    *diff_dst_md(), ncw, nchw, nwc, nhwc);
            if (dat_tag == format_tag::undef) return status::unimplemented;
            if (!memory_desc_matches_tag(*diff_src_md(), dat_tag)) {
                return status::unimplemented;
            }

            resampling_impl_.reset(new cudnn_resampling_bwd_impl_t());
            return resampling_impl_->init(this);
        }
        std::shared_ptr<cudnn_resampling_impl_base_t> resampling_impl_;
    };
    status_t init(engine_t *engine) override {
        return prepare_coordinate_grid<float>(engine, pd());
    }

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
