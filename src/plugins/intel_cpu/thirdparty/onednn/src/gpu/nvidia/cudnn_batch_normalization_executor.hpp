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

#ifndef GPU_NVIDIA_CUDNN_BATCH_NORMALIZATION_EXECUTOR_HPP
#define GPU_NVIDIA_CUDNN_BATCH_NORMALIZATION_EXECUTOR_HPP

#include "common/batch_normalization_pd.hpp"
#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "gpu/nvidia/cudnn_batch_normalization_impl.hpp"
#include "gpu/nvidia/sycl_cuda_engine.hpp"
#include "gpu/nvidia/sycl_cuda_scoped_context.hpp"
#include "gpu/nvidia/sycl_cuda_stream.hpp"
#include "gpu/nvidia/sycl_cuda_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

struct bnorm_exec_base_t {
    virtual status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            const std::shared_ptr<cudnn_batch_normalization_impl_base_t>
                    bnorm_impl) const = 0;
    virtual ~bnorm_exec_base_t() = default;

protected:
    template <typename T, cl::sycl::access::mode md, typename sc_t>
    void *mean_var_ptr(cl::sycl::accessor<T, 1, md> acc, sc_t &sc,
            const cl::sycl::interop_handler &ih) const {
        return sc.template memory<void *>(ih, acc);
    }

    template <typename sc_t>
    std::nullptr_t mean_var_ptr(std::nullptr_t acc, sc_t &,
            const cl::sycl::interop_handler &ih) const {
        return acc;
    }

    template <typename read_acc_t, typename write_acc_t, typename wkspace_st_t,
            typename float_acc_t, typename maybe_nullptr_t>
    void interop_task_fwd(
            std::shared_ptr<cudnn_batch_normalization_impl_base_t> bnorm_impl,
            engine_t *engine, cl::sycl::handler &cgh,
            nvidia::sycl_cuda_stream_t *cuda_stream, read_acc_t src_acc,
            write_acc_t dst_acc, maybe_nullptr_t mean_acc,
            maybe_nullptr_t var_acc, float_acc_t scale_acc,
            float_acc_t bias_acc, wkspace_st_t wkspace_st, bool init_ss,
            bool init_mean_var) const {

        std::shared_ptr<
                cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::write>>
                wkspace_acc;
        if (!wkspace_st->is_null()) {
            wkspace_acc.reset(new cl::sycl::accessor<uint8_t, 1,
                    cl::sycl::access::mode::write>(
                    utils::downcast<sycl::sycl_buffer_memory_storage_t *>(
                            wkspace_st)
                            ->buffer()
                            .template get_access<cl::sycl::access::mode::write>(
                                    cgh)));
        }

        maybe_init_mean_var(cuda_stream, mean_acc, var_acc, init_mean_var);
        maybe_init_ss(cuda_stream, scale_acc, bias_acc, init_ss);
        cgh.interop_task([=](const cl::sycl::interop_handler &ih) {
            auto &sycl_engine = *utils::downcast<sycl_cuda_engine_t *>(engine);
            auto sc = cuda_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = cuda_stream->get_cudnn_handle();

            auto x = sc.memory<void *>(ih, src_acc);
            auto y = sc.memory<void *>(ih, dst_acc);
            auto mean = mean_var_ptr(mean_acc, sc, ih);
            auto var = mean_var_ptr(var_acc, sc, ih);
            auto scale = sc.memory<float *>(ih, scale_acc);
            auto bias = sc.memory<float *>(ih, bias_acc) + bnorm_impl->C();
            uint8_t *y_prime = nullptr, *save_mean = nullptr,
                    *save_var = nullptr;
            if (!wkspace_st->is_null()) {
                save_mean = sc.memory<uint8_t *>(ih, *wkspace_acc);
                save_var = save_mean + bnorm_impl->mean_var_size_bytes();
                y_prime = save_var + bnorm_impl->mean_var_size_bytes();
            }

            std::shared_ptr<bnorm_args_t> args(new bnorm_fwd_args_t(x, y, mean,
                    var, scale, bias, y_prime, save_mean, save_var));

            bnorm_impl->execute(handle, args);
        });
    }

    template <typename read_acc_t, typename write_acc_t, typename ss_acc_t,
            typename d_ss_acc_t>
    void interop_task_bwd(
            std::shared_ptr<cudnn_batch_normalization_impl_base_t> bnorm_impl,
            engine_t *engine, cl::sycl::handler &cgh,
            nvidia::sycl_cuda_stream_t *cuda_stream, read_acc_t src_acc,
            read_acc_t diff_dst_acc, write_acc_t diff_src_acc,
            ss_acc_t scale_acc, ss_acc_t bias_acc,
            d_ss_acc_t diff_scaleshift_acc, read_acc_t wkspace_acc,
            std::shared_ptr<cl::sycl::accessor<uint8_t, 1,
                    cl::sycl::access::mode::read_write,
                    cl::sycl::access::target::global_buffer>>
                    temp_relu_output,
            bool init_ss, bool init_mean_var) const {

        maybe_init_ss(cuda_stream, scale_acc, bias_acc, init_ss);
        cgh.interop_task([=](const cl::sycl::interop_handler &ih) {
            auto &sycl_engine = *utils::downcast<sycl_cuda_engine_t *>(engine);
            auto sc = cuda_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = cuda_stream->get_cudnn_handle();

            auto x = sc.memory<void *>(ih, src_acc);
            auto dy = sc.memory<void *>(ih, diff_dst_acc);
            auto dx = sc.memory<void *>(ih, diff_src_acc);
            auto scale = sc.memory<uint8_t *>(ih, scale_acc);
            auto bias = sc.memory<uint8_t *>(ih, bias_acc)
                    + (bnorm_impl->C() * sizeof(float));
            auto diff_scale = sc.memory<uint8_t *>(ih, diff_scaleshift_acc);
            auto diff_bias = diff_scale + (bnorm_impl->C() * sizeof(float));
            auto save_mean = sc.memory<uint8_t *>(ih, wkspace_acc);
            auto save_var = save_mean + bnorm_impl->mean_var_size_bytes();
            auto wkspace = save_var + bnorm_impl->mean_var_size_bytes();
            auto relu_dy = bnorm_impl->fuse_norm_relu()
                    ? sc.memory<void *>(ih, *temp_relu_output)
                    : nullptr;

            std::shared_ptr<bnorm_args_t> args(
                    new bnorm_bwd_args_t(x, dx, dy, save_mean, save_var, scale,
                            bias, diff_scale, diff_bias, wkspace, relu_dy));

            bnorm_impl->execute(handle, args);
        });
    }

    template <typename T>
    void maybe_init_ss(
            nvidia::sycl_cuda_stream_t *cuda_stream, T, T, bool) const {}

    template <typename T>
    void maybe_init_ss(nvidia::sycl_cuda_stream_t *cuda_stream,
            cl::sycl::accessor<T, 1, cl::sycl::access::mode::write> scale_acc,
            cl::sycl::accessor<T, 1, cl::sycl::access::mode::write> bias_acc,
            bool init_ss) const {
        if (init_ss) {
            constexpr T scale_val = 1, bias_val = 0;
            cuda_stream->interop_task([&](cl::sycl::handler &cgh) {
                cgh.fill(scale_acc, scale_val);
            });

            cuda_stream->interop_task([&](cl::sycl::handler &cgh) {
                cgh.fill(bias_acc, bias_val);
            });
        }
    }

    // Handle the cases when mean and var are read-only accessors or nullptr
    template <typename T>
    void maybe_init_mean_var(
            nvidia::sycl_cuda_stream_t *cuda_stream, T, T, bool) const {}

    template <typename T>
    void maybe_init_mean_var(nvidia::sycl_cuda_stream_t *cuda_stream,
            cl::sycl::accessor<T, 1, cl::sycl::access::mode::write> mean_acc,
            cl::sycl::accessor<T, 1, cl::sycl::access::mode::write> var_acc,
            bool init_mean_var) const {
        if (init_mean_var) {
            constexpr T mean_var_val = 0;
            cuda_stream->interop_task([&](cl::sycl::handler &cgh) {
                cgh.fill(mean_acc, mean_var_val);
            });

            cuda_stream->interop_task([&](cl::sycl::handler &cgh) {
                cgh.fill(var_acc, mean_var_val);
            });
        }
    }
};

struct bnorm_exec_fwd_inf_t : public bnorm_exec_base_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            std::shared_ptr<cudnn_batch_normalization_impl_base_t> bnorm_impl)
            const override {
        nvidia::sycl_cuda_stream_t *cuda_stream
                = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

        auto wkspace_storage = bnorm_impl->is_training()
                ? ctx.output(DNNL_ARG_WORKSPACE)->memory_storage()
                : &memory_storage_t::empty_storage();

        auto n_channels = bnorm_impl->C();
        cl::sycl::buffer<float> scaleshift_buff(n_channels * 2);
        return cuda_stream->interop_task([&](cl::sycl::handler &cgh) {
            auto src_acc = CTX_IN_ACCESSOR(DNNL_ARG_SRC);
            auto dst_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DST);
            auto scale_acc
                    = scaleshift_buff.get_access<cl::sycl::access::mode::write>(
                            cgh, n_channels, 0);
            auto bias_acc
                    = scaleshift_buff.get_access<cl::sycl::access::mode::write>(
                            cgh, n_channels, n_channels);
            bool init_ss = true, init_mean_var = false;

            interop_task_fwd(bnorm_impl, engine, cgh, cuda_stream, src_acc,
                    dst_acc, nullptr, nullptr, scale_acc, bias_acc,
                    wkspace_storage, init_ss, init_mean_var);
        });
    }
};

struct bnorm_exec_fwd_inf_ss_t : public bnorm_exec_base_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            std::shared_ptr<cudnn_batch_normalization_impl_base_t> bnorm_impl)
            const override {
        nvidia::sycl_cuda_stream_t *cuda_stream
                = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

        auto wkspace_storage = bnorm_impl->is_training()
                ? ctx.output(DNNL_ARG_WORKSPACE)->memory_storage()
                : &memory_storage_t::empty_storage();

        auto scaleshift_buff
                = utils::downcast<sycl::sycl_buffer_memory_storage_t *>(
                        &CTX_IN_STORAGE(DNNL_ARG_SCALE_SHIFT))
                          ->buffer();
        auto n_channels = bnorm_impl->C();

        return cuda_stream->interop_task([&](cl::sycl::handler &cgh) {
            auto src_acc = CTX_IN_ACCESSOR(DNNL_ARG_SRC);
            auto dst_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DST);
            auto scale_acc
                    = scaleshift_buff.get_access<cl::sycl::access::mode::read>(
                            cgh, n_channels, 0);
            auto bias_acc
                    = scaleshift_buff.get_access<cl::sycl::access::mode::read>(
                            cgh, n_channels, n_channels);
            bool init_ss = false, init_mean_var = false;

            interop_task_fwd(bnorm_impl, engine, cgh, cuda_stream, src_acc,
                    dst_acc, nullptr, nullptr, scale_acc, bias_acc,
                    wkspace_storage, init_ss, init_mean_var);
        });
    }
};

struct bnorm_exec_fwd_inf_stats_t : public bnorm_exec_base_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            std::shared_ptr<cudnn_batch_normalization_impl_base_t> bnorm_impl)
            const override {
        nvidia::sycl_cuda_stream_t *cuda_stream
                = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

        auto wkspace_storage = bnorm_impl->is_training()
                ? ctx.output(DNNL_ARG_WORKSPACE)->memory_storage()
                : &memory_storage_t::empty_storage();

        auto n_channels = bnorm_impl->C();
        cl::sycl::buffer<float> scaleshift_buff(n_channels * 2);
        return cuda_stream->interop_task([&](cl::sycl::handler &cgh) {
            auto src_acc = CTX_IN_ACCESSOR(DNNL_ARG_SRC);
            auto dst_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DST);
            auto mean_acc = CTX_IN_ACCESSOR(DNNL_ARG_MEAN);
            auto var_acc = CTX_IN_ACCESSOR(DNNL_ARG_VARIANCE);
            auto scale_acc
                    = scaleshift_buff.get_access<cl::sycl::access::mode::write>(
                            cgh, n_channels, 0);
            auto bias_acc
                    = scaleshift_buff.get_access<cl::sycl::access::mode::write>(
                            cgh, n_channels, n_channels);
            bool init_ss = true, init_mean_var = false;

            interop_task_fwd(bnorm_impl, engine, cgh, cuda_stream, src_acc,
                    dst_acc, mean_acc, var_acc, scale_acc, bias_acc,
                    wkspace_storage, init_ss, init_mean_var);
        });
    }
};

struct bnorm_exec_fwd_inf_ss_stats_t : public bnorm_exec_base_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            std::shared_ptr<cudnn_batch_normalization_impl_base_t> bnorm_impl)
            const override {
        nvidia::sycl_cuda_stream_t *cuda_stream
                = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

        auto wkspace_storage = bnorm_impl->is_training()
                ? ctx.output(DNNL_ARG_WORKSPACE)->memory_storage()
                : &memory_storage_t::empty_storage();

        auto scaleshift_buff
                = utils::downcast<sycl::sycl_buffer_memory_storage_t *>(
                        &CTX_IN_STORAGE(DNNL_ARG_SCALE_SHIFT))
                          ->buffer();
        auto n_channels = bnorm_impl->C();

        return cuda_stream->interop_task([&](cl::sycl::handler &cgh) {
            auto src_acc = CTX_IN_ACCESSOR(DNNL_ARG_SRC);
            auto dst_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DST);
            auto mean_acc = CTX_IN_ACCESSOR(DNNL_ARG_MEAN);
            auto var_acc = CTX_IN_ACCESSOR(DNNL_ARG_VARIANCE);
            auto scale_acc
                    = scaleshift_buff.get_access<cl::sycl::access::mode::read>(
                            cgh, n_channels, 0);
            auto bias_acc
                    = scaleshift_buff.get_access<cl::sycl::access::mode::read>(
                            cgh, n_channels, n_channels);
            bool init_ss = false, init_mean_var = false;

            interop_task_fwd(bnorm_impl, engine, cgh, cuda_stream, src_acc,
                    dst_acc, mean_acc, var_acc, scale_acc, bias_acc,
                    wkspace_storage, init_ss, init_mean_var);
        });
    }
};

struct bnorm_exec_fwd_t : public bnorm_exec_base_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            std::shared_ptr<cudnn_batch_normalization_impl_base_t> bnorm_impl)
            const override {
        nvidia::sycl_cuda_stream_t *cuda_stream
                = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

        auto wkspace_storage = bnorm_impl->is_training()
                ? ctx.output(DNNL_ARG_WORKSPACE)->memory_storage()
                : &memory_storage_t::empty_storage();

        auto n_channels = bnorm_impl->C();

        cl::sycl::buffer<float> scaleshift_buff(n_channels * 2);
        return cuda_stream->interop_task([&](cl::sycl::handler &cgh) {
            auto src_acc = CTX_IN_ACCESSOR(DNNL_ARG_SRC);
            auto dst_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DST);
            auto mean_acc = CTX_OUT_ACCESSOR(DNNL_ARG_MEAN);
            auto var_acc = CTX_OUT_ACCESSOR(DNNL_ARG_VARIANCE);
            auto scale_acc
                    = scaleshift_buff.get_access<cl::sycl::access::mode::write>(
                            cgh, n_channels, 0);
            auto bias_acc
                    = scaleshift_buff.get_access<cl::sycl::access::mode::write>(
                            cgh, n_channels, n_channels);
            bool init_ss = true, init_mean_var = true;

            interop_task_fwd(bnorm_impl, engine, cgh, cuda_stream, src_acc,
                    dst_acc, mean_acc, var_acc, scale_acc, bias_acc,
                    wkspace_storage, init_ss, init_mean_var);
        });
    }
};

struct bnorm_exec_fwd_ss_t : public bnorm_exec_base_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            std::shared_ptr<cudnn_batch_normalization_impl_base_t> bnorm_impl)
            const override {
        nvidia::sycl_cuda_stream_t *cuda_stream
                = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

        auto wkspace_storage = bnorm_impl->is_training()
                ? ctx.output(DNNL_ARG_WORKSPACE)->memory_storage()
                : &memory_storage_t::empty_storage();

        auto scaleshift_buff
                = utils::downcast<sycl::sycl_buffer_memory_storage_t *>(
                        &CTX_IN_STORAGE(DNNL_ARG_SCALE_SHIFT))
                          ->buffer();
        auto n_channels = bnorm_impl->C();

        return cuda_stream->interop_task([&](cl::sycl::handler &cgh) {
            auto src_acc = CTX_IN_ACCESSOR(DNNL_ARG_SRC);
            auto dst_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DST);
            auto mean_acc = CTX_OUT_ACCESSOR(DNNL_ARG_MEAN);
            auto var_acc = CTX_OUT_ACCESSOR(DNNL_ARG_VARIANCE);
            auto scale_acc
                    = scaleshift_buff.get_access<cl::sycl::access::mode::write>(
                            cgh, n_channels, 0);
            auto bias_acc
                    = scaleshift_buff.get_access<cl::sycl::access::mode::write>(
                            cgh, n_channels, n_channels);
            bool init_ss = false, init_mean_var = true;

            interop_task_fwd(bnorm_impl, engine, cgh, cuda_stream, src_acc,
                    dst_acc, mean_acc, var_acc, scale_acc, bias_acc,
                    wkspace_storage, init_ss, init_mean_var);
        });
    }
};

struct bnorm_exec_bwd_t : public bnorm_exec_base_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            std::shared_ptr<cudnn_batch_normalization_impl_base_t> bnorm_impl)
            const override {
        nvidia::sycl_cuda_stream_t *cuda_stream
                = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

        auto n_channels = bnorm_impl->C();
        cl::sycl::buffer<float> scaleshift_buff(n_channels * 2);
        cl::sycl::buffer<float> diff_scaleshift_buff(n_channels * 2);

        return cuda_stream->interop_task([&](cl::sycl::handler &cgh) {
            auto src_acc = CTX_IN_ACCESSOR(DNNL_ARG_SRC);
            auto diff_dst_acc = CTX_IN_ACCESSOR(DNNL_ARG_DIFF_DST);
            auto diff_src_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DIFF_SRC);
            auto wkspace_acc = CTX_IN_ACCESSOR(DNNL_ARG_WORKSPACE);
            auto diff_scaleshift_acc
                    = diff_scaleshift_buff
                              .get_access<cl::sycl::access::mode::read>(cgh);
            auto scale_acc
                    = scaleshift_buff.get_access<cl::sycl::access::mode::write>(
                            cgh, n_channels, 0);
            auto bias_acc
                    = scaleshift_buff.get_access<cl::sycl::access::mode::write>(
                            cgh, n_channels, n_channels);
            bool init_ss = true, init_mean_var = false;

            std::shared_ptr<cl::sycl::accessor<uint8_t, 1,
                    cl::sycl::access::mode::read_write,
                    cl::sycl::access::target::global_buffer>>
                    temp_relu_output = nullptr;
            if (bnorm_impl->fuse_norm_relu()) {
                temp_relu_output = std::make_shared<cl::sycl::accessor<uint8_t,
                        1, cl::sycl::access::mode::read_write,
                        cl::sycl::access::target::global_buffer>>(
                        CTX_SCRATCH_ACCESSOR(memory_tracking::names::key_none));
            }

            interop_task_bwd(bnorm_impl, engine, cgh, cuda_stream, src_acc,
                    diff_dst_acc, diff_src_acc, scale_acc, bias_acc,
                    diff_scaleshift_acc, wkspace_acc, temp_relu_output, init_ss,
                    init_mean_var);
        });
    }
};

struct bnorm_exec_bwd_dw_ss_t : public bnorm_exec_base_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            std::shared_ptr<cudnn_batch_normalization_impl_base_t> bnorm_impl)
            const override {
        nvidia::sycl_cuda_stream_t *cuda_stream
                = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

        auto scaleshift_buff
                = utils::downcast<sycl::sycl_buffer_memory_storage_t *>(
                        &CTX_IN_STORAGE(DNNL_ARG_SCALE_SHIFT))
                          ->buffer();

        auto n_channels = bnorm_impl->C();

        return cuda_stream->interop_task([&](cl::sycl::handler &cgh) {
            auto src_acc = CTX_IN_ACCESSOR(DNNL_ARG_SRC);
            auto diff_dst_acc = CTX_IN_ACCESSOR(DNNL_ARG_DIFF_DST);
            auto diff_src_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DIFF_SRC);
            auto diff_scaleshift_acc
                    = CTX_OUT_ACCESSOR(DNNL_ARG_DIFF_SCALE_SHIFT);
            auto scale_acc
                    = scaleshift_buff.get_access<cl::sycl::access::mode::read>(
                            cgh, n_channels, 0);
            auto bias_acc
                    = scaleshift_buff.get_access<cl::sycl::access::mode::read>(
                            cgh, n_channels, n_channels);
            auto wkspace_acc = CTX_IN_ACCESSOR(DNNL_ARG_WORKSPACE);
            bool init_ss = false, init_mean_var = false;

            std::shared_ptr<cl::sycl::accessor<uint8_t, 1,
                    cl::sycl::access::mode::read_write,
                    cl::sycl::access::target::global_buffer>>
                    temp_relu_output = nullptr;
            if (bnorm_impl->fuse_norm_relu()) {
                temp_relu_output = std::make_shared<cl::sycl::accessor<uint8_t,
                        1, cl::sycl::access::mode::read_write,
                        cl::sycl::access::target::global_buffer>>(
                        CTX_SCRATCH_ACCESSOR(memory_tracking::names::key_none));
            }

            interop_task_bwd(bnorm_impl, engine, cgh, cuda_stream, src_acc,
                    diff_dst_acc, diff_src_acc, scale_acc, bias_acc,
                    diff_scaleshift_acc, wkspace_acc, temp_relu_output, init_ss,
                    init_mean_var);
        });
    }
};

struct bnorm_exec_bwd_d_ss_t : public bnorm_exec_base_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            std::shared_ptr<cudnn_batch_normalization_impl_base_t> bnorm_impl)
            const override {
        nvidia::sycl_cuda_stream_t *cuda_stream
                = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

        auto scaleshift_buff
                = utils::downcast<sycl::sycl_buffer_memory_storage_t *>(
                        &CTX_IN_STORAGE(DNNL_ARG_SCALE_SHIFT))
                          ->buffer();
        auto n_channels = bnorm_impl->C();

        cl::sycl::buffer<float> diff_scaleshift_buff(n_channels * 2);
        return cuda_stream->interop_task([&](cl::sycl::handler &cgh) {
            auto src_acc = CTX_IN_ACCESSOR(DNNL_ARG_SRC);
            auto diff_dst_acc = CTX_IN_ACCESSOR(DNNL_ARG_DIFF_DST);
            auto diff_src_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DIFF_SRC);
            auto scale_acc
                    = scaleshift_buff.get_access<cl::sycl::access::mode::read>(
                            cgh, n_channels, 0);
            auto bias_acc
                    = scaleshift_buff.get_access<cl::sycl::access::mode::read>(
                            cgh, n_channels, n_channels);
            auto diff_scaleshift_acc
                    = diff_scaleshift_buff
                              .get_access<cl::sycl::access::mode::read>(cgh);
            auto wkspace_acc = CTX_IN_ACCESSOR(DNNL_ARG_WORKSPACE);
            bool init_ss = false, init_mean_var = false;

            std::shared_ptr<cl::sycl::accessor<uint8_t, 1,
                    cl::sycl::access::mode::read_write,
                    cl::sycl::access::target::global_buffer>>
                    temp_relu_output = nullptr;
            if (bnorm_impl->fuse_norm_relu()) {
                temp_relu_output = std::make_shared<cl::sycl::accessor<uint8_t,
                        1, cl::sycl::access::mode::read_write,
                        cl::sycl::access::target::global_buffer>>(
                        CTX_SCRATCH_ACCESSOR(memory_tracking::names::key_none));
            }

            interop_task_bwd(bnorm_impl, engine, cgh, cuda_stream, src_acc,
                    diff_dst_acc, diff_src_acc, scale_acc, bias_acc,
                    diff_scaleshift_acc, wkspace_acc, temp_relu_output, init_ss,
                    init_mean_var);
        });
    }
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
