/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include <utility>

#include "cpu/x64/ip_convolution.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

status_t ip_convolution_fwd_t::execute(const exec_ctx_t &ctx) const {
    using namespace memory_tracking::names;

    exec_args_t ip_args = ctx.args();

    exec_ctx_t conv_ctx(ctx, std::move(ip_args));

    nested_scratchpad_t ns(ctx, key_nested, ip_p_);
    conv_ctx.set_scratchpad_grantor(ns.grantor());

    return ip_p_->execute(conv_ctx);
}

status_t ip_convolution_bwd_data_t::execute(const exec_ctx_t &ctx) const {
    using namespace memory_tracking::names;

    exec_args_t ip_args = ctx.args();

    exec_ctx_t conv_ctx(ctx, std::move(ip_args));

    nested_scratchpad_t ns(ctx, key_nested, ip_p_);
    conv_ctx.set_scratchpad_grantor(ns.grantor());

    return ip_p_->execute(conv_ctx);
}

status_t ip_convolution_bwd_weights_t::execute(const exec_ctx_t &ctx) const {
    using namespace memory_tracking::names;

    exec_args_t ip_args = ctx.args();

    exec_ctx_t conv_ctx(ctx, std::move(ip_args));

    nested_scratchpad_t ns(ctx, key_nested, ip_p_);
    conv_ctx.set_scratchpad_grantor(ns.grantor());

    return ip_p_->execute(conv_ctx);
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
