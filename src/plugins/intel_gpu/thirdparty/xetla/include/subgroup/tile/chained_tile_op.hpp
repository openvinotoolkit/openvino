/*******************************************************************************
* Copyright (c) 2022-2023 Intel Corporation
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

/// @file
/// C++ API

#pragma once

#include "subgroup/tile/common.hpp"

namespace gpu::xetla::subgroup {

template <int idx, typename tile_op_args_t>
struct tile_op_arg_helper_t {
    tile_op_args_t args;
    inline tile_op_arg_helper_t(tile_op_args_t args_) : args(args_) {}
    // Be aware of the risks: Rule of three (copy constructor, copy assignment, destructor)
    // Please check if you need to add self-define destructor
    // ~tile_op_arg_helper_t(){}
    inline tile_op_arg_helper_t(
            const tile_op_arg_helper_t<idx, tile_op_args_t> &args_helper)
        : args(args_helper.args) {}
    // Be aware of the risks: Rule of three (copy constructor, copy assignment, destructor)
    // Please check if you need to add self-define destructor
    // inline ~tile_op_arg_helper_t(){}
    inline tile_op_arg_helper_t &operator=(
            const tile_op_arg_helper_t<idx, tile_op_args_t> &args_helper) {
        this->args = args_helper.args;
        return *this;
    }
    inline tile_op_arg_helper_t() = default;
    // Be aware of the risks: Rule of three (copy constructor, copy assignment, destructor)
    // Please check if you need to add self-define destructor
    // ~tile_op_arg_helper_t(){}

    inline tile_op_args_t get_args() const { return args; }
    inline void set_args(const tile_op_args_t &new_args) { args = new_args; }
};

template <int idx, typename... tile_op_args_t>
struct chained_tile_op_arg_t {};

template <int idx, typename curr_args_t, typename... remain_args_t>
struct chained_tile_op_arg_t<idx, curr_args_t, remain_args_t...>
    : public tile_op_arg_helper_t<idx, curr_args_t>,
      public chained_tile_op_arg_t<idx + 1, remain_args_t...> {
    inline chained_tile_op_arg_t() = default;
    inline chained_tile_op_arg_t(
            curr_args_t curr_args, remain_args_t... remain_args)
        : tile_op_arg_helper_t<idx, curr_args_t>(curr_args)
        , chained_tile_op_arg_t<idx + 1, remain_args_t...>(remain_args...) {}

    inline chained_tile_op_arg_t(
            chained_tile_op_arg_t<idx, curr_args_t, remain_args_t...> const
                    &args)
            = default;

    template <int idx_, typename T>
    inline T get() const {
        return this->tile_op_arg_helper_t<idx_, T>::get_args();
    }
    template <int idx_, typename T>
    inline void set(T new_args) {
        this->tile_op_arg_helper_t<idx_, T>::set_args(new_args);
    }
};

template <typename... tile_op_t>
struct chained_tile_op_t {
    using arguments_t
            = chained_tile_op_arg_t<0, typename tile_op_t::arguments_t...>;
    static constexpr int list_size = sizeof...(tile_op_t);
    template <typename matAcc_t, typename coord_t>
    __XETLA_API KERNEL_FUNC void operator()(matAcc_t &matAcc,
            const coord_t &coord, const arguments_t &args_helper,
            uint32_t slm_base = 0, uint32_t nbarrier_base = 0) {
        if constexpr (list_size == 0) {
            return;
        } else {
            chained_tile_op_helper<tile_op_t...> chained_tile_op;
            chained_tile_op(
                    matAcc, coord, args_helper, slm_base, nbarrier_base);
        }
    }

private:
    template <typename... total_tile_op_t>
    struct chained_tile_op_helper {};

    template <typename curr_tile_op_t, typename... remain_tile_op_t>
    struct chained_tile_op_helper<curr_tile_op_t, remain_tile_op_t...> {
        using curr_tile_op_args_t = typename curr_tile_op_t::arguments_t;
        static constexpr int curr_idx
                = list_size - sizeof...(remain_tile_op_t) - 1;
        template <typename matAcc_t, typename coord_t>
        __XETLA_API KERNEL_FUNC void operator()(matAcc_t &matAcc,
                const coord_t &coord, const arguments_t &args_helper,
                uint32_t slm_base = 0, uint32_t nbarrier_base = 0) {
            curr_tile_op_t curr_tile_op;
            //call the actual tile op
            curr_tile_op(matAcc, coord,
                    args_helper.template get<curr_idx, curr_tile_op_args_t>(),
                    slm_base, nbarrier_base);
            if constexpr (sizeof...(remain_tile_op_t) > 0) {
                chained_tile_op_helper<remain_tile_op_t...> remain_tile_op;
                remain_tile_op(
                        matAcc, coord, args_helper, slm_base, nbarrier_base);
            }
        }
    };
};

} // namespace gpu::xetla::subgroup
