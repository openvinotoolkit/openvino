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

#pragma once

#ifdef _WIN32
#include "../../../common/utils/cm/common.hpp"
#else
#include "common/utils/cm/common.hpp"
#endif

namespace gpu::xetla {

// meta_value type

namespace impl {
struct meta_impl_base {
    struct _default {};
};

static constexpr meta_impl_base::_default meta_impl_base_default;

template <typename T, auto value_>
struct meta_value_impl : meta_impl_base {
    using type = T;
    static constexpr T value = value_;
};

template <typename T = meta_impl_base::_default>
struct meta_type_impl : meta_impl_base {
    using type = T;
};
} // namespace impl

template <auto d_ = impl::meta_impl_base_default, typename T = decltype(d_)>
using meta_value = impl::meta_value_impl<T, d_>;

template <auto d_ = impl::meta_impl_base_default, typename T = decltype(d_)>
using meta_value_t = typename meta_value<d_, T>::type;

template <typename T = impl::meta_impl_base::_default>
using meta_type = impl::meta_type_impl<T>;

template <typename T>
using meta_type_t = typename T::type;

// shape type

template <auto... dims>
struct shape {
    static constexpr size_t size = sizeof...(dims);

    struct impl {
        template <size_t i_, size_t cur_, auto... dims_>
        struct dim_impl;

        template <size_t i_, auto dim, auto... dims_>
        struct dim_impl<i_, i_, dim, dims_...> {
            static constexpr auto value = dim;
        };

        template <size_t i_, size_t cur_, auto dim, auto... dims_>
        struct dim_impl<i_, cur_, dim, dims_...>
            : dim_impl<i_, cur_ + 1, dims_...> {
            static_assert(i_ < size, "i_ exceeded shape size");
            static_assert(cur_ < size, "cur_ exceeded shape size");
        };
    };

    template <size_t i>
    static constexpr auto dim() {
        return impl::template dim_impl<i, 0, dims...>::value;
    }
};

template <auto... dims>
static inline constexpr auto shape_v = shape<dims...> {};

// dict type

template <auto key_, typename val_>
struct elem_t {
    static constexpr auto key = key_;
    using value = val_;
};

template <auto key_, typename T>
struct elem_t_t : elem_t<key_, meta_type<T>> {};

template <auto key_, auto val_, typename T = decltype(val_)>
struct elem_v_t : elem_t<key_, meta_value<val_, T>> {};

template <typename... Args>
struct dict_t {
    static constexpr size_t arg_size = sizeof...(Args);

    struct impl {
        template <typename T>
        struct type_identity {
            using type = T;
        };

        struct empty_dict : type_identity<dict_t<>> {};
        using this_t = dict_t<Args...>;

        static constexpr int key_not_found = -1;

        template <auto value_, typename type_>
        struct find_elem_impl_ret_type {
            static constexpr auto value = value_;
            using type = type_;
        };

        template <auto key_, size_t cur_, typename... Elems>
        struct find_elem_impl;

        template <auto key_, size_t cur_, typename elem_, typename... Elems>
        struct find_elem_impl<key_, cur_, elem_, Elems...> {
            static constexpr bool match_key = (key_ == elem_::key);
            using ret = typename std::conditional<match_key,
                    find_elem_impl_ret_type<cur_, elem_>,
                    find_elem_impl<key_, cur_ + 1, Elems...>>::type;
            static constexpr int value = ret::value;
            using type = typename ret::type;
        };

        template <auto key_, size_t cur_>
        struct find_elem_impl<key_, cur_> {
            static constexpr int value = key_not_found;
            using type = void;
        };

        template <auto key_>
        static constexpr int find_elem_index
                = find_elem_impl<key_, 0, Args...>::value;

        template <auto key_>
        struct find_elem : find_elem_impl<key_, 0, Args...>::type {};

        template <typename... Elems>
        struct prepend_key_impl : type_identity<dict_t<Elems..., Args...>> {};

        template <typename... Elems>
        struct append_key_impl : type_identity<dict_t<Args..., Elems...>> {};

        template <typename dict_t_>
        struct merge_dict_impl;

        template <typename... Elems>
        struct merge_dict_impl<dict_t<Elems...>> : append_key_impl<Elems...> {};

        template <typename dict_t_, int cur_i, int begin_i, int end_i>
        struct slicing_key_impl : empty_dict {};

        template <int cur_i, int begin_i, int end_i, typename e_,
                typename... Elems>
        struct slicing_key_impl<dict_t<e_, Elems...>, cur_i, begin_i, end_i> {
            using nxt_dict = typename std::conditional<(cur_i + 1 < end_i)
                            && (sizeof...(Elems) > 0),
                    slicing_key_impl<dict_t<Elems...>, cur_i + 1, begin_i,
                            end_i>,
                    empty_dict>::type::type;
            using type = typename std::conditional<(begin_i <= cur_i)
                            && (cur_i < end_i),
                    typename nxt_dict::impl::template prepend_key_impl<e_>,
                    type_identity<nxt_dict>>::type::type;
        };

        template <typename dict_t_, typename e_, int e_index>
        struct replace_key_impl {
            using pre_dict = typename slicing_key_impl<dict_t_, 0, 0,
                    e_index>::type::impl::template append_key_impl<e_>::type;
            using post_dict = typename slicing_key_impl<dict_t_, 0, e_index + 1,
                    dict_t_::arg_size>::type;
            using type = typename pre_dict::impl::template merge_dict_impl<
                    post_dict>::type;
        };

        template <typename dict_t_, typename... Elems>
        struct update_dict_impl : type_identity<dict_t_> {};

        template <typename dict_t_, typename e_, typename... Elems>
        struct update_dict_impl<dict_t_, e_, Elems...>
            : update_dict_impl<typename update_dict_impl<dict_t_, e_>::type,
                      Elems...> {};

        template <typename dict_t_, typename e_, typename... Elems>
        struct update_dict_impl<dict_t_, dict_t<e_, Elems...>>
            : update_dict_impl<dict_t_, e_, Elems...> {};

        template <typename dict_t_, typename e_>
        struct update_dict_impl<dict_t_, e_> {
            static constexpr int e_index
                    = dict_t_::impl::template find_elem_index<e_::key>;
            using type = typename std::conditional<e_index != key_not_found,
                    replace_key_impl<dict_t_, e_, e_index>,
                    typename dict_t_::impl::template append_key_impl<e_>>::
                    type::type;
        };

        template <typename U, template <typename> typename G>
        struct update_generator_impl {
            using res_t = typename G<U>::type;
            using type = typename update_dict_impl<this_t, res_t>::type;
        };
    };

    template <auto key_>
    using find_elem_t = typename impl::template find_elem<key_>::value;

    template <auto key_>
    using find_elem_v_type = decltype(find_elem_t<key_>::value);

    template <auto key_>
    static inline constexpr find_elem_v_type<key_> find_elem_v
            = find_elem_t<key_>::value;

    template <typename e_, typename... Elems>
    using update_t =
            typename impl::template update_dict_impl<typename impl::this_t, e_,
                    Elems...>::type;

    template <typename T>
    using update_dict_t =
            typename impl::template update_dict_impl<typename impl::this_t,
                    T>::type;

    template <template <typename> typename G>
    using update_generator_t =
            typename impl::template update_generator_impl<typename impl::this_t,
                    G>::type;
};

} // namespace gpu::xetla