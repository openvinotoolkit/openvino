// Copyright (c) 2016 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <initializer_list>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>


namespace cldnn { namespace gpu {
namespace mputils
{
template <typename ... Tys> struct type_tuple;

template <std::size_t ... Idxs> struct index_tuple {};

// -----------------------------------------------------------------------------------------------------------------------

template <typename TypeTupleTy, typename ElemTy> struct count_tt;

template <typename Ty, typename ... Tys, typename ElemTy>
struct count_tt<type_tuple<Ty, Tys ...>, ElemTy>
    : std::integral_constant<std::size_t, count_tt<type_tuple<Tys ...>, ElemTy>::value + static_cast<std::size_t>(std::is_same<Ty, ElemTy>::value)>
{};

template <typename ElemTy>
struct count_tt<type_tuple<>, ElemTy>
    : std::integral_constant<std::size_t, 0>
{};

// -----------------------------------------------------------------------------------------------------------------------

template <typename TypeTupleTy> struct size_tt;

template <typename ... Tys>
struct size_tt<type_tuple<Tys ...>> : std::integral_constant<std::size_t, sizeof...(Tys)>
{};

// -----------------------------------------------------------------------------------------------------------------------

template <typename TypeTupleTy, typename ElemTy> struct split_tt;

namespace detail
{
template <typename TypeTupleTy, typename ElemTy, typename FirstTupleTy> struct split_tt_helper1;

template <typename Ty, typename ... Tys, typename ElemTy, typename ... FirstTys>
struct split_tt_helper1<type_tuple<Ty, Tys ...>, ElemTy, type_tuple<FirstTys ...>>
    : split_tt_helper1<type_tuple<Tys ...>, ElemTy, type_tuple<FirstTys ..., Ty>>
{};

template <typename Ty, typename ... Tys, typename ... FirstTys>
struct split_tt_helper1<type_tuple<Ty, Tys ...>, Ty, type_tuple<FirstTys ...>>
{
    using first_type  = type_tuple<FirstTys ...>;
    using second_type = type_tuple<Tys ...>;
};

template <typename ElemTy, typename ... FirstTys>
struct split_tt_helper1<type_tuple<>, ElemTy, type_tuple<FirstTys ...>>
{
    using first_type  = type_tuple<>;
    using second_type = type_tuple<FirstTys ...>;
};
} // namespace detail

template <typename ... Tys, typename ElemTy>
struct split_tt<type_tuple<Tys ...>, ElemTy>
    : detail::split_tt_helper1<type_tuple<Tys ...>, ElemTy, type_tuple<>>
{};

// -----------------------------------------------------------------------------------------------------------------------

template <typename TypeTupleTy, typename ElemTy> struct index_of_tt;

static constexpr std::size_t npos = static_cast<std::size_t>(-1);

namespace detail
{
template <typename TypeTupleTy, typename ElemTy, std::size_t Pos> struct index_of_tt_helper1;

template <typename Ty, typename ... Tys, typename ElemTy, std::size_t Pos>
struct index_of_tt_helper1<type_tuple<Ty, Tys ...>, ElemTy, Pos>
    : index_of_tt_helper1<type_tuple<Tys ...>, ElemTy, Pos + 1>
{};

template <typename Ty, typename ... Tys, std::size_t Pos>
struct index_of_tt_helper1<type_tuple<Ty, Tys ...>, Ty, Pos>
    : std::integral_constant<std::size_t, Pos>
{};

template <typename ElemTy, std::size_t Pos>
struct index_of_tt_helper1<type_tuple<>, ElemTy, Pos>
    : std::integral_constant<std::size_t, npos>
{};
} // namespace detail

template <typename ... Tys, typename ElemTy>
struct index_of_tt<type_tuple<Tys ...>, ElemTy>
    : detail::index_of_tt_helper1<type_tuple<Tys ...>, ElemTy, 0>
{};

// -----------------------------------------------------------------------------------------------------------------------

template <typename TypeTupleTy, typename ElemTy> struct remove_tt;

namespace detail
{
template <typename TypeTupleTy, typename ElemTy, typename ResultTupleTy> struct remove_tt_helper1;

template <typename Ty, typename ... Tys, typename ElemTy, typename ... ResultTys>
struct remove_tt_helper1<type_tuple<Ty, Tys ...>, ElemTy, type_tuple<ResultTys ...>>
    : remove_tt_helper1<type_tuple<Tys ...>, ElemTy, type_tuple<ResultTys ..., Ty>>
{};

template <typename Ty, typename ... Tys, typename ... ResultTys>
struct remove_tt_helper1<type_tuple<Ty, Tys ...>, Ty, type_tuple<ResultTys ...>>
    : remove_tt_helper1<type_tuple<Tys ...>, Ty, type_tuple<ResultTys ...>>
{};

template <typename ElemTy, typename ... ResultTys>
struct remove_tt_helper1<type_tuple<>, ElemTy, type_tuple<ResultTys ...>>
{
    using type = type_tuple<ResultTys ...>;
};
} // namespace detail

template <typename ... Tys, typename ElemTy>
struct remove_tt<type_tuple<Tys ...>, ElemTy>
    : detail::remove_tt_helper1<type_tuple<Tys ...>, ElemTy, type_tuple<>>
{};

template <typename TypeTupleTy, typename ElemTy>
using remove_tt_t = typename remove_tt<TypeTupleTy, ElemTy>::type;

// -----------------------------------------------------------------------------------------------------------------------

template <template <typename ...> class VariadicTTy, typename TypeTupleTy> struct make_vttype_tt;

template <template <typename ...> class VariadicTTy, typename ... Tys>
struct make_vttype_tt<VariadicTTy, type_tuple<Tys ...>>
{
    using type = VariadicTTy<Tys ...>;
};

template <template <typename ...> class VariadicTTy, typename TypeTupleTy>
using make_vttype_tt_t = typename make_vttype_tt<VariadicTTy, TypeTupleTy>::type;

// -----------------------------------------------------------------------------------------------------------------------

template <typename TypeTupleTy> struct make_indexer_tt;

namespace detail
{
template <typename TypeTupleTy, std::size_t Idx, typename IdxTupleTy> struct make_indexer_tt_helper1;

template <typename Ty, typename ... Tys, std::size_t Idx, std::size_t ... Idxs>
struct make_indexer_tt_helper1<type_tuple<Ty, Tys ...>, Idx, index_tuple<Idxs ...>>
    : make_indexer_tt_helper1<type_tuple<Tys ...>, Idx + 1, index_tuple<Idxs ..., Idx>>
{};

template <std::size_t Idx, typename IdxTupleTy>
struct make_indexer_tt_helper1<type_tuple<>, Idx, IdxTupleTy>
{
    using type = IdxTupleTy;
};

} // namespace detail

template <typename ... Tys>
struct make_indexer_tt<type_tuple<Tys ...>>
    : detail::make_indexer_tt_helper1<type_tuple<Tys ...>, 0, index_tuple<>>
{};

template <typename TypeTupleTy>
using make_indexer_tt_t = typename make_indexer_tt<TypeTupleTy>::type;

// -----------------------------------------------------------------------------------------------------------------------

namespace detail
{
template <template <typename> class DefaultValSelectorTTy,
          std::size_t DefaultedStartPos,
          std::size_t Idx,
          typename ArgTy>
constexpr auto select_arg_or_default(ArgTy&& arg) -> typename std::decay<ArgTy>::type
{
    return (Idx < DefaultedStartPos)
        ? std::forward<ArgTy>(arg)
        : DefaultValSelectorTTy<typename std::decay<ArgTy>::type>::value;
}

template <template <typename> class DefaultValSelectorTTy,
          std::size_t DefaultedStartPos,
          std::size_t ... Idxs,
          typename ... ArgTys>
constexpr auto make_partially_defaulted_std_tuple(index_tuple<Idxs ...>&&, ArgTys&& ... args)
    -> std::tuple<typename std::decay<ArgTys>::type ...>
{
    return std::make_tuple(
        select_arg_or_default<DefaultValSelectorTTy, DefaultedStartPos, Idxs>(std::forward<ArgTys>(args)) ...);
}
} // namespace detail

template <template <typename> class DefaultValSelectorTTy,
          std::size_t DefaultedStartPos,
          typename ... ArgTys>
constexpr auto make_partially_defaulted_std_tuple(ArgTys&& ... args) -> std::tuple<typename std::decay<ArgTys>::type ...>
{
    return detail::make_partially_defaulted_std_tuple<DefaultValSelectorTTy, DefaultedStartPos>(
        make_indexer_tt_t<type_tuple<ArgTys ...>>(),
        std::forward<ArgTys>(args) ...);
}

// -----------------------------------------------------------------------------------------------------------------------

} // namespace mputils

/// Marker type that separates required selectors from optional ones in kernel selector signature.
struct kd_optional_selector_t {};

template <typename Ty>
struct kd_default_value_selector
{
    static constexpr Ty value = static_cast<Ty>(0);
};

template<typename KernelDataTy, typename OuterTy, std::size_t ReqSelectorCount, typename SelectorsTupleTy>
class kd_selector;

template<typename KernelDataTy, typename OuterTy, std::size_t ReqSelectorCount, typename ... SelectorTys>
class kd_selector<KernelDataTy, OuterTy, ReqSelectorCount, mputils::type_tuple<SelectorTys ...>>
{
    using _selector_types = mputils::type_tuple<SelectorTys...>;
    static_assert(mputils::count_tt<_selector_types, kd_optional_selector_t>::value == 0,
                  "Optional selectors separator can be specified only in template alias. "
                  "Please do not use this class directly - use kd_selector_t alias instead.");
    static_assert(mputils::size_tt<_selector_types>::value > 0,
                  "At least one selector type must be specified.");
    static_assert(ReqSelectorCount <= mputils::size_tt<_selector_types>::value,
                  "Number of required selectors is invalid.");


public:
    using key_type = mputils::make_vttype_tt_t<std::tuple, _selector_types>;

    using hash_type = std::hash<key_type>;
    using mapped_type = KernelDataTy (*)(const OuterTy&);
    using map_type = std::unordered_map<key_type, mapped_type, hash_type>;
    using value_type = typename map_type::value_type;


private:
    map_type _kernel_map;



    template <std::size_t Idx>
    KernelDataTy _get_kernel(mputils::index_tuple<Idx>&&, const OuterTy& outer, const SelectorTys& ... selectors)
    {
        auto value = _kernel_map.find(
            mputils::make_partially_defaulted_std_tuple<kd_default_value_selector, Idx - 1>(selectors ...));
        if (value == _kernel_map.end())
            return _get_kernel(mputils::index_tuple<Idx - 1>(), outer, selectors ...);

        return value->second(outer);
    }

    static KernelDataTy _get_kernel(mputils::index_tuple<ReqSelectorCount>&&, const OuterTy&, const SelectorTys& ...)
    {
        throw std::runtime_error("ERROR: no default element in map for kernel data!!!");
    }

public:
    kd_selector(const std::initializer_list<value_type>& l)
        : _kernel_map(l)
    {}

    KernelDataTy get_kernel(const OuterTy& outer, const SelectorTys& ... selectors)
    {
        return _get_kernel(mputils::index_tuple<sizeof...(SelectorTys) + 1>(), outer, selectors ...);
    }
};

template<typename KernelDataTy, typename OuterTy, typename ... SelectorTys>
using kd_selector_t = kd_selector<KernelDataTy,
                                  OuterTy,
                                  mputils::index_of_tt<mputils::type_tuple<SelectorTys ...>, kd_optional_selector_t>::value != mputils::npos
                                    ? mputils::index_of_tt<mputils::type_tuple<SelectorTys ...>, kd_optional_selector_t>::value
                                    : sizeof...(SelectorTys),
                                  mputils::remove_tt_t<mputils::type_tuple<SelectorTys ...>, kd_optional_selector_t>>;

} }
