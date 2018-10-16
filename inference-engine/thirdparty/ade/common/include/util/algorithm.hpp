// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef UTIL_ALGORITHM_HPP
#define UTIL_ALGORITHM_HPP

#include <algorithm>
#include <utility>
#include <iterator>
#include "util/type_traits.hpp"
#include "util/checked_cast.hpp"

namespace util
{
template <typename container_t, typename predicate_t>
inline bool any_of(container_t&& c, predicate_t&& p)
{
    return std::any_of(std::begin(c), std::end(c), std::forward<predicate_t>(p));
}

template <typename container_t, typename predicate_t>
inline bool all_of(container_t&& c, predicate_t&& p)
{
    return std::all_of(std::begin(c), std::end(c), std::forward<predicate_t>(p));
}

template<typename container_t, typename predicate_t>
inline decltype(std::begin(std::declval<container_t>())) find_if(container_t&& c, predicate_t&& p)
{
    return std::find_if(std::begin(c), std::end(c), std::forward<predicate_t>(p));
}

template<typename container_t, typename value_t>
inline decltype(std::begin(std::declval<container_t>())) find(container_t&& c, const value_t& val)
{
    return std::find(std::begin(c), std::end(c), val);
}

template<typename C, typename T>
inline bool contains(const C& cont, const T& val)
{
    return cont.end() != cont.find(val);
}

template<typename container_t, typename iterator_t>
void unstable_erase(container_t&& c, iterator_t&& it)
{
    *it = std::move(c.back());
    c.pop_back();
}

template<typename container_t, typename output_iterator_t>
inline remove_reference_t<output_iterator_t> copy(container_t &&c, output_iterator_t &&it)
{
    return std::copy(std::begin(c), std::end(c), std::forward<output_iterator_t>(it));
}

template<typename container_t, typename output_iterator_t, typename predicate_t>
inline remove_reference_t<output_iterator_t> copy_if(container_t &&c, output_iterator_t &&it, predicate_t&& p)
{
    return std::copy_if(std::begin(c), std::end(c), std::forward<output_iterator_t>(it), std::forward<predicate_t>(p));
}

template<typename container_t, typename output_iterator_t, typename predicate_t>
inline remove_reference_t<output_iterator_t> transform(container_t &&c, output_iterator_t &&it, predicate_t&& p)
{
    return std::transform(std::begin(c), std::end(c), std::forward<output_iterator_t>(it), std::forward<predicate_t>(p));
}

template<typename container_t, typename predicate_t>
inline decltype(std::begin(std::declval<container_t>())) max_element(container_t&& c, predicate_t&& p)
{
    return std::max_element(std::begin(c), std::end(c), std::forward<predicate_t>(p));
}

template<typename IterIn, typename IterOut>
inline void convert(IterIn inFirst, IterIn inLast, IterOut outFirst)
{
    typedef typename std::iterator_traits<IterIn>::value_type InT;
    typedef typename std::iterator_traits<IterOut>::value_type OutT;
    std::transform(inFirst, inLast, outFirst, [](InT v) {
        return checked_cast<OutT>(v);
    });
}

template<typename IterIn, typename N, typename IterOut>
inline void convert_n(IterIn inFirst, N n, IterOut outFirst)
{
    util::convert(inFirst, inFirst+n, outFirst);
}

template<typename container_t, typename value_t>
void fill(container_t&& c, value_t&& val)
{
    std::fill(std::begin(c), std::end(c), std::forward<value_t>(val));
}

namespace details
{

template<std::size_t I, typename Target, typename First, typename... Remaining>
struct type_list_index_helper
{
    static const constexpr bool is_same = std::is_same<Target, First>::value;
    static const constexpr std::size_t value =
            std::conditional<is_same, std::integral_constant<std::size_t, I>, type_list_index_helper<I + 1, Target, Remaining...>>::type::value;
};

template<std::size_t I, typename Target, typename First>
struct type_list_index_helper<I, Target, First>
{
    static_assert(std::is_same<Target, First>::value, "Type not found");
    static const constexpr std::size_t value = I;
};

}

template<typename Target, typename... Types>
struct type_list_index
{
    static const constexpr std::size_t value = ::util::details::type_list_index_helper<0, Target, Types...>::value;
};
} // namespace util

#endif // UTIL_ALGORITHM_HPP
