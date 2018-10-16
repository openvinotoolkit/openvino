// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef UTIL_CHAIN_RANGE_HPP
#define UTIL_CHAIN_RANGE_HPP

#include <utility>
#include <type_traits>

#include "util/range.hpp"
#include "util/type_traits.hpp"
#include "util/assert.hpp"
#include "util/range_iterator.hpp"

namespace util
{
inline namespace Range
{
template<typename Range1, typename Range2>
struct ChainRange : public IterableRange<ChainRange<Range1, Range2>>
{
    Range1 range1;
    Range2 range2;

    ChainRange() = default;
    ChainRange(const ChainRange&) = default;
    ChainRange(ChainRange&&) = default;
    ChainRange& operator=(const ChainRange&) = default;
    ChainRange& operator=(ChainRange&&) = default;

    template<typename R1, typename R2>
    ChainRange(R1&& r1, R2&& r2):
        range1(std::forward<R1>(r1)),
        range2(std::forward<R2>(r2))
    {

    }

    // common_type will strip references from types but we want to retain them
    // if types exactly the same
    template<typename First, typename Second>
    using return_type_helper_t = typename std::conditional<
                                                           std::is_same<First, Second>::value,
                                                           First,
                                                           util::common_type_t<First, Second>
                                                          >::type;

    bool empty() const
    {
        return range1.empty() && range2.empty();
    }

    void popFront()
    {
        ASSERT(!empty());
        if (!range1.empty())
        {
            range1.popFront();
        }
        else
        {
            range2.popFront();
        }
    }

    auto front()
    ->return_type_helper_t<decltype(this->range1.front()),decltype(this->range2.front())>
    {
        ASSERT(!empty());
        if (!range1.empty())
        {
            return range1.front();
        }
        return range2.front();
    }

    auto front() const
    ->return_type_helper_t<decltype(this->range1.front()),decltype(this->range2.front())>
    {
        ASSERT(!empty());
        if (!range1.empty())
        {
            return range1.front();
        }
        return range2.front();
    }

    template<typename R1 = Range1, typename R2 = Range2,
             util::enable_b_t<(details::has_size_fun<R1>::value &&
                               details::has_size_fun<R2>::value)> = true> //SFINAE
    auto size() const
    ->util::common_type_t<decltype(details::range_size_wrapper(std::declval<R1>())),
                          decltype(details::range_size_wrapper(std::declval<R2>()))>
    {
        return details::range_size_wrapper(range1) + details::range_size_wrapper(range2);
    }

    template<typename R1 = Range1, typename R2 = Range2,
             util::enable_b_t<(sizeof(std::declval<R1>()[0]) > 0 &&
                               sizeof(std::declval<R2>()[0]) > 0)> = true> //SFINAE
    auto operator[](std::size_t index)
    ->return_type_helper_t<decltype(std::declval<R1>()[0]),decltype(std::declval<R2>()[0])>
    {
        ASSERT(index < size());
        const auto size1 = details::range_size_wrapper(range1);
        if (index < size1)
        {
            return range1[index];
        }
        return range2[index - size1];
    }

    template<typename R1 = Range1, typename R2 = Range2,
             util::enable_b_t<(sizeof(std::declval<const R1>()[0]) > 0 &&
                               sizeof(std::declval<const R2>()[0]) > 0)> = true> //SFINAE
    auto operator[](std::size_t index) const
    ->return_type_helper_t<decltype(std::declval<const R1>()[0]),decltype(std::declval<const R2>()[0])>
    {
        ASSERT(index < size());
        const auto size1 = details::range_size_wrapper(range1);
        if (index < size1)
        {
            return range1[index];
        }
        return range2[index - size1];
    }
};

template<typename Range1, typename Range2>
inline auto size(const ChainRange<Range1, Range2>& range)
->decltype(range.size())
{
    return range.size();
}

template <typename Range1, typename... Ranges>
struct chain_type;

template <typename Range1>
struct chain_type<Range1> { using type = remove_reference_t<Range1>; };

template <typename Range1, typename Range2, typename... Ranges>
struct chain_type<Range1, Range2, Ranges...>
{
    using type = ChainRange<remove_reference_t<Range1>,
                            typename chain_type<remove_reference_t<Range2>, remove_reference_t<Ranges>...>::type>;
};

template<typename Range>
auto chain(Range&& r)
->typename chain_type<Range>::type
{
    return std::forward<Range>(r);
}

template<typename Range1, typename Range2, typename... Ranges>
auto chain(Range1&& r1, Range2&& r2, Ranges&&... ranges)
->typename chain_type<Range1, Range2, Ranges...>::type
{
    using RangeT = typename chain_type<Range1, Range2, Ranges...>::type;
    return RangeT(std::forward<Range1>(r1), chain(std::forward<Range2>(r2), std::forward<Ranges>(ranges)...));
}

}
}

#endif // UTIL_CHAIN_RANGE_HPP
