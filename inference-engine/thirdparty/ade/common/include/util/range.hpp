// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef UTIL_RANGE_HPP
#define UTIL_RANGE_HPP

#include <iterator>
#include <utility>

#include "util/assert.hpp"
#include "util/type_traits.hpp"

namespace util
{

namespace details
{
struct RangeIncrementer
{
    template<typename Rng>
    void operator()(Rng& range) const
    {
        range.popFront();
    }
};
struct RangeChecker
{
    bool empty = false;

    template<typename Rng>
    void operator()(Rng& range)
    {
        empty = empty || range.empty();
    }
};
struct RangeCleaner
{
    template<typename Rng>
    void operator()(Rng& range) const
    {
        range.clear();
    }
};

//SFINAE and decltype doesn't work wery well together
// so this additional wrapper is needed
template<typename T>
struct range_size_wrapper_helper
{
    using type = decltype(size(std::declval<T>()));
};

template<typename T>
inline auto range_size_wrapper(const T& r)
->typename range_size_wrapper_helper<T>::type
{
    return size(r);
}

template<typename T>
class has_size_fun
{
    using y = char;
    using n = long;

    template<typename C>
    static y test(decltype(size(std::declval<C>()))*);
    template<typename C>
    static n test(...);
public:
    static const constexpr bool value = (sizeof(test<T>(0)) == sizeof(y));
};

}
inline namespace Range
{
template<typename BeginT, typename EndT = BeginT>
struct IterRange
{
    BeginT beginIter;
    EndT   endIter;

    BeginT begin() const
    {
        return beginIter;
    }

    EndT end() const
    {
        return endIter;
    }

    bool empty() const
    {
        return beginIter == endIter;
    }

    void popFront()
    {
        ASSERT(!empty());
        ++beginIter;
    }

    auto front() -> decltype(*beginIter)
    {
        ASSERT(!empty());
        return *beginIter;
    }

    auto front() const -> decltype(*beginIter)
    {
        ASSERT(!empty());
        return *beginIter;
    }

    bool operator==(const IterRange<BeginT, EndT>& rhs) const
    {
        return beginIter == rhs.beginIter && endIter == rhs.endIter;
    }

    bool operator!=(const IterRange<BeginT, EndT>& rhs) const
    {
        return !(*this == rhs);
    }

    template<typename I1 = BeginT, typename I2 = EndT,
             util::enable_b_t<(sizeof(std::declval<I2>() - std::declval<I1>()) > 0)> = true> //SFINAE
    auto size() const
    ->decltype(std::declval<I2>() - std::declval<I1>())
    {
        return endIter - beginIter;
    }

    // TODO: bidirectional and random access ranges
};

template<typename BeginT, typename EndT>
inline auto size(const IterRange<BeginT, EndT>& range)
->decltype(range.size())
{
    return range.size();
}

template<typename T>
inline auto toRange(T&& val) -> IterRange<decltype(std::begin(val)), decltype(std::end(val))>
{
    return {std::begin(val), std::end(val)};
}

template<typename T>
inline auto toRangeReverse(T&& val) -> IterRange<decltype(val.rbegin()), decltype(val.rbegin())>
{
    // TODO: use c++14 std::rbegin, std::rend
    return {val.rbegin(), val.rend()};
}

template<typename Iter>
inline auto toRange(const std::pair<Iter,Iter>& val) -> IterRange<decltype(val.first), decltype(val.second)>
{
    return {val.first, val.second};
}

template<typename Iter>
inline auto toRange(Iter&& val, std::size_t count) -> IterRange<remove_reference_t<Iter>, remove_reference_t<Iter>>
{
    return {std::forward<Iter>(val), std::next(val, count)};
}

template<typename T>
inline auto index(T&& val)->decltype(std::get<0>(std::forward<T>(val)))
{
    return std::get<0>(std::forward<T>(val));
}

template<int I = 0, typename T>
inline auto value(T&& val)->decltype(std::get<I + 1>(std::forward<T>(val)))
{
    static_assert(I >= 0,"Invalid I");
    return std::get<I + 1>(std::forward<T>(val));
}

template<typename Range>
inline void advance_range(Range&& range)
{
    range.popFront();
}

}
}

#endif // UTIL_RANGE_HPP
