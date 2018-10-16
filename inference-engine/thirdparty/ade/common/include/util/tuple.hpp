// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef UTIL_TUPLE_HPP
#define UTIL_TUPLE_HPP

#include <tuple>
#include <utility>
#include <type_traits>

namespace util
{

namespace details
{
template<int...>
struct Seq { };

template<int N, int... S>
struct Gens : Gens<N-1, N-1, S...> { };

template<int... S>
struct Gens<0, S...>
{
  typedef Seq<S...> type;
};

template<int N>
using gen_t = typename Gens<N>::type;

template<std::size_t I = 0, typename F, typename... TupleTypes>
inline auto tupleForeachImpl(const std::tuple<TupleTypes...>& /*tup*/, F&& /*fun*/) -> typename std::enable_if<(I == sizeof...(TupleTypes))>::type
{
}

template<std::size_t I = 0, typename F, typename... TupleTypes>
inline auto tupleForeachImpl(const std::tuple<TupleTypes...>& tup, F&& fun) -> typename std::enable_if<(I < sizeof...(TupleTypes))>::type
{
    fun(std::get<I>(tup));
    tupleForeachImpl<I + 1, F, TupleTypes...>(tup, std::forward<F>(fun));
}

template<std::size_t I = 0, typename F, typename... TupleTypes>
inline auto tupleForeachImpl(std::tuple<TupleTypes...>& tup, F&& fun) -> typename std::enable_if<(I < sizeof...(TupleTypes))>::type
{
    fun(std::get<I>(tup));
    tupleForeachImpl<I + 1, F, TupleTypes...>(tup, std::forward<F>(fun));
}

template<typename T>
struct fix_rval
{
    using type = typename std::remove_reference<T>::type;
};

template<typename T>
struct fix_rval<T&>
{
    using type = T&;
};


template<typename T>
using fix_rval_t = typename fix_rval<T>::type;

}

inline namespace Range
{

/// Converts rvalue refs to values
/// but preserves lvalue refs
template<typename... Elements>
inline auto tuple_remove_rvalue_refs(Elements&&... args)->
        std::tuple<details::fix_rval_t<Elements>...>
{
    return std::tuple<details::fix_rval_t<Elements>...>(std::forward<Elements>(args)...);
}

template<typename F, typename... TupleTypes>
inline void tupleForeach(const std::tuple<TupleTypes...>& tup, F&& fun)
{
    details::tupleForeachImpl(tup, std::forward<F>(fun));
}

template<typename F, typename... TupleTypes>
inline void tupleForeach(std::tuple<TupleTypes...>& tup, F&& fun)
{
    details::tupleForeachImpl(tup, std::forward<F>(fun));
}

}
}

#endif // UTIL_TUPLE_HPP
