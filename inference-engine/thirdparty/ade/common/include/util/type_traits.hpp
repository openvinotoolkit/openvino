// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef UTIL_TYPE_TRAITS_HPP
#define UTIL_TYPE_TRAITS_HPP

#include <type_traits>

namespace util {

template<bool value>
using bool_c = std::integral_constant<bool, value>;

template<typename T>
using not_ = bool_c<!T::value>;

template<typename...>
struct or_;

template<>
struct or_<> : std::true_type {};

template<typename T>
struct or_<T> : T {};

template<typename T0, typename T1>
struct or_<T0, T1> : bool_c< T0::value || T1::value > {};

template<typename T0, typename ...T>
struct or_<T0, T...> : or_< T0, or_<T...> > {};

template<typename...>
struct and_;

template<>
struct and_<> : std::true_type {};

template<typename T>
struct and_<T> : T {};

template<typename T0, typename T1>
struct and_<T0, T1> : bool_c< T0::value && T1::value > {};

template<typename T0, typename ...T>
struct and_<T0, T...> : and_< T0, and_<T...> > {};


template<typename T, typename ...Ts>
struct is_one_of : or_< std::is_same<T,Ts>... > {};

template<bool v>
using enable_b_t = typename std::enable_if< v, bool >::type;

template<typename ...Ts>
using require_t = enable_b_t< and_<Ts...>::value >;

template<typename T, typename ...Ts>
using enable_t = typename std::enable_if< and_<Ts...>::value, T >::type;

template<bool B, class T = void >
using enable_if_t = typename std::enable_if<B, T>::type;

template<typename T>
using decay_t = typename std::decay<T>::type;

template< class T >
using remove_reference_t = typename std::remove_reference<T>::type;

template< class T >
using remove_pointer_t = typename std::remove_pointer<T>::type;

template< bool B, class T, class F >
using conditional_t = typename std::conditional<B,T,F>::type;

template<typename... Types>
using common_type_t = typename std::common_type<Types...>::type;

template<class T>
using result_of_t = typename std::result_of<T>::type;

} // namespace util

#endif // UTIL_TYPE_TRAITS_HPP
