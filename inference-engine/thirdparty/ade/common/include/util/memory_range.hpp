// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef UTIL_MEMORYRANGE_HPP
#define UTIL_MEMORYRANGE_HPP

#include <type_traits>
#include <cstring> //size_t
#include <algorithm>
#include <array>

#include "util/type_traits.hpp"
#include "util/assert.hpp"

namespace util
{

/// Non owning view over the data in another container
/// T can be cv-qualified object or cv-qualified void
template<typename T>
struct MemoryRange
{
   static_assert(std::is_object<T>::value || std::is_void<T>::value, "Invalid type");
   using value_type = T;
   T*     data = nullptr;
   size_t size = 0;

   MemoryRange()                              = default;
   MemoryRange(const MemoryRange&)            = default;
   MemoryRange& operator=(const MemoryRange&) = default;

   MemoryRange(T* data_, size_t size_):
      data(data_), size(size_)
   {
      ASSERT((nullptr != data) || (0 == size)); //size must be 0 for null data
   }

   /// Slice view
   /// if T is void start and newSize addressable in bytes
   /// otherwise in elements
   MemoryRange<T> Slice(size_t start, size_t newSize) const
   {
      ASSERT(nullptr != data);
      ASSERT((start + newSize) <= size);
      using temp_t = typename std::conditional<std::is_void<T>::value, char, T>::type;
      return MemoryRange<T>(static_cast<temp_t*>(data) + start, newSize);
   }

   template<typename Dummy = void>
   typename std::enable_if<!std::is_void<T>::value, typename std::conditional<true, T*, Dummy>::type >::type
   begin()
   {
      return data;
   }

   template<typename Dummy = void>
   typename std::enable_if<!std::is_void<T>::value, typename std::conditional<true, T*, Dummy>::type >::type
   end()
   {
      return data + size;
   }

   template<typename Dummy = void>
   typename std::enable_if<!std::is_void<T>::value, typename std::conditional<true, const T*, Dummy>::type >::type
   begin() const
   {
      return data;
   }

   template<typename Dummy = void>
   typename std::enable_if<!std::is_void<T>::value, typename std::conditional<true, const T*, Dummy>::type >::type
   end() const
   {
      return data + size;
   }

   template<typename Dummy = T> // hack for sfinae
   typename std::enable_if<(sizeof(Dummy) > 0), Dummy& >::type
   operator[](size_t index)
   {
      ASSERT(index < size);
      return data[index];
   }

   template<typename Dummy = T> // hack for sfinae
   typename std::enable_if<(sizeof(Dummy) > 0), const Dummy& >::type
   operator[](size_t index) const
   {
      ASSERT(index < size);
      return data[index];
   }

   template<typename NewT>
   MemoryRange<NewT> reinterpret() const
   {
      const size_t elem_size     = sizeof(util::conditional_t< std::is_void<T>::value, char, T >);
      const size_t elem_size_new = sizeof(util::conditional_t< std::is_void<NewT>::value, char, NewT >);
      const size_t newSize = (size * elem_size) / elem_size_new;
      return MemoryRange<NewT>(static_cast<NewT*>(data), newSize);
   }

   template<typename Dummy = T> // hack for sfinae
   typename std::enable_if<(sizeof(Dummy) > 0), Dummy& >::type
   front()
   {
      ASSERT(size > 0);
      return data[0];
   }

   template<typename Dummy = T> // hack for sfinae
   typename std::enable_if<(sizeof(Dummy) > 0), const Dummy& >::type
   front() const
   {
      ASSERT(size > 0);
      return data[0];
   }

   template<typename Dummy = T> // hack for sfinae
   typename std::enable_if<(sizeof(Dummy) > 0), Dummy& >::type
   back()
   {
      ASSERT(size > 0);
      return data[size - 1];
   }

   template<typename Dummy = T> // hack for sfinae
   typename std::enable_if<(sizeof(Dummy) > 0), const Dummy& >::type
   back() const
   {
      ASSERT(size > 0);
      return data[size - 1];
   }

   bool empty() const
   {
       return 0 == size;
   }

   void popFront()
   {
       ASSERT(!empty());
       *this = Slice(1, size - 1);
   }
};

template<typename T>
inline MemoryRange<T> memory_range(T* ptr, const std::size_t size)
{
    return {ptr, size};
}

template<typename T, std::size_t size_>
inline MemoryRange<T> memory_range(T (&range)[size_])
{
    return memory_range(&range[0], size_);
}

template<typename T, std::size_t size_>
inline MemoryRange<T> memory_range(std::array<T, size_>& arr)
{
    return memory_range(arr.data(), size_);
}

template<typename T>
inline T* data(const MemoryRange<T>& range)
{
    return range.data;
}

template<typename T, std::size_t size>
inline T* data(T (&range)[size])
{
    return &(range[0]);
}

template<typename T>
inline std::size_t size(const MemoryRange<T>& range)
{
    return range.size;
}

template<typename T, std::size_t size_>
inline std::size_t size(T (&)[size_])
{
    return size_;
}

template<typename T>
inline bool operator==(const MemoryRange<T>& range, std::nullptr_t)
{
    return range.data == nullptr;
}

template<typename T>
inline bool operator==(std::nullptr_t, const MemoryRange<T>& range)
{
    return range.data == nullptr;
}

template<typename T>
inline bool operator!=(const MemoryRange<T>& range, std::nullptr_t)
{
    return range.data != nullptr;
}

template<typename T>
inline bool operator!=(std::nullptr_t, const MemoryRange<T>& range)
{
    return range.data != nullptr;
}

template<typename T>
inline MemoryRange<T> slice(const MemoryRange<T>& range, const std::size_t start, const std::size_t newSize)
{
    return range.Slice(start, newSize);
}

template<typename T, std::size_t size>
inline MemoryRange<T> slice(T (&range)[size], const std::size_t start, const std::size_t newSize)
{
    return memory_range(&range[0], size).Slice(start, newSize);
}

template<typename SrcRange, typename DstRange>
inline auto raw_copy(const SrcRange& src, DstRange&& dst)
-> decltype(slice(dst, size(src), size(dst) - size(src)))
{
    static_assert(std::is_same< util::decay_t<decltype(*data(src))>,
                                util::decay_t<decltype(*data(dst))> >::value, "Types must be same");
    static_assert(std::is_pod< util::decay_t<decltype(*data(src))> >::value, "Types must be pod");
    static_assert(sizeof(src[std::size_t{}]) == sizeof(dst[std::size_t{}]), "Size mismatch");
    const auto src_size = size(src);
    const auto dst_size = size(dst);
    ASSERT(nullptr != src);
    ASSERT(nullptr != dst);
    ASSERT(dst_size >= src_size);
    auto src_data = data(src);
    auto dst_data = data(dst);
    // Check overlap
    ASSERT(((src_data < dst_data) || (src_data >= (dst_data + dst_size))) &&
           ((dst_data < src_data) || (dst_data >= (src_data + src_size))));
    std::copy_n(src_data, src_size, dst_data);

    return slice(dst, size(src), size(dst) - size(src));
}

inline MemoryRange<void> raw_copy(const MemoryRange<void>& src, MemoryRange<void> dst)
{
    return raw_copy(src.reinterpret<const char>(), dst.reinterpret<char>()).reinterpret<void>();
}

inline MemoryRange<void> raw_copy(const MemoryRange<const void>& src, MemoryRange<void> dst)
{
    return raw_copy(src.reinterpret<const char>(), dst.reinterpret<char>()).reinterpret<void>();
}

} // util

#endif // UTIL_MEMORYRANGE_HPP
