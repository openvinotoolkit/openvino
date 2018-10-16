// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef UTIL_IOTA_RANGE_HPP
#define UTIL_IOTA_RANGE_HPP

#include <type_traits>
#include <cassert>
#include <cinttypes>
#include <limits>

namespace util
{

inline namespace Range
{

template<typename T, std::int32_t step = 0>
struct IotaRange
{
   static_assert(std::is_integral<T>::value,"T must be integral");

   inline void check() const
   {
      if ( step > 0)
      {
         assert(to >= from);
      }
      else if (step < 0)
      {
         assert(from >= to);
      }
      else assert(!"Zero step");

      assert(0 == ((to - from) % step));
   }

   struct iterator
   {
      inline bool operator==(iterator const& other) const
      {
         return value == other.value;
      }

      inline bool operator!=(iterator const& other) const
      {
         return value != other.value;
      }

      inline T const& operator*() const
      {
         return value;
      }

      inline iterator& operator++()
      {
         value += step;
         return *this;
      }

      T value;
   };

   bool empty() const
   {
       return to == from;
   }

   const T& front() const
   {
       assert(!empty());
       return from;
   }

   void popFront()
   {
       assert(!empty());
       from += step;
   }

   inline iterator begin() const
   {
      check();
      return {from};
   }

   inline iterator end() const
   {
      check();
      return {to};
   }

   bool operator==(const IotaRange<T,step>& rhs) const
   {
       return from == rhs.from && to == rhs.to;
   }
   bool operator!=(const IotaRange<T,step>& rhs) const
   {
       return !(*this == rhs);
   }

   /*const*/ T from;
   const T to;
};

template<typename T>
struct IotaRange<T,0>
{
   static_assert(std::is_integral<T>::value,"T must be integral");

   inline void check() const
   {
      if ( step > 0)
      {
         assert(to >= from);
      }
      else if (step < 0)
      {
         assert(from >= to);
      }
      else assert(!"Zero step");

      assert(0 == ((to - from) % step));
   }

   struct iterator
   {
      inline bool operator==(iterator const& other) const
      {
         assert(step == other.step);
         return value == other.value;
      }

      inline bool operator!=(iterator const& other) const
      {
         assert(step == other.step);
         return value != other.value;
      }

      inline T const& operator*() const
      {
         return value;
      }

      inline iterator& operator++()
      {
         value += step;
         return *this;
      }

      T value;
      T step;
   };

   bool empty() const
   {
       return to == from;
   }

   const T& front() const
   {
       assert(!empty());
       return from;
   }

   void popFront()
   {
       assert(!empty());
       from += step;
   }

   inline iterator begin() const
   {
      check();
      return {from, step};
   }

   inline iterator end() const
   {
      check();
      return {to, step};
   }

   bool operator==(const IotaRange<T,0>& rhs) const
   {
       return from == rhs.from && to == rhs.to && step == rhs.step;
   }
   bool operator!=(const IotaRange<T,0>& rhs) const
   {
       return !(*this == rhs);
   }

   /*const*/ T from;
   const T to;
   const T step;
};

template<typename T>
inline IotaRange<T, 1> iota()
{ return {   0, std::numeric_limits<T>::max()}; }

template<typename T>
inline IotaRange<T, 1> iota(const T to)
{ return {   0, to}; }

template<typename T>
inline IotaRange<T, 1> iota(const T from, const T to)
{ return {from, to}; }

template<typename T>
inline IotaRange<T>    iota(const T from, const T to, const T step)
{ return {from, to, step}; }

}
}


#endif // UTIL_IOTA_RANGE_HPP
