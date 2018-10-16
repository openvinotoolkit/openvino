// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef UTIL_ANY_HPP
#define UTIL_ANY_HPP

#include <memory>
#include <type_traits>
#include <utility>
#include <typeinfo>

#include "assert.hpp"

//TODO - drop any from ADE metadata, using only typed graph
//and to introduce bind() instead

namespace internal
{
    template <class T, class Source>
    T down_cast(Source operand)
    {
#if defined(__GXX_RTTI) || defined(_CPPRTTI)
       return dynamic_cast<T>(operand);
#else
    #warning used static cast instead of dynamic because RTTI is disabled
       return static_cast<T>(operand);
#endif
    }
}

namespace util
{
   class bad_any_cast : public std::bad_cast
   {
   public:
       virtual const char* what() const noexcept override
       {
           return "Bad any cast";
       }
   };

   //modeled against C++17 std::any
   class any
   {
   private:
      struct holder;
      using holder_ptr = std::unique_ptr<holder>;
      struct holder
      {
         virtual holder_ptr clone() = 0;
         virtual ~holder() = default;
      };

      template <typename value_t>
      struct holder_impl : holder
      {
         value_t v;
         template<typename arg_t>
         holder_impl(arg_t&& a) : v(std::forward<arg_t>(a)) {}
         holder_ptr clone() override { return holder_ptr(new holder_impl (v));}
      };

      holder_ptr hldr;
   public:
      template<class value_t>
      any(value_t&& arg) :  hldr(new holder_impl<typename std::decay<value_t>::type>( std::forward<value_t>(arg))) {}

      any(any const& src) : hldr( src.hldr ? src.hldr->clone() : nullptr) {}
      //simple hack in order not to write enable_if<not any> for the template constructor
      any(any & src) : any (const_cast<any const&>(src)) {}

      any()       = default;
      any(any&& ) = default;

      any& operator=(any&&) = default;

      any& operator=(any const& src)
      {
         any copy(src);
         swap(*this, copy);
         return *this;
      }

      template<class value_t>
      friend value_t* any_cast(any* operand);

      template<class value_t>
      friend const value_t* any_cast(const any* operand);

      friend void swap(any & lhs, any& rhs)
      {
         swap(lhs.hldr, rhs.hldr);
      }

   };

   template<class value_t>
   value_t* any_cast(any* operand)
   {
      auto casted = internal::down_cast<any::holder_impl<typename std::decay<value_t>::type> *>(operand->hldr.get());
      if (casted){
         return & (casted->v);
      }
      return nullptr;
   }

   template<class value_t>
   const value_t* any_cast(const any* operand)
   {
      auto casted = internal::down_cast<any::holder_impl<typename std::decay<value_t>::type> *>(operand->hldr.get());
      if (casted){
         return & (casted->v);
      }
      return nullptr;
   }

   template<class value_t>
   value_t& any_cast(any& operand)
   {
      auto ptr = any_cast<value_t>(&operand);
      if (ptr)
      {
         return *ptr;
      }

      throw_error(bad_any_cast());
   }


   template<class value_t>
   const value_t& any_cast(const any& operand)
   {
      auto ptr = any_cast<value_t>(&operand);
      if (ptr)
      {
         return *ptr;
      }

      throw_error(bad_any_cast());
   }
}

#endif // UTIL_ANY_HPP
