// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef UTIL_CALL_ONCE_HPP
#define UTIL_CALL_ONCE_HPP

#include <type_traits>
#include <mutex>

namespace util
{
// Special helper for thread-safe call once per graph execution
class CallOnce
{
   using Storage = typename std::aligned_storage<sizeof(std::once_flag), alignof(std::once_flag)>::type;

   Storage  m_flagStorage;

   CallOnce(const CallOnce&)             = delete;
   CallOnce& operator=(const CallOnce&)  = delete;
   CallOnce(const CallOnce&&)            = delete;
   CallOnce& operator=(const CallOnce&&) = delete;

public:

   CallOnce() { reset(); }

   template<typename F> void operator()(F&& f)
   {
      std::call_once(*reinterpret_cast<std::once_flag*>(&m_flagStorage), std::forward<F>(f));
   }

   void reset()
   {
      new (&m_flagStorage) std::once_flag;
   }
};

} // namespace DMIP

#endif // UTIL_CALL_ONCE_HPP
