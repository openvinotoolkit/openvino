/*
 INTEL CONFIDENTIAL
 Copyright 2017 Intel Corporation.

 The source code contained or described herein and all documents related
 to the source code ("Material") are owned by Intel Corporation or its suppliers
 or licensors. Title to the Material remains with Intel Corporation or its suppliers
 and licensors. The Material may contain trade secrets and proprietary
 and confidential information of Intel Corporation and its suppliers and licensors,
 and is protected by worldwide copyright and trade secret laws and treaty provisions.
 No part of the Material may be used, copied, reproduced, modified, published,
 uploaded, posted, transmitted, distributed, or disclosed in any way without Intel's
 prior express written permission.

 No license under any patent, copyright, trade secret or other intellectual
 property right is granted to or conferred upon you by disclosure or delivery
 of the Materials, either expressly, by implication, inducement, estoppel
 or otherwise. Any license under such intellectual property rights must
 be express and approved by Intel in writing.

 Unless otherwise agreed by Intel in writing, you may not remove or alter this notice
 or any other notice embedded in Materials by Intel or Intel's suppliers or licensors
 in any way.
*/

#pragma once

#include "common.h"

namespace GNA
{

// Generic not defined Address class
template<typename T> class Address;

// Address class for operating on const pointers
template<typename T> class Address<T*const>
{
public:
    Address() = default;
    Address(void * const value) :
        buffer(const_cast<void*>(value))
    {}
    Address(const void * const value) :
        buffer(const_cast<void*>(value))
    {}
    Address(const T * value) :
        buffer(const_cast<T*>(value))
    {}
    Address(const Address& address) :
        buffer(address.buffer)
    {}
    template<class C> Address(const Address<C*>& address) :
        buffer(const_cast<C*>(address.Get()))
    {}
    template<class C> Address(const Address<C*const>& address) :
        buffer(address.Get())
    {}
    ~Address() = default;

    Address operator+(const uint32_t& right) const
    {
        Address plus(*this);
        plus.buffer = plus.Get<T>() + right;
        return plus;
    }
    Address operator-(const uint32_t& right) const
    {
        Address left(*this);
        left.buffer = left.Get<T>() - right;
        return left;
    }

    bool operator ==(const std::nullptr_t &right) const
    {
        return right == buffer;
    }

    explicit operator bool() const
    {
        return (nullptr != buffer);
    }

    bool operator!() const
    {
        return (nullptr == buffer);
    }

    template<class X> operator X* () const
    {
        return static_cast<X *>(buffer);
    }

    template<class X> X * Get() const
    {
        return static_cast<X *>(buffer);
    }

    T * Get() const
    {
        return static_cast<T *>(buffer);
    }

    T& operator*() const
    {
        return *(static_cast<T*>(buffer));
    }

    template<class X> uint32_t GetOffset(const Address<X*const>& base) const
    {
        if (this->operator! ()) return 0;
        return static_cast<uint32_t>(
            reinterpret_cast<uintptr_t>(
              reinterpret_cast<uint8_t*>(this->Get<uint8_t>() - base.template Get<uint8_t>())));
    }

    bool InRange(void *memory, uint32_t memorySize) const
    {
        if (buffer < memory)
        {
            return false;
        }

        auto memoryEnd = reinterpret_cast<uint8_t*>(memory) + memorySize;
        if (buffer >= memoryEnd)
        {
            return false;
        }

        return true;
    }

protected:
    void * buffer = nullptr;
};

// Address class for operating on non-const pointers
template<typename T> class Address<T *> : public Address<T*const>
{
public:
    Address() = default;
    Address(void * value) :
        Address<T*const>(value)
    {}
    Address(const void * value) :
        Address<T*const>(value)
    {}
    template<class C> Address(const Address<C*>& address) :
        Address<T*const>(address)
    {}
    template<class C> Address(const Address<C*const>& address) :
        Address<T*const>(address.Get())
    {}

    template<class X> X * Get() const
    {
        return static_cast<X *>(this->buffer);
    }

    T * Get() const
    {
        return static_cast<T*>(this->buffer);
    }

    T * operator->() const
    {
        return static_cast<T * const>(this->buffer);
    }

    const Address operator++(int)
    {
        Address tmp{ *this };
        this->buffer = this->Get() + 1;
        return tmp;
    }
    const Address& operator-=(const uint32_t& right)
    {
        this->buffer = this->Get() - right;
        return *this;
    }
    const Address& operator+=(const uint32_t& right)
    {
        this->buffer = this->Get() + right;
        return *this;
    }
    const Address& operator =(const Address& right)
    {
        this->buffer = right.buffer;
        return *this;
    }
    const Address& operator =(const T& right)
    {
        *this->buffer = right;
        return *this;
    }
};

// Address Aliases
using BaseAddress = Address<uint8_t * const>;

// Functor for getting buffer offset for HW
using GetHwOffset = std::function<uint32_t(const BaseAddress&)>;

}
