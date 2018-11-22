/*
// Copyright (c) 2016 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <atomic>
#include <type_traits>

namespace cldnn
{

template <class T>
struct refcounted_obj_ptr;

/**
 * \brief Base class for all reference counted pointers aka PIMPL implementations
 */
// TODO refine this code for multithreading support
template<class T>
class refcounted_obj
{
public:
    using ptr = refcounted_obj_ptr<typename std::remove_const<T>::type>;
    using cptr = refcounted_obj_ptr<typename std::add_const<T>::type>;

    refcounted_obj()
        : _ref_count(1)
    {}

    virtual ~refcounted_obj() = default;

    void add_ref() const
    {
        ++_ref_count;
    }

    void release() const
    {
        if ((--_ref_count) == 0) delete static_cast<const T*>(this);
    }

    int get_ref_count() const
    {
        return _ref_count;
    }

private:
    mutable std::atomic_int _ref_count;
};

template<class T>
struct refcounted_obj_ptr
{
    template <class U = T>
    refcounted_obj_ptr(T* ptr, bool add_ref = true) : _ptr(ptr)
    {
        static_assert(std::is_base_of<refcounted_obj<typename std::remove_const<U>::type>, U>::value, "Object handled with refcounted_obj_ptr should derive from refcounted_obj");
        if(add_ref) ptr_add_ref();
    }

    //for refcounted_obj_ptr<const T>, allow contruction from T*
    template <class U = T, class = typename std::enable_if<std::is_const<U>::value>::type>
    refcounted_obj_ptr(typename std::remove_const<T>::type* ptr, bool add_ref = true) : _ptr(ptr)
    {
        static_assert(std::is_base_of<refcounted_obj<typename std::remove_const<U>::type>, U>::value, "Object handled with refcounted_obj_ptr should derive from refcounted_obj");
        if (add_ref) ptr_add_ref();
    }

    constexpr refcounted_obj_ptr() : _ptr(nullptr){}

    refcounted_obj_ptr(const refcounted_obj_ptr& other)
        : _ptr(other._ptr)
    {
        ptr_add_ref();
    }

    refcounted_obj_ptr& operator=(const refcounted_obj_ptr& other)
    {
        if (this == &other)
            return *this;
        ptr_release();
        _ptr = other._ptr;
        ptr_add_ref();
        return *this;
    }

    refcounted_obj_ptr(refcounted_obj_ptr&& other) noexcept
    {
        _ptr = other._ptr;
        other._ptr = nullptr;
    }

    refcounted_obj_ptr& operator=(refcounted_obj_ptr&& other)
    {
        if (this == &other)
            return *this;
        ptr_release();
        _ptr = other._ptr;
        other._ptr = nullptr;
        return *this;
    }

    ~refcounted_obj_ptr() { ptr_release(); _ptr = nullptr; }

    T* detach()
    {
        T* result = _ptr;
        _ptr = nullptr;
        return result;
    }

    void reset(T* ptr, bool add_ref = true)
    {
        ptr_release();
        _ptr = ptr;
        if (add_ref) ptr_add_ref();
    }

    operator bool() const { return _ptr != nullptr; }
    T* get() const { return _ptr; }
    T& operator*() const { return *get(); }
    T* operator->() const { return get(); }

    friend bool operator==(const refcounted_obj_ptr& lhs, const refcounted_obj_ptr& rhs)
    {
        return lhs._ptr == rhs._ptr;
    }

    friend bool operator!=(const refcounted_obj_ptr& lhs, const refcounted_obj_ptr& rhs)
    {
        return !(lhs == rhs);
    }

    // for refcounted_obj_ptr<T>, allow conversion to refcounted_obj_ptr<const T>
	template <class R>
	operator refcounted_obj_ptr<const R> () const
    {
        return refcounted_obj_ptr<const R>(_ptr);
    }

private:
    T* _ptr;
    void ptr_add_ref() { if (_ptr) _ptr->add_ref(); }
    void ptr_release() { if (_ptr) _ptr->release(); }
};

}
