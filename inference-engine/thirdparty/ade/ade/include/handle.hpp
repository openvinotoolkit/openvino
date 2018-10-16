// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef HANDLE_HPP
#define HANDLE_HPP

#include <memory>
#include <iosfwd>
#include <functional> //std::hash

#include "util/assert.hpp"

namespace ade
{

// Non owning graph elements pointer
// Elements owned by graph
template<typename T>
class Handle final
{
    friend class Graph;

    std::weak_ptr<T> m_ptr;

    Handle(const std::shared_ptr<T>& obj):
        m_ptr(obj)
    {
        ASSERT(nullptr != obj);
    }

    static T* check(T* val)
    {
        ASSERT_STRONG(nullptr != val);
        return val;
    }

public:
    Handle() = default;
    Handle(std::nullptr_t) {}
    Handle(const Handle&) = default;
    Handle& operator=(const Handle&) = default;
    Handle& operator=(std::nullptr_t) { m_ptr.reset(); return *this; }
    Handle(Handle&&) = default;
    Handle& operator=(Handle&&) = default;

    // Graphs not intended for multithreaded modification so this should be safe
    T* get() const { return m_ptr.lock().get(); }

    T& operator*() const { return *check(get()); }

    T* operator->() const { return check(get()); }

    bool operator==(const Handle& other) const
    {
        return get() == other.get();
    }

    bool operator!=(const Handle& other) const
    {
        return get() != other.get();
    }

    friend bool operator==(std::nullptr_t, const Handle& other)
    {
        return nullptr == other.get();
    }

    friend bool operator==(const Handle& other, std::nullptr_t)
    {
        return nullptr == other.get();
    }

    friend bool operator!=(std::nullptr_t, const Handle& other)
    {
        return nullptr != other.get();
    }

    friend bool operator!=(const Handle& other, std::nullptr_t)
    {
        return nullptr != other.get();
    }
};

template<typename T>
inline std::ostream& operator<<(std::ostream& os, const Handle<T>& arg)
{
    os << arg.get();
    return os;
}

/// Generally it is not safe to use handles as keys so we dont provide default
/// hash, user must use this hasher explicitly if he really want to
template<typename T>
struct HandleHasher final
{
    std::size_t operator()(const ade::Handle<T>& handle) const
    {
        ASSERT(nullptr != handle);
        return std::hash<decltype(handle.get())>()(handle.get());
    }
};


}

#endif // HANDLE_HPP
