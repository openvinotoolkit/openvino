// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for generic LockedMemory<> and different variations of locks
 * @file ie_locked_memory.hpp
 */
#pragma once

#include <iostream>
#include "ie_allocator.hpp"
#include <utility>

namespace InferenceEngine {
namespace details {
/**
 * @brief This class is a LockedMemory concept for hardware memory
 */
template<class T>
class LockedMemoryBase {
    IAllocator *_allocator = nullptr;
    void *_handle = nullptr;
    mutable T *_locked = nullptr;
    LockOp _lockFlag = LOCK_FOR_WRITE;

protected:
    /**
     * @brief An offset size.
     * The default value is 0.
     */
    size_t _offset = 0;

public:
    /**
     * @brief A constructor
     * @param ptr Pointer to an IAllocator object
     * @param handle Handle provided by allocator->Alloc()
     * @param lockFlag Read/Write type of mapping
     * @param offsetInBytes Offset in originally locked region
     */
    LockedMemoryBase(IAllocator *ptr, void *handle, LockOp lockFlag, size_t offsetInBytes)
            : _allocator(ptr), _handle(handle), _lockFlag(lockFlag), _offset(offsetInBytes) {
    }

    /**
     * @brief A copy constructor
     * @param that An rvalue reference for the other LockedMemoryBase instance
     */
    LockedMemoryBase(LockedMemoryBase &&that)
            : _allocator(that._allocator), _handle(that._handle), _lockFlag(that._lockFlag),
              _offset(that._offset) {
        that._locked = nullptr;
    }

    /**
     * @brief A virtual destructor
     */
    virtual ~LockedMemoryBase() {
        if (_locked != nullptr) {
            _allocator->unlock(_handle);
        }
    }

protected:
    /**
     * @brief Compares referenced values
     * @param pointer Pointer to the object to compare with
     * @return True if all handlers are nullptr or referenced values are equal, false otherwise
     */
    bool isEqualTo(const T *pointer) const {
        if (pointer == nullptr &&
            (_allocator == nullptr || _handle == nullptr)) {
            return true;
        }
        return dereference() == pointer;
    }

    /**
     * @brief Gets the locked object.
     * Locks the handler and casts memory to the object of the given template type.
     * @return The pointer to the locked object, nullptr otherwise
     */
    virtual T *dereference() const {
        if (_locked != nullptr) return _locked;

        if (_allocator == nullptr) {
            return nullptr;
        }

        if (_handle == nullptr) {
            return nullptr;
        }

        uint8_t *pBytes = reinterpret_cast<uint8_t *>(_allocator->lock(_handle, _lockFlag));

        return _locked = reinterpret_cast<T *> (pBytes + _offset);
    }
};
}  // namespace details


/**
 * @brief This class represents locked memory for read/write memory
 */
template<class T>
class LockedMemory : public details::LockedMemoryBase<T> {
    using base = details::LockedMemoryBase<T>;

public:
    /**
     * @brief A constructor
     * @param ptr Pointer to IAllocator object
     * @param handle Handle provided by allocator
     * @param offsetInBytes Offset in originally locked region
     */
    LockedMemory(IAllocator *ptr, void *handle, size_t offsetInBytes = 0)
            : base(ptr, handle, LOCK_FOR_WRITE, offsetInBytes) {
    }

    /**
     * @brief A default copy constructor, accepting rvalue
     */
    LockedMemory(LockedMemory<T> &&) = default;

    /**
     * @brief A default copy constructor that accepts rvalue
     * Also sets the offset value for the new memory object
     * @param that Rvalue reference for the other LockedMemoryBase instance
     * @param offset Offset value
     */
    LockedMemory(LockedMemory<T> &&that, size_t offset)
            : base(std::move(that)) {
        base::_offset = offset;
    }

    /**
     * @brief A disabled copy constructor for lvalue
     */
    LockedMemory(const LockedMemory<T> &) = delete;

    /**
     * @brief Gets a pointer to the stored object.
     * Dereferences from the base class.
     * @return The pointer to the object of the given template type
     */
    operator T *() {
        return base::dereference();
    }

    /**
     * @brief Gets the const pointer to the stored object.
     * Dereferences from the base class.
     * @return The const pointer object of the given template type.
     */
    operator const T *() const {
        return base::dereference();
    }

    /**
     * @brief Compares stored object with the given one
     * @return true if objects are equal, false otherwise
     */
    bool operator==(const T *pointer) const {
        // special case with nullptr
        return base::isEqualTo(pointer);
    }

    /**
     * @brief Compares the object with the one stored in the memory.
     * @return true if objects are equal, false otherwise
     */
    friend bool operator==(const T *pointer, const LockedMemory<T> &lm) {
        return lm.operator==(pointer);
    }

    /**
     * @brief Casts stored object to any provided type.
     * Uses reinterpret_cast.
     * @tparam S Type to be casted to
     * @return Casted to the given type object
     */
    template<class S, typename = std::enable_if<std::is_pointer<S>::value>>
    S as() {
        return reinterpret_cast<S>(base::dereference());
    }

    /**
     * @brief Casts stored object to any provided type.
     * Uses reinterpret_cast.
     * @tparam S Type to be casted to
     * @return Casted to the given type const object
     */
    template<class S, typename = std::enable_if<std::is_pointer<S>::value>>
    const S as() const {
        return reinterpret_cast<S>(base::dereference());
    }
};

/**
 * @brief This class is for <void*> data and allows casting to any pointers
 */
template<>
class LockedMemory<void> : public details::LockedMemoryBase<void> {
    using base = details::LockedMemoryBase<void>;

public:
    /**
     * @brief A constructor
     * @param ptr Pointer to IAllocator object
     * @param handle Handle provided by allocator
     * @param offsetInBytes Offset in originally locked region
     */
    LockedMemory(IAllocator *ptr, void *handle, size_t offsetInBytes)
            : base(ptr, handle, LOCK_FOR_WRITE, offsetInBytes) {
    }

    /**
     * @brief A default copy constructor that accepts rvalue
     */
    LockedMemory(LockedMemory<void> &&) = default;

    /**
     * @brief A default copy constructor that accepts rvalue
     * Also sets the offset value for the new memory object
     * @param that Rvalue reference for the other LockedMemoryBase instance
     * @param offset Offset value
     */
    LockedMemory(LockedMemory<void> &&that, size_t offset)
            : base(std::move(that)) {
        base::_offset = offset;
    }

    /**
     * @brief A disabled copy constructor for lvalue
     */
    LockedMemory(const LockedMemory<void> &) = delete;

    /**
     * @brief Gets the pointer to the stored object of the given template type.
     * Dereferences from the base class.
     * @tparam S Type to be casted to
     * @return The pointer to the object of the given template type
     */
    template<class S>
    operator S *() {
        return reinterpret_cast<S *>(base::dereference());
    }

    /**
     * @brief Compares stored object with the given one.
     * @return true if objects are equal, false otherwise
     */
    bool operator==(const void *pointer) const {
        // special case with nullptr
        return base::isEqualTo(pointer);
    }

    /**
     * @brief Compares the object with the one stored in the memory.
     * @return true if objects are equal, false otherwise
     */
    friend bool operator==(const void *pointer, const LockedMemory<void> &lm) {
        return lm.operator==(pointer);
    }

    /**
     * @brief Casts stored object to any given type.
     * Uses reinterpret_cast.
     * @tparam S Type to be casted to
     * @return Casted to the given type object
     */
    template<class S, typename = std::enable_if<std::is_pointer<S>::value>>
    S as() {
        return reinterpret_cast<S>(dereference());
    }

    /**
     * @brief Casts stored object to any given type.
     * Uses reinterpret_cast.
     * @tparam S Type to be casted to
     * @return Casted to the given type const object
     */
    template<class S, typename = std::enable_if<std::is_pointer<S>::value>>
    const S as() const {
        return reinterpret_cast<S>(dereference());
    }
};


/**
 * @brief This class is for read-only segments
 */
template<class T>
class LockedMemory<const T> : public details::LockedMemoryBase<T> {
    using base = details::LockedMemoryBase<T>;

public:
    /**
     * @brief A constructor
     * @param ptr Pointer to IAllocator object
     * @param handle Handle provided by allocator
     * @param offsetInBytes Offset in originally locked region
     */
    LockedMemory(IAllocator *ptr, void *handle, size_t offset)
            : base(ptr, handle, LOCK_FOR_READ, offset) {
    }

    /**
     * @brief A default copy constructor that accepts rvalue
     */
    LockedMemory(LockedMemory<const T> &&) = default;

    /**
     * @brief A default copy constructor that accepts rvalue.
     * Also sets the offset value for the new memory object
     * @param that Rvalue reference for the other LockedMemoryBase instance
     * @param offset Offset value
     */
    LockedMemory(LockedMemory<const T> &&that, size_t offset)
            : base(std::move(that)) {
        base::_offset = offset;
    }

    /**
     * @brief A disabled copy constructor for lvalue
     */
    LockedMemory(const LockedMemory<const T> &) = delete;

    /**
     * @brief Gets the const pointer to the stored object.
     * Dereferences from the base class.
     * @return The pointer to the object.
     */
    operator const T *() const {
        return base::dereference();
    }

    /**
     * @brief Compares stored object with the given one
     * @return true if objects are equal, false otherwise
     */
    bool operator==(const T *pointer) const {
        // special case with nullptr
        return base::isEqualTo(pointer);
    }

    /**
     * @brief Compares the object with the one stored in the memory.
     * @return true if objects are equal, false otherwise
     */
    friend bool operator==(const T *pointer, const LockedMemory<const T> &lm) {
        return lm.operator==(pointer);
    }

    /**
     * @brief Casts stored object to any given type.
     * Uses reinterpret_cast.
     * @tparam S Type to be casted to
     * @return Casted to the given type object
     */
    template<class S, typename = std::enable_if<std::is_pointer<S>::value && std::is_const<S>::value>>
    S as() const {
        return reinterpret_cast<S>(base::dereference());
    }
};
}  // namespace InferenceEngine
