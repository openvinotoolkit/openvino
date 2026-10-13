//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <type_traits>

namespace intel_npu::vm {

// NOLINTNEXTLINE(readability-identifier-naming) - keep std::span-compatible naming
inline constexpr size_t dynamic_extent = static_cast<size_t>(-1);

// Represents a contiguous sequence of objects, with the first element of the sequence at position zero
template <typename T>
class Span {
public:
    using element_type = T;
    using value_type = std::remove_cv_t<T>;
    using size_type = size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using iterator = pointer;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_iterator = const_pointer;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    // NOLINTNEXTLINE(readability-identifier-naming) - keep std::span-compatible naming
    static constexpr size_t extent = dynamic_extent;

private:
    T* _data{};
    size_t _size{};

public:
    // Constructs an empty span
    constexpr Span() noexcept = default;

    // Constructs a span over [data, data + size)
    constexpr Span(pointer data, size_type size) noexcept: _data{data}, _size{size} {
        assert((data != nullptr || size == 0) && "Span cannot have a null data pointer if size is greater than zero");
    }

    // Constructs a span over [first, last)
    constexpr Span(pointer first, pointer last) noexcept: Span(first, static_cast<size_type>(last - first)) {
        assert((first != nullptr || last == first) &&
               "Span cannot have a null data pointer if size is greater than zero");
        assert((last >= first) && "Span cannot be constructed from an invalid [first, last) range");
    }

    template <size_t N>
    // Constructs a span over all elements in a C-style array
    constexpr Span(element_type (&arr)[N]) noexcept  // NOLINT(cppcoreguidelines-avoid-c-arrays)
            : Span(arr, N) {
    }

    // Constructs a span over all elements in a mutable std::array
    template <typename U, size_t N, typename = std::enable_if_t<std::is_convertible_v<U*, T*>>>
    constexpr Span(std::array<U, N>& arr) noexcept: Span(arr.data(), arr.size()) {
    }

    // Constructs a span over all elements in a const std::array
    template <typename U, size_t N, typename = std::enable_if_t<std::is_convertible_v<const U*, T*>>>
    constexpr Span(const std::array<U, N>& arr) noexcept: Span(arr.data(), arr.size()) {
    }

    // Constructs a span from another span when element pointers are convertible
    template <typename U, typename = std::enable_if_t<std::is_convertible_v<U*, T*>>>
    constexpr Span(const Span<U>& other) noexcept: Span(other.data(), other.size()) {
    }

    // Returns a reference to the element at index i, with bounds checking. Throws std::out_of_range if i >= size()
    constexpr reference at(size_type i) const {
        if (i >= size()) {
            throw std::out_of_range("Span index out of bounds");
        }
        return *std::next(_data, static_cast<difference_type>(i));
    }

    // Returns a reference to the element at index i
    constexpr reference operator[](size_type i) const noexcept {
        assert(i < size() && "Span index out of bounds");
        return *std::next(begin(), static_cast<difference_type>(i));
    }

    // Returns a reference to the first element
    constexpr reference front() const noexcept {
        assert(!empty() && "Span::front called on an empty span");
        return *begin();
    }

    // Returns a reference to the last element
    constexpr reference back() const noexcept {
        assert(!empty() && "Span::back called on an empty span");
        return *(end() - 1);
    }

    // Returns a pointer to the first element, or nullptr for a default-constructed empty span
    constexpr pointer data() const noexcept {
        return _data;
    }

    // Returns the number of elements
    constexpr size_type size() const noexcept {
        return _size;
    }

    // Returns the number of bytes in the view
    constexpr size_type size_bytes() const noexcept {  // NOLINT(readability-identifier-naming)
        return _size * sizeof(element_type);
    }

    // Returns true if the span has zero elements
    constexpr bool empty() const noexcept {
        return _size == 0;
    }

    // Returns an iterator to the first element
    constexpr iterator begin() const noexcept {
        return _data;
    }

    // Returns an iterator one past the last element
    constexpr iterator end() const noexcept {
        return (_data == nullptr) ? nullptr : std::next(_data, static_cast<difference_type>(_size));
    }

    // Returns a const iterator to the first element
    constexpr const_iterator cbegin() const noexcept {
        return _data;
    }

    // Returns a const iterator one past the last element
    constexpr const_iterator cend() const noexcept {
        return (_data == nullptr) ? nullptr : std::next(_data, static_cast<difference_type>(_size));
    }

    // Returns a reverse iterator to the first element of the reversed view
    constexpr reverse_iterator rbegin() const noexcept {
        return reverse_iterator(end());
    }

    // Returns a reverse iterator one past the last element of the reversed view
    constexpr reverse_iterator rend() const noexcept {
        return reverse_iterator(begin());
    }

    // Returns a const reverse iterator to the first element of the reversed view
    constexpr const_reverse_iterator crbegin() const noexcept {
        return const_reverse_iterator(cend());
    }

    // Returns a const reverse iterator one past the last element of the reversed view
    constexpr const_reverse_iterator crend() const noexcept {
        return const_reverse_iterator(cbegin());
    }

    // Returns a subview of the first count elements
    constexpr Span<T> first(size_type count) const noexcept {
        assert(count <= _size && "Span::first count exceeds span size");
        return Span<T>(_data, count);
    }

    // Returns a subview of the last count elements
    constexpr Span<T> last(size_type count) const noexcept {
        assert(count <= _size && "Span::last count exceeds span size");
        return Span<T>(std::next(_data, static_cast<difference_type>(_size - count)), count);
    }

    // Returns a subview with compile-time offset/count
    // If Count is dynamic_extent, the subview extends to the end
    template <size_t Offset, size_t Count = dynamic_extent>
    constexpr Span<T> subspan() const noexcept {
        assert(Offset <= _size && "Span::subspan offset exceeds span size");
        if constexpr (Count == dynamic_extent) {
            return Span<T>(std::next(_data, static_cast<difference_type>(Offset)), _size - Offset);
        }

        assert(Count <= _size - Offset && "Span::subspan count exceeds available range");
        return Span<T>(std::next(_data, static_cast<difference_type>(Offset)), Count);
    }

    // Returns a subview with runtime offset/count
    // If count is dynamic_extent, the subview extends to the end
    constexpr Span<T> subspan(size_type offset, size_type count = dynamic_extent) const noexcept {
        assert(offset <= _size && "Span::subspan offset exceeds span size");
        if (count == dynamic_extent) {
            return Span<T>(std::next(_data, static_cast<difference_type>(offset)), _size - offset);
        }

        assert(count <= _size - offset && "Span::subspan count exceeds available range");
        return Span<T>(std::next(_data, static_cast<difference_type>(offset)), count);
    }
};

// Returns a read-only byte view over the same memory.
template <typename T>
constexpr Span<const std::byte> as_bytes(Span<T> span) noexcept {  // NOLINT(readability-identifier-naming)
    return Span<const std::byte>(static_cast<const std::byte*>(static_cast<const void*>(span.data())),
                                 span.size_bytes());
}

// Returns a writable byte view over the same memory.
template <typename T, typename = std::enable_if_t<!std::is_const_v<T>>>
constexpr Span<std::byte> as_writable_bytes(Span<T> span) noexcept {  // NOLINT(readability-identifier-naming)
    return Span<std::byte>(static_cast<std::byte*>(static_cast<void*>(span.data())), span.size_bytes());
}

}  // namespace intel_npu::vm
