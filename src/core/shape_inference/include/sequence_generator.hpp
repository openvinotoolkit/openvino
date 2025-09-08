// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace ov {
/** \brief Enumerate directions */
enum Direction : uint8_t { FORWARD, BACKWARD };

/**
 * \brief Infinite generator of sequence increasing values.
 *
 * Start value can be specified.
 *
 * \tparam T Type of sequence values (must support `++` or '--' operators).
 */
template <class T, Direction D = Direction::FORWARD>
class SeqGen {
    T _counter;

public:
    constexpr SeqGen(const T& start) : _counter{start} {}

    template <Direction Di = D, typename std::enable_if<Di == Direction::FORWARD>::type* = nullptr>
    T operator()() {
        return _counter++;
    }

    template <Direction Di = D, typename std::enable_if<Di == Direction::BACKWARD>::type* = nullptr>
    T operator()() {
        return _counter--;
    }
};
}  // namespace ov
