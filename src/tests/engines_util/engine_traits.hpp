// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace ngraph {
namespace test {
/// These templates should be specialized for each test engine and they should contain
/// a "static constexpr const bool value" member set to true or false.
/// These traits are used in engine_factory.hpp

/// Indicates that a given Engine can be constructed for different devices (IE engines)
template <typename Engine>
struct supports_devices;

/// Indicates that a given Engine supports dynamic shapes
template <typename Engine>
struct supports_dynamic;

/// Example:
///
// template <>
// struct supports_dynamic<EngineName> {
//     static constexpr const bool value = true;
// };
}  // namespace test
}  // namespace ngraph
