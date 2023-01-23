// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/any.hpp"

namespace ov {
namespace frontend {

// Extendable type system which reflects Framework data types
// Type nestings are built with the help of ov::Any
namespace type {

struct Tensor {
    Tensor() = default;
    explicit Tensor(const Any& _element_type) : element_type(_element_type) {}
    Any element_type;
};

struct Tuple;

struct List {
    List() = default;

    // Specifies list of elements of element_type type, all elements have the same given type
    explicit List(const Any& _element_type) : element_type(_element_type) {}
    Any element_type;
};

struct Str {};

struct PyNone {};

struct Optional;
struct Dict;
struct NamedTuple;
struct Union;

}  // namespace type

/// Plays a role of node, block and module decoder
class IDecoder {
public:
    virtual ~IDecoder() = default;
};

}  // namespace frontend
}  // namespace ov
