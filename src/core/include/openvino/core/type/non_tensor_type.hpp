// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once


#include "openvino/core/core_visibility.hpp"

namespace ov {
namespace element {
namespace StructuralType {

/// Tensor type of any element; not used to annotate simple tensors with representable types
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

struct TensorListWithRank {
    TensorListWithRank(const Any& _element_type, size_t _rank) : element_type(_element_type), rank(_rank) {}

    Any element_type;
    size_t rank;
};

struct Ragged {
    Ragged() = default;

    // Specifies list of elements of element_type type, all elements have the same given type
    explicit Ragged(const Any& _element_type) : element_type(_element_type) {}
    Any element_type;
};

struct Str {};

struct Optional;
struct Dict;
struct NamedTuple;
struct Union;

inline void print(std::ostream& out, const Any& x) {
    if (x.is<element::Type>()) {
        out << x.as<element::Type>();
    } else if (x.is<Tensor>()) {
        out << "tensor[";
        print(out, x.as<Tensor>().element_type);
        out << "]";
    } else if (x.is<List>()) {
        out << "list[";
        print(out, x.as<List>().element_type);
        out << "]";
    } else if (x.is<TensorListWithRank>()) {
        auto tlwr = x.as<TensorListWithRank>();
        out << "tensor_list_with_rank[";
        print(out, tlwr.element_type);
        out << ", rank = " << tlwr.rank << "]";
    } else if (x.is<Ragged>()) {
        out << "ragged[";
        print(out, x.as<Ragged>().element_type);
        out << "]";
    } else if (x.is<Str>()) {
        out << "str";
    } else {
        out << "<UNKNWON_ANY_TYPE_ERROR>";
    }
    out << std::flush;
}

inline bool operator== (const Str& x, const Str& y) { return true; }

}  // namespace StructuralType
}  // namespace element
}  // namespace ov
