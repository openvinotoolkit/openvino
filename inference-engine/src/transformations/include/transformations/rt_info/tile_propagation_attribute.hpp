// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <assert.h>
#include <functional>
#include <memory>
#include <string>
#include <set>

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>
#include <transformations_visibility.hpp>


namespace ngraph {

class TRANSFORMATIONS_API TileElement {
public:
    int64_t h_begin, h_end, w_begin, w_end;

    TileElement() = default;

    TileElement(std::vector<int64_t> slice) {
        assert(slice.size() == 4);
        h_begin = slice[0];
        h_end = slice[1];
        w_begin = slice[2];
        w_end = slice[3];
    }
};

class TRANSFORMATIONS_API Tiles {
public:
    std::vector<std::vector<TileElement>> tiles;

    Tiles(int64_t h, int64_t w)
        : tiles(h, std::vector<TileElement>(w)) {}
};

extern template class TRANSFORMATIONS_API VariantImpl<Tiles>;

template<>
class TRANSFORMATIONS_API VariantWrapper<Tiles> : public VariantImpl<Tiles> {
public:
    static constexpr VariantTypeInfo type_info{"Variant::RuntimeAttribute::Tiles", 0};

    const VariantTypeInfo &get_type_info() const override {
        return type_info;
    }

    VariantWrapper(const value_type &value) : VariantImpl<value_type>(value) {}
};

TRANSFORMATIONS_API Tiles getTiles(const Input<Node> & input);

TRANSFORMATIONS_API bool hasTiles(const Input<Node> & input);

}  // namespace ngraph
