// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>

#include "transformations/rt_info/tile_propagation_attribute.hpp"

namespace ngraph {

template class ngraph::VariantImpl<Tiles>;

constexpr VariantTypeInfo VariantWrapper<Tiles>::type_info;

Tiles getTiles(const Input<Node> & input) {
    const auto &rtInfo = input.get_rt_info();
    using TilesWraper = VariantWrapper<Tiles>;

    if (!rtInfo.count(TilesWraper::type_info.name)) {
        throw ngraph_error("There is no tile attribute");
    }

    const auto &attr = rtInfo.at(TilesWraper::type_info.name);
    return as_type_ptr<TilesWraper>(attr)->get();
}

bool hasTiles(const Input<Node> & input) {
    const auto &rtInfo = input.get_rt_info();
    using TilesWraper = VariantWrapper<Tiles>;

    return rtInfo.count(TilesWraper::type_info.name);
}

}  // namespace ngraph
