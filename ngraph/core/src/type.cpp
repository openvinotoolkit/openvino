// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/type.hpp"
#include "ngraph/util.hpp"

namespace std
{
    size_t std::hash<ngraph::DiscreteTypeInfo>::operator()(const ngraph::DiscreteTypeInfo& k) const
    {
        size_t name_hash = hash<string>()(string(k.name));
        size_t version_hash = hash<decltype(k.version)>()(k.version);
        // don't use parent for hash calculation, it is not a part of type (yet)
        return ngraph::hash_combine(vector<size_t>{name_hash, version_hash});
    }
} // namespace std
