//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

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
}
