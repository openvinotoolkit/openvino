//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include <mutex>

#include "ngraph/factory.hpp"
#include "ngraph/node.hpp"
#include "ngraph/ops.hpp"

using namespace std;

namespace ngraph
{
    mutex& get_registry_mutex()
    {
        static mutex registry_mutex;
        return registry_mutex;
    }

    template class NGRAPH_API FactoryRegistry<Node>;

    template <>
    FactoryRegistry<Node>& FactoryRegistry<Node>::get()
    {
        static FactoryRegistry<Node> registry;
        static mutex init_guard;
        // TODO: Add a lock
        if (registry.m_factory_map.size() == 0)
        {
            lock_guard<mutex> guard(init_guard);
            if (registry.m_factory_map.size() == 0)
            {
#define NGRAPH_OP(NAME, NAMESPACE, VERSION) registry.register_factory<NAMESPACE::NAME>();
#include "ngraph/op/op_version_tbl.hpp"
#undef NGRAPH_OP
            }
        }
        return registry;
    }
}
