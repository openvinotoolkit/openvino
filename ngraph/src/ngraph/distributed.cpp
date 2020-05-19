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

#include "ngraph/distributed.hpp"
#include "ngraph/distributed/null.hpp"
#include "ngraph/log.hpp"
#include "ngraph/type.hpp"

using namespace ngraph;

namespace ngraph
{
    template <>
    EnumNames<reduction::Type>& EnumNames<reduction::Type>::get()
    {
        static auto enum_names = EnumNames<reduction::Type>("reduction::Type",
                                                            {{"SUM", reduction::Type::SUM},
                                                             {"PROD", reduction::Type::PROD},
                                                             {"MIN", reduction::Type::MIN},
                                                             {"MAX", reduction::Type::MAX}});
        return enum_names;
    }

    constexpr DiscreteTypeInfo AttributeAdapter<reduction::Type>::type_info;
}

std::ostream& reduction::operator<<(std::ostream& out, const reduction::Type& obj)
{
    return out << as_string(obj);
}

static std::unique_ptr<DistributedInterface> s_distributed_interface;

void ngraph::set_distributed_interface(std::unique_ptr<DistributedInterface> distributed_interface)
{
    NGRAPH_DEBUG << "Setting distributed interface to: " << distributed_interface->get_name();
    s_distributed_interface = std::move(distributed_interface);
}

DistributedInterface* ngraph::get_distributed_interface()
{
    if (nullptr == s_distributed_interface)
    {
        set_distributed_interface(
            std::unique_ptr<DistributedInterface>(new ngraph::distributed::Null()));
    }
    return s_distributed_interface.get();
}
