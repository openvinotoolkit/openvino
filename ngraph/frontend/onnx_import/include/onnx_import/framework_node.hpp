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

#include "ngraph/node.hpp"
#include "ngraph/visibility.hpp"

namespace ngraph
{
namespace frontend
{

class NGRAPH_API FrameworkNode : public ngraph::Node
{
public:

    using Node::Node;

    // TODO: get_meta_attribute<T>(name) for op_type, domain, name etc.; all properties that are not in FW list of op attributes

    // TODO: get_attribute<T>(name) for real op attributes, the set is individual for each op type
};

} // namespace frontend
} // namespace ngraph
