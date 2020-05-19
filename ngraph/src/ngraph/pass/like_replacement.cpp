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

#include <functional>
#include <memory>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

#include "like_replacement.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/stop_gradient.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

static bool replace_broadcast_like(const std::shared_ptr<ngraph::Node>& node)
{
    // Replace a broadcast like with the broadcast to eliminate the pseudo-dependency on the "like"
    // argument
    auto broadcast_like = as_type_ptr<op::BroadcastLike>(node);
    replace_node(node,
                 make_shared<op::Broadcast>(broadcast_like->get_argument(0),
                                            broadcast_like->get_broadcast_shape(),
                                            broadcast_like->get_broadcast_axes()));
    return true;
}

static bool replace_scalar_constant_like(const std::shared_ptr<Node>& node)
{
    auto scalar_constant_like = as_type_ptr<op::ScalarConstantLike>(node);
    replace_node(node, scalar_constant_like->as_constant());
    return true;
}

static const map<NodeTypeInfo, function<bool(const shared_ptr<Node>&)>> dispatcher{
    {op::BroadcastLike::type_info, replace_broadcast_like},
    {op::ScalarConstantLike::type_info, replace_scalar_constant_like}};

bool pass::LikeReplacement::run_on_function(shared_ptr<Function> function_ptr)
{
    static const map<NodeTypeInfo, function<bool(const shared_ptr<Node>&)>> dispatcher{
        {op::BroadcastLike::type_info, replace_broadcast_like},
        {op::ScalarConstantLike::type_info, replace_scalar_constant_like}};

    bool clobbered = false;
    for (const auto& n : function_ptr->get_ops())
    {
        // Work around a warning [-Wpotentially-evaluated-expression]
        auto handler = dispatcher.find(n->get_type_info());
        if (handler != dispatcher.end())
        {
            clobbered = handler->second(n) || clobbered;
        }
    }

    return clobbered;
}
