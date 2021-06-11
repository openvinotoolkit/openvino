/*******************************************************************************
 * Copyright 2017-2020 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/
#pragma once

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>
#include "node_context.hpp"


namespace ngraph
{
namespace frontend
{
namespace tensorflow
{

/// Abstract representation for an input model graph that gives nodes in topologically sorted order
class GraphIterator
{
public:

    virtual size_t size () const = 0;

    /// Set iterator to the start position
    virtual void reset () = 0;

    /// Moves to the next node in the graph
    virtual void next () = 0;

    /// Returns true if iterator goes out of the range of available nodes
    virtual bool is_end () const = 0;

    /// Return NodeContext for the current node that iterator points to
    virtual std::shared_ptr<detail::TFNodeDecoder> get () const = 0;
};

}
}
}

