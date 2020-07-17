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

#pragma once

#include <memory> // std::shared_ptr, std::make_shared
#include <onnx/onnx_pb.h>
#include <vector>

#include "core/node.hpp"
#include "default_opset.hpp"
#include "ngraph/node.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace utils
        {
            ///
            /// \brief      Return a vector of final nGraph nodes obtained by expanding the ONNX
            /// fuction.
            ///
            /// \param[in]  node                The node representing incoming ONNX fuction.
            ///
            /// \return     Vector of nGraph nodes equivalent of the final outputs of the ONNX
            /// fuction.
            ///
            NodeVector expand_onnx_function(const Node& node);

            ///
            /// \brief      Return Proto Type with set type and shape.
            ///
            /// \param[in]  type                The nGraph type that will be translated to a
            /// suitable Proto Type.
            /// \param[in]  shape               The shape of tensor for the target Proto Type.
            ///
            /// \return     Proto Type equivalent of the given nGraph type and shape.
            ///
            ONNX_NAMESPACE::TypeProto get_proto_type(element::Type type, Shape shape);

            ///
            /// \brief      Return a vector of nGraph nodes representing expanded ONNX fuction.
            ///
            /// \param[in]  node                The Proto Node representing incoming ONNX fuction.
            /// \param[in]  graph               The Proto Graph with the node with ONNX fuction.
            /// \param[in]  opset_version       The opset version of the ONNX function.
            ///
            /// \return     Vector of nGraph nodes equivalent of the ONNX fuction.
            ///
            std::vector<std::shared_ptr<ngraph::Node>>
                get_nodes_from_onnx_function(ONNX_NAMESPACE::NodeProto* node,
                                             ONNX_NAMESPACE::GraphProto graph,
                                             int opset_version);
        }
    }
}