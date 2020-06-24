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

#include <algorithm>
#include <cctype>
#include <functional>
#include <locale>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "dict_attribute_visitor.hpp"
#include "ngraph/check.hpp"
#include "ngraph/except.hpp"
#include "ngraph/node.hpp"
#include "ngraph/opsets/opset.hpp"
#include "node_factory.hpp"
#include "tensor_iterator_builder.hpp"

namespace py = pybind11;

namespace
{
    class NodeFactory
    {
    public:
        NodeFactory() {}
        NodeFactory(const std::string& opset_name)
            : m_opset{get_opset(opset_name)}
        {
        }

        std::shared_ptr<ngraph::Node> create(const std::string op_type_name,
                                             const ngraph::NodeVector& arguments,
                                             const py::dict& attributes = py::dict())
        {
            std::shared_ptr<ngraph::Node> op_node =
                std::shared_ptr<ngraph::Node>(m_opset.create(op_type_name));

            NGRAPH_CHECK(op_node != nullptr, "Couldn't create operator: ", op_type_name);
            NGRAPH_CHECK(!op_node->is_constant(),
                         "Currently NodeFactory doesn't support Constant node: ",
                         op_type_name);

            if (op_type_name == "TensorIterator")
            {
                // XXX: How to differentiate opsets?
                return util::TensorIteratorBuilder(arguments, attributes)
                    .configure(std::static_pointer_cast<ngraph::op::TensorIterator>(op_node));
            }

            util::DictAttributeDeserializer visitor(attributes);

            op_node->set_arguments(arguments);
            op_node->visit_attributes(visitor);
            op_node->constructor_validate_and_infer_types();

            return op_node;
        }

    private:
        const ngraph::OpSet& get_opset(std::string opset_ver)
        {
            std::locale loc;
            std::transform(opset_ver.begin(), opset_ver.end(), opset_ver.begin(), [&loc](char c) {
                return std::tolower(c, loc);
            });

            using OpsetFunction = std::function<const ngraph::OpSet&()>;

            static const std::map<std::string, OpsetFunction> s_opsets{
                {"opset0", OpsetFunction(ngraph::get_opset0)},
                {"opset1", OpsetFunction(ngraph::get_opset1)},
                {"opset2", OpsetFunction(ngraph::get_opset2)},
                {"opset3", OpsetFunction(ngraph::get_opset3)},
            };

            auto it = s_opsets.find(opset_ver);
            if (it == s_opsets.end())
            {
                throw ngraph::ngraph_error("Unsupported opset version requested.");
            }
            return it->second();
        }

        const ngraph::OpSet& m_opset{ngraph::get_opset0()};
    };
}

void regclass_pyngraph_NodeFactory(py::module m)
{
    py::class_<NodeFactory> node_factory(m, "NodeFactory");
    node_factory.doc() = "NodeFactory creates nGraph nodes";

    node_factory.def(py::init());
    node_factory.def(py::init<std::string>());

    node_factory.def("create", &NodeFactory::create);

    node_factory.def("__repr__", [](const NodeFactory& self) { return "<NodeFactory>"; });
}
