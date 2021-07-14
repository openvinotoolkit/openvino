// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/op/util/variable.hpp"
#include "ngraph/opsets/opset.hpp"
#include "node_factory.hpp"

namespace py = pybind11;

namespace
{
    class NodeFactory
    {
    public:
        NodeFactory() {}
        NodeFactory(const std::string& opset_name)
            : m_opset(get_opset(opset_name))
        {
        }

        std::shared_ptr<ngraph::Node> create(const std::string op_type_name,
                                             const ngraph::OutputVector& arguments,
                                             const py::dict& attributes = py::dict())
        {
            std::shared_ptr<ngraph::Node> op_node =
                std::shared_ptr<ngraph::Node>(m_opset.create(op_type_name));

            NGRAPH_CHECK(op_node != nullptr, "Couldn't create operator: ", op_type_name);
            NGRAPH_CHECK(!ngraph::op::is_constant(op_node),
                         "Currently NodeFactory doesn't support Constant node: ",
                         op_type_name);

            util::DictAttributeDeserializer visitor(attributes, m_variables);

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
                {"opset1", OpsetFunction(ngraph::get_opset1)},
                {"opset2", OpsetFunction(ngraph::get_opset2)},
                {"opset3", OpsetFunction(ngraph::get_opset3)},
                {"opset4", OpsetFunction(ngraph::get_opset4)},
                {"opset5", OpsetFunction(ngraph::get_opset5)},
                {"opset6", OpsetFunction(ngraph::get_opset6)},
                {"opset7", OpsetFunction(ngraph::get_opset7)},
                {"opset8", OpsetFunction(ngraph::get_opset8)},
            };

            auto it = s_opsets.find(opset_ver);
            if (it == s_opsets.end())
            {
                throw ngraph::ngraph_error("Unsupported opset version requested.");
            }
            return it->second();
        }

        const ngraph::OpSet& m_opset = ngraph::get_opset8();
        std::unordered_map<std::string, std::shared_ptr<ngraph::Variable>> m_variables;
    };
} // namespace

void regclass_pyngraph_NodeFactory(py::module m)
{
    py::class_<NodeFactory> node_factory(m, "NodeFactory");
    node_factory.doc() = "NodeFactory creates nGraph nodes";

    node_factory.def(py::init());
    node_factory.def(py::init<std::string>());

    node_factory.def("create", &NodeFactory::create);

    node_factory.def("__repr__", [](const NodeFactory& self) { return "<NodeFactory>"; });
}
