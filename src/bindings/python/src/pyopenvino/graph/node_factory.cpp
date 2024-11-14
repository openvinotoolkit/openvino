// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "node_factory.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cctype>
#include <functional>
#include <locale>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "dict_attribute_visitor.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/op_extension.hpp"
#include "openvino/core/so_extension.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/opsets/opset.hpp"
#include "openvino/util/log.hpp"
#include "pyopenvino/core/common.hpp"
#include "pyopenvino/utils/utils.hpp"

namespace py = pybind11;

namespace {
class NodeFactory {
public:
    NodeFactory() {}
    NodeFactory(const std::string& opset_name) : m_opset(get_opset(opset_name)) {}

    std::shared_ptr<ov::Node> create(const std::string op_type_name,
                                     const ov::OutputVector& arguments,
                                     const py::dict& attributes = py::dict()) {
        // Check for available extensions first, because they may override ops from main opset
        auto ext_it = m_opset_so_extensions.find(op_type_name);
        if (ext_it != m_opset_so_extensions.end()) {
            auto op_extension = std::dynamic_pointer_cast<ov::BaseOpExtension>(ext_it->second->extension());
            OPENVINO_ASSERT(op_extension);  // guaranteed by add_extension method
            util::DictAttributeDeserializer visitor(attributes, m_variables);
            auto outputs = op_extension->create(arguments, visitor);

            OPENVINO_ASSERT(outputs.size() > 0,
                            "Failed to create extension operation with type: ",
                            op_type_name,
                            " because it doesn't contain output ports. Operation should has at least one output port.");

            auto node = outputs[0].get_node_shared_ptr();
            return node;
        } else {
            std::shared_ptr<ov::Node> op_node = std::shared_ptr<ov::Node>(m_opset.create(op_type_name));

            OPENVINO_ASSERT(op_node != nullptr, "Couldn't create operation: ", op_type_name);
            OPENVINO_ASSERT(!ov::op::util::is_constant(op_node),
                            "Currently NodeFactory doesn't support Constant operation: ",
                            op_type_name);

            util::DictAttributeDeserializer visitor(attributes, m_variables);

            op_node->set_arguments(arguments);
            op_node->visit_attributes(visitor);
            op_node->constructor_validate_and_infer_types();

            return op_node;
        }
    }

    std::shared_ptr<ov::Node> create(const std::string op_type_name) {
        // Check for available extensions first, because they may override ops from main opset
        auto ext_it = m_opset_so_extensions.find(op_type_name);
        // No way to instantiate operation without inputs, so if extension operation is found report an error.
        OPENVINO_ASSERT(ext_it == m_opset_so_extensions.end(),
                        "Couldn't create operation of type ",
                        op_type_name,
                        " from an extension library as no inputs were provided. Currently NodeFactory doesn't support ",
                        "operations without inputs. Provide at least one input.");

        std::shared_ptr<ov::Node> op_node = std::shared_ptr<ov::Node>(m_opset.create(op_type_name));

        OPENVINO_ASSERT(op_node != nullptr, "Couldn't create operation: ", op_type_name);
        OPENVINO_ASSERT(!ov::op::util::is_constant(op_node),
                        "Currently NodeFactory doesn't support Constant node: ",
                        op_type_name);

        OPENVINO_WARN("Empty op created! Please assign inputs and attributes and run validate() before op is used.");

        return op_node;
    }

    void add_extension(const std::shared_ptr<ov::Extension>& extension) {
        auto so_extension = std::dynamic_pointer_cast<ov::detail::SOExtension>(extension);
        ov::Extension::Ptr extension_extracted = so_extension ? so_extension->extension() : extension;
        if (auto op_extension = std::dynamic_pointer_cast<ov::BaseOpExtension>(extension_extracted)) {
            auto op_type = op_extension->get_type_info().name;
            // keep so extension instead of extension_extracted to hold loaded library
            m_opset_so_extensions[op_type] = so_extension;
        }
    }

    void add_extension(const std::vector<std::shared_ptr<ov::Extension>>& extensions) {
        // Load extension library, seach for operation extensions (derived from ov::BaseOpExtension) and keep
        // them in m_opset_so_extensions for future use in create methods.
        // NodeFactory provides a simplified API for node creation without involving version of operation.
        // It means all operations share the same name space and real operation versions (opsets) from extension
        // library are ignored.
        for (auto extension : extensions) {
            auto so_extension = std::dynamic_pointer_cast<ov::detail::SOExtension>(extension);
            ov::Extension::Ptr extension_extracted = so_extension ? so_extension->extension() : extension;
            if (auto op_extension = std::dynamic_pointer_cast<ov::BaseOpExtension>(extension_extracted)) {
                auto op_type = op_extension->get_type_info().name;
                // keep so extension instead of extension_extracted to hold loaded library
                m_opset_so_extensions[op_type] = so_extension;
            }
        }
    }

    void add_extension(const std::string& lib_path) {
        // Load extension library, seach for operation extensions (derived from ov::BaseOpExtension) and keep
        // them in m_opset_so_extensions for future use in create methods.
        // NodeFactory provides a simplified API for node creation without involving version of operation.
        // It means all operations share the same name space and real operation versions (opsets) from extension
        // library are ignored.
        auto extensions = ov::detail::load_extensions(lib_path);
        for (auto extension : extensions) {
            auto so_extension = std::dynamic_pointer_cast<ov::detail::SOExtension>(extension);
            ov::Extension::Ptr extension_extracted = so_extension ? so_extension->extension() : extension;
            if (auto op_extension = std::dynamic_pointer_cast<ov::BaseOpExtension>(extension_extracted)) {
                auto op_type = op_extension->get_type_info().name;
                // keep so extension instead of extension_extracted to hold loaded library
                m_opset_so_extensions[op_type] = so_extension;
            }
        }
    }

private:
    const ov::OpSet& get_opset(std::string opset_ver) {
        std::locale loc;
        std::transform(opset_ver.begin(), opset_ver.end(), opset_ver.begin(), [&loc](char c) {
            return std::tolower(c, loc);
        });

        const auto& s_opsets = ov::get_available_opsets();

        auto it = s_opsets.find(opset_ver);
        OPENVINO_ASSERT(it != s_opsets.end(), "Unsupported opset version requested.");
        return it->second();
    }

    const ov::OpSet& m_opset = ov::get_opset13();
    std::map<std::string, std::shared_ptr<ov::detail::SOExtension>> m_opset_so_extensions;
    std::unordered_map<std::string, std::shared_ptr<ov::op::util::Variable>> m_variables;
};
}  // namespace

void regclass_graph_NodeFactory(py::module m) {
    py::class_<NodeFactory> node_factory(m, "NodeFactory");
    node_factory.doc() = "NodeFactory creates nGraph nodes";

    node_factory.def(py::init());
    node_factory.def(py::init<std::string>());

    node_factory.def("create", [](NodeFactory& self, const std::string name) {
        return self.create(name);
    });
    node_factory.def(
        "create",
        [](NodeFactory& self, const std::string name, const ov::OutputVector& arguments, const py::dict& attributes) {
            return self.create(name, arguments, attributes);
        });

    node_factory.def("add_extension", [](NodeFactory& self, const std::shared_ptr<ov::Extension>& extension) {
        return self.add_extension(extension);
    });

    node_factory.def("add_extension",
                     [](NodeFactory& self, const std::vector<std::shared_ptr<ov::Extension>>& extension) {
                         return self.add_extension(extension);
                     });

    node_factory.def("add_extension", [](NodeFactory& self, const py::object& lib_path) {
        return self.add_extension(Common::utils::convert_path_to_string(lib_path));
    });

    node_factory.def("__repr__", [](const NodeFactory& self) {
        return Common::get_simple_repr(self);
    });
}
