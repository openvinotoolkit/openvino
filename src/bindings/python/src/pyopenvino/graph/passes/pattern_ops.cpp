// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/graph/passes/pattern_ops.hpp"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iterator>
#include <sstream>
#include <string>

#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "pyopenvino/core/common.hpp"

static ov::NodeTypeInfo get_type(const std::string& type_name) {
    // Supported types: opsetX.OpName or opsetX::OpName
    std::string opset_type;
    auto it = type_name.cbegin();
    while (it != type_name.cend() && *it != '.' && *it != ':') {
        opset_type += *it;
        ++it;
    }

    // Skip delimiter
    while (it != type_name.cend() && (*it == '.' || *it == ':')) {
        ++it;
    }

    // Get operation type name
    std::string operation_type(it, type_name.end());

    const auto& opsets = ov::get_available_opsets();
    OPENVINO_ASSERT(opsets.count(opset_type), "Unsupported opset type: ", opset_type);

    const ov::OpSet& m_opset = opsets.at(opset_type)();
    OPENVINO_ASSERT(m_opset.contains_type(operation_type), "Unrecognized operation type: ", operation_type);

    return m_opset.create(operation_type)->get_type_info();
}

inline std::vector<ov::NodeTypeInfo> get_types(const std::vector<std::string>& type_names) {
    std::vector<ov::NodeTypeInfo> types;
    for (const auto& type_name : type_names) {
        types.emplace_back(get_type(type_name));
    }
    return types;
}

using Predicate = const ov::pass::pattern::op::ValuePredicate;

static void reg_pattern_wrap_type(py::module m) {
    py::class_<ov::pass::pattern::op::WrapType, std::shared_ptr<ov::pass::pattern::op::WrapType>, ov::Node> wrap_type(
        m,
        "WrapType");
    wrap_type.doc() = "openvino.runtime.passes.WrapType wraps ov::pass::pattern::op::WrapType";

    wrap_type.def(py::init([](const std::string& type_name) {
                      return std::make_shared<ov::pass::pattern::op::WrapType>(get_type(type_name));
                  }),
                  py::arg("type_name"),
                  R"(
                  Create WrapType with given node type.

                  :param type_name: node type. For example: "opset8.Abs"
                  :type type_name: str
    )");

    wrap_type.def(py::init([](const std::string& type_name, const Predicate& pred) {
                      return std::make_shared<ov::pass::pattern::op::WrapType>(get_type(type_name), pred);
                  }),
                  py::arg("type_name"),
                  py::arg("pred"),
                  R"(
                  Create WrapType with given node type and predicate.

                  :param type_name: node type. For example: "opset8.Abs"
                  :type type_name: str

                  :param predicate: Function that performs additional checks for matching.
                  :type predicate: function
    )");

    wrap_type.def(py::init([](const std::string& type_name, const ov::Output<ov::Node>& input) {
                      return std::make_shared<ov::pass::pattern::op::WrapType>(get_type(type_name),
                                                                               nullptr,
                                                                               ov::OutputVector{input});
                  }),
                  py::arg("type_name"),
                  py::arg("input"),
                  R"(
                  Create WrapType with given node type and input node.

                  :param type_name: node type. For example: "opset8.Abs"
                  :type type_name: str

                  :param input: Node output.
                  :type input: openvino.runtime.Output
    )");

    wrap_type.def(py::init([](const std::string& type_name, const std::shared_ptr<ov::Node>& input) {
                      return std::make_shared<ov::pass::pattern::op::WrapType>(get_type(type_name),
                                                                               nullptr,
                                                                               ov::OutputVector{input});
                  }),
                  py::arg("type_name"),
                  py::arg("input"),
                  R"(
                  Create WrapType with given node type and input node.

                  :param type_name: node type. For example: opset8.Abs
                  :type type_name: str

                  :param input: Input node.
                  :type input: openvino.runtime.Node
    )");

    wrap_type.def(py::init([](const std::string& type_name, const ov::Output<ov::Node>& input, const Predicate& pred) {
                      return std::make_shared<ov::pass::pattern::op::WrapType>(get_type(type_name),
                                                                               pred,
                                                                               ov::OutputVector{input});
                  }),
                  py::arg("type_name"),
                  py::arg("input"),
                  py::arg("predicate"),
                  R"(
                  Create WrapType with given node type, input node and predicate.

                  :param type_name: node type. For example: "opset8.Abs"
                  :type type_name: str

                  :param input: Node output.
                  :type input: openvino.runtime.Output

                  :param predicate: Function that performs additional checks for matching.
                  :type predicate: function
    )");

    wrap_type.def(
        py::init([](const std::string& type_name, const std::shared_ptr<ov::Node>& input, const Predicate& pred) {
            return std::make_shared<ov::pass::pattern::op::WrapType>(get_type(type_name),
                                                                     pred,
                                                                     ov::OutputVector{input});
        }),
        py::arg("type_name"),
        py::arg("input"),
        py::arg("predicate"),
        R"(
                  Create WrapType with given node type, input node and predicate.

                  :param type_name: node type. For example: "opset8.Abs"
                  :type type_name: str

                  :param input: Input node.
                  :type input: openvino.runtime.Node

                  :param predicate: Function that performs additional checks for matching.
                  :type predicate: function
    )");

    wrap_type.def(py::init([](const std::string& type_name, const ov::OutputVector& inputs) {
                      return std::make_shared<ov::pass::pattern::op::WrapType>(get_type(type_name), nullptr, inputs);
                  }),
                  py::arg("type_name"),
                  py::arg("inputs"),
                  R"(
                  Create WrapType with given node type and input nodes.

                  :param type_name: node type. For example: "opset8.Abs"
                  :type type_name: str

                  :param inputs: Node outputs.
                  :type inputs: List[openvino.runtime.Output]
    )");

    wrap_type.def(py::init([](const std::string& type_name, const ov::NodeVector& inputs) {
                      return std::make_shared<ov::pass::pattern::op::WrapType>(get_type(type_name),
                                                                               nullptr,
                                                                               ov::as_output_vector(inputs));
                  }),
                  py::arg("type_name"),
                  py::arg("inputs"),
                  R"(
                  Create WrapType with given node type and input nodes.

                  :param type_name: node type. For example: "opset8.Abs"
                  :type type_name: str

                  :param inputs: Input nodes.
                  :type inputs: List[openvino.runtime.Node]
    )");

    wrap_type.def(py::init([](const std::string& type_name, const ov::OutputVector& inputs, const Predicate& pred) {
                      return std::make_shared<ov::pass::pattern::op::WrapType>(get_type(type_name), pred, inputs);
                  }),
                  py::arg("type_name"),
                  py::arg("inputs"),
                  py::arg("predicate"),
                  R"(
                  Create WrapType with given node type, input nodes and predicate.

                  :param type_name: node type. For example: "opset8.Abs"
                  :type type_name: str

                  :param inputs: Node outputs.
                  :type inputs: List[openvino.runtime.Output]

                  :param predicate: Function that performs additional checks for matching.
                  :type predicate: function
    )");

    wrap_type.def(py::init([](const std::string& type_name, const ov::NodeVector& inputs, const Predicate& pred) {
                      return std::make_shared<ov::pass::pattern::op::WrapType>(get_type(type_name),
                                                                               pred,
                                                                               ov::as_output_vector(inputs));
                  }),
                  py::arg("type_name"),
                  py::arg("inputs"),
                  py::arg("predicate"),
                  R"(
                  Create WrapType with given node type, input nodes and predicate.

                  :param type_name: node type. For example: "opset8.Abs"
                  :type type_name: str

                  :param inputs: Input nodes.
                  :type inputs: List[openvino.runtime.Node]

                  :param predicate: Function that performs additional checks for matching.
                  :type predicate: function
    )");

    wrap_type.def(py::init([](const std::vector<std::string>& type_names) {
                      return std::make_shared<ov::pass::pattern::op::WrapType>(get_types(type_names));
                  }),
                  py::arg("type_names"),
                  R"(
                  Create WrapType with given node types.

                  :param type_names: node types. For example: ["opset8.Abs", "opset8.Relu"]
                  :type type_names: List[str]
    )");

    wrap_type.def(py::init([](const std::vector<std::string>& type_names, const Predicate& pred) {
                      return std::make_shared<ov::pass::pattern::op::WrapType>(get_types(type_names), pred);
                  }),
                  py::arg("type_names"),
                  py::arg("predicate"),
                  R"(
                  Create WrapType with given node types and predicate.

                  :param type_names: node types. For example: ["opset8.Abs", "opset8.Relu"]
                  :type type_names: List[str]

                  :param predicate: Function that performs additional checks for matching.
                  :type predicate: function
    )");

    wrap_type.def(py::init([](const std::vector<std::string>& type_names, const ov::Output<ov::Node>& input) {
                      return std::make_shared<ov::pass::pattern::op::WrapType>(get_types(type_names),
                                                                               nullptr,
                                                                               ov::OutputVector{input});
                  }),
                  py::arg("type_names"),
                  py::arg("input"),
                  R"(
                  Create WrapType with given node types and input.

                  :param type_names: node types. For example: ["opset8.Abs", "opset8.Relu"]
                  :type type_names: List[str]

                  :param input: Node output.
                  :type input: openvino.runtime.Output
    )");

    wrap_type.def(py::init([](const std::vector<std::string>& type_names, const std::shared_ptr<ov::Node>& input) {
                      return std::make_shared<ov::pass::pattern::op::WrapType>(get_types(type_names),
                                                                               nullptr,
                                                                               ov::OutputVector{input});
                  }),
                  py::arg("type_names"),
                  py::arg("input"),
                  R"(
                  Create WrapType with given node types and input.

                  :param type_name: node types. For example: ["opset8.Abs", "opset8.Relu"]
                  :type type_name: List[str]

                  :param input: Input node.
                  :type input: openvino.runtime.Node
    )");

    wrap_type.def(
        py::init(
            [](const std::vector<std::string>& type_names, const ov::Output<ov::Node>& input, const Predicate& pred) {
                return std::make_shared<ov::pass::pattern::op::WrapType>(get_types(type_names),
                                                                         pred,
                                                                         ov::OutputVector{input});
            }),
        py::arg("type_names"),
        py::arg("input"),
        py::arg("predicate"),
        R"(
        Create WrapType with given node types, input and predicate.

        :param type_names: node types. For example: ["opset8.Abs", "opset8.Relu"]
        :type type_names: List[str]

        :param input: Node output.
        :type input: openvino.runtime.Output

        :param predicate: Function that performs additional checks for matching.
        :type predicate: function
    )");

    wrap_type.def(py::init([](const std::vector<std::string>& type_names,
                              const std::shared_ptr<ov::Node>& input,
                              const Predicate& pred) {
                      return std::make_shared<ov::pass::pattern::op::WrapType>(get_types(type_names),
                                                                               pred,
                                                                               ov::OutputVector{input});
                  }),
                  py::arg("type_names"),
                  py::arg("input"),
                  py::arg("predicate"),
                  R"(
        Create WrapType with given node types, input and predicate.

        :param type_names: node types. For example: ["opset8.Abs", "opset8.Relu"]
        :type type_names: List[str]

        :param input: Input node.
        :type input: openvino.runtime.Node

        :param predicate: Function that performs additional checks for matching.
        :type predicate: function
    )");

    wrap_type.def(py::init([](const std::vector<std::string>& type_names, const ov::OutputVector& inputs) {
                      return std::make_shared<ov::pass::pattern::op::WrapType>(get_types(type_names), nullptr, inputs);
                  }),
                  py::arg("type_names"),
                  py::arg("inputs"),
                  R"(
      Create WrapType with given node types and input.

      :param type_names: node types. For example: ["opset8.Abs", "opset8.Relu"]
      :type type_names: List[str]

      :param inputs: Nodes outputs.
      :type inputs: List[openvino.runtime.Output]
    )");

    wrap_type.def(py::init([](const std::vector<std::string>& type_names, const ov::NodeVector& inputs) {
                      return std::make_shared<ov::pass::pattern::op::WrapType>(get_types(type_names),
                                                                               nullptr,
                                                                               ov::as_output_vector(inputs));
                  }),
                  py::arg("type_names"),
                  py::arg("inputs"),
                  R"(
        Create WrapType with given node types and inputs.

        :param type_names: node types. For example: ["opset8.Abs", "opset8.Relu"]
        :type type_names: List[str]

        :param inputs: Input nodes.
        :type inputs: List[openvino.runtime.Node]
    )");

    wrap_type.def(
        py::init([](const std::vector<std::string>& type_names, const ov::OutputVector& inputs, const Predicate& pred) {
            return std::make_shared<ov::pass::pattern::op::WrapType>(get_types(type_names), pred, inputs);
        }),
        py::arg("type_names"),
        py::arg("inputs"),
        py::arg("predicate"),
        R"(
        Create WrapType with given node types, inputs and predicate.

        :param type_names: node types. For example: ["opset8.Abs", "opset8.Relu"]
        :type type_names: List[str]

        :param inputs: Nodes outputs.
        :type inputs: List[openvino.runtime.Output]

        :param predicate: Function that performs additional checks for matching.
        :type predicate: function
    )");

    wrap_type.def(
        py::init([](const std::vector<std::string>& type_names, const ov::NodeVector& inputs, const Predicate& pred) {
            return std::make_shared<ov::pass::pattern::op::WrapType>(get_types(type_names),
                                                                     pred,
                                                                     ov::as_output_vector(inputs));
        }),
        py::arg("type_names"),
        py::arg("inputs"),
        py::arg("predicate"),
        R"(
        Create WrapType with given node types, inputs and predicate.

        :param type_names: node types. For example: ["opset8.Abs", "opset8.Relu"]
        :type type_names: List[str]

        :param inputs: Input nodes.
        :type inputs: List[openvino.runtime.Node]

        :param predicate: Function that performs additional checks for matching.
        :type predicate: function
    )");

    wrap_type.def("__repr__", [](const ov::pass::pattern::op::WrapType& self) {
        return Common::get_simple_repr(self);
    });
}

static void reg_pattern_or(py::module m) {
    py::class_<ov::pass::pattern::op::Or, std::shared_ptr<ov::pass::pattern::op::Or>, ov::Node> or_type(m, "Or");
    or_type.doc() = "openvino.runtime.passes.Or wraps ov::pass::pattern::op::Or";

    or_type.def(py::init([](const ov::OutputVector& inputs) {
                    return std::make_shared<ov::pass::pattern::op::Or>(inputs);
                }),
                py::arg("inputs"),
                R"(
                Create pattern Or operation which is used to match any of given inputs.

                :param inputs: Operation inputs.
                :type inputs: List[openvino.runtime.Output]
    )");

    or_type.def(py::init([](const ov::NodeVector& inputs) {
                    return std::make_shared<ov::pass::pattern::op::Or>(ov::as_output_vector(inputs));
                }),
                py::arg("inputs"),
                R"(
                Create pattern Or operation which is used to match any of given inputs.

                :param inputs: Operation inputs.
                :type inputs: List[openvino.runtime.Node]
    )");

    or_type.def("__repr__", [](const ov::pass::pattern::op::Or& self) {
        return Common::get_simple_repr(self);
    });
}

static void reg_pattern_any_input(py::module m) {
    py::class_<ov::pass::pattern::op::Label, std::shared_ptr<ov::pass::pattern::op::Label>, ov::Node> any_input(
        m,
        "AnyInput");
    any_input.doc() = "openvino.runtime.passes.AnyInput wraps ov::pass::pattern::op::Label";

    any_input.def(py::init([]() {
                      return std::make_shared<ov::pass::pattern::op::Label>();
                  }),
                  R"(
                  Create pattern AnyInput operation which is used to match any type of node.
    )");

    any_input.def(py::init([](const Predicate& pred) {
                      return std::make_shared<ov::pass::pattern::op::Label>(ov::element::dynamic,
                                                                            ov::PartialShape::dynamic(),
                                                                            pred);
                  }),
                  py::arg("predicate"),
                  R"(
                  Create pattern AnyInput operation which is used to match any type of node.

                  :param pred: Function that performs additional checks for matching.
                  :type pred: function
    )");

    any_input.def("__repr__", [](const ov::pass::pattern::op::Label& self) {
        return Common::get_simple_repr(self);
    });
}

static void reg_pattern_optional(py::module m) {
    py::class_<ov::pass::pattern::op::Optional, std::shared_ptr<ov::pass::pattern::op::Optional>, ov::Node>
        optional_type(m, "Optional");
    optional_type.doc() = "openvino.runtime.passes.Optional wraps ov::pass::pattern::op::Optional";

    optional_type.def(py::init([](const std::vector<std::string>& type_names) {
                          return std::make_shared<ov::pass::pattern::op::Optional>(get_types(type_names));
                      }),
                      py::arg("type_names"),
                      R"(
        Create Optional with the given node type.

        :param type_names: node type. For example: ["opset8.Abs", "opset8.Relu"]
        :type type_names: List[str]
    )");

    optional_type.def(py::init([](const std::vector<std::string>& type_names, const ov::Output<ov::Node>& input) {
                          return std::make_shared<ov::pass::pattern::op::Optional>(get_types(type_names),
                                                                                   ov::OutputVector{input},
                                                                                   nullptr);
                      }),
                      py::arg("type_names"),
                      py::arg("input"),
                      R"(
        Create Optional with the given node type and input node.

        :param type_names: node type. For example: ["opset8.Abs", "opset8.Relu"]
        :type type_names: List[str]

        :param input: input node's output.
        :type input: openvino.runtime.Output
    )");

    optional_type.def(py::init([](const std::vector<std::string>& type_names, const std::shared_ptr<ov::Node>& input) {
                          return std::make_shared<ov::pass::pattern::op::Optional>(get_types(type_names),
                                                                                   ov::OutputVector{input},
                                                                                   nullptr);
                      }),
                      py::arg("type_names"),
                      py::arg("input"),
                      R"(
        Create Optional with the given node type, input node and predicate.

        :param type_names: node type. For example: ["opset8.Abs", "opset8.Relu"]
        :type type_names: List[str]

        :param input: input node.
        :type input: openvino.runtime.Node
    )");

    optional_type.def(
        py::init([](const std::vector<std::string>& type_names, const ov::OutputVector& inputs) {
            return std::make_shared<ov::pass::pattern::op::Optional>(get_types(type_names), inputs, nullptr);
        }),
        py::arg("type_names"),
        py::arg("inputs"),
        R"(
        Create Optional with the given node type and input node.

        :param type_names: node type. For example: ["opset8.Abs", "opset8.Relu"]
        :type type_names: List[str]

        :param inputs: input node's output list.
        :type inputs: List[openvino.runtime.Output]
    )");

    optional_type.def(py::init([](const std::vector<std::string>& type_names, const ov::NodeVector& inputs) {
                          return std::make_shared<ov::pass::pattern::op::Optional>(get_types(type_names),
                                                                                   ov::as_output_vector(inputs),
                                                                                   nullptr);
                      }),
                      py::arg("type_names"),
                      py::arg("inputs"),
                      R"(
        Create Optional with the given node type and input node.

        :param type_names: node type. For example: ["opset8.Abs", "opset8.Relu"]
        :type type_names: List[str]

        :param inputs: input node list
        :type inputs: List[openvino.runtime.Node]
    )");

    optional_type.def(py::init([](const std::vector<std::string>& type_names, const Predicate& predicate) {
                          return std::make_shared<ov::pass::pattern::op::Optional>(get_types(type_names),
                                                                                   ov::OutputVector{},
                                                                                   predicate);
                      }),
                      py::arg("type_names"),
                      py::arg("predicate"),
                      R"(
        Create Optional with the given node type and predicate.

        :param type_names: node type. For example: ["opset8.Abs", "opset8.Relu"]
        :type type_names: List[str]

        :param predicate: Function that performs additional checks for matching.
        :type predicate: function
    )");

    optional_type.def(py::init([](const std::vector<std::string>& type_names,
                                  const ov::Output<ov::Node>& input,
                                  const Predicate& predicate) {
                          return std::make_shared<ov::pass::pattern::op::Optional>(get_types(type_names),
                                                                                   ov::OutputVector{input},
                                                                                   predicate);
                      }),
                      py::arg("type_names"),
                      py::arg("input"),
                      py::arg("predicate"),
                      R"(
        Create Optional with the given node type, input node and predicate.

        :param type_names: node type. For example: ["opset8.Abs", "opset8.Relu"]
        :type type_names: List[str]

        :param input: input node's output.
        :type input: openvino.runtime.Output

        :param predicate: Function that performs additional checks for matching.
        :type predicate: function
    )");

    optional_type.def(py::init([](const std::vector<std::string>& type_names,
                                  const std::shared_ptr<ov::Node>& input,
                                  const Predicate& predicate) {
                          return std::make_shared<ov::pass::pattern::op::Optional>(get_types(type_names),
                                                                                   ov::as_output_vector({input}),
                                                                                   predicate);
                      }),
                      py::arg("type_names"),
                      py::arg("input"),
                      py::arg("predicate"),
                      R"(
        Create Optional with the given node type, input node and predicate.

        :param type_names: node type. For example: ["opset8.Abs", "opset8.Relu"]
        :type type_names: List[str]

        :param input: input node
        :type input: openvino.runtime.Node

        :param predicate: Function that performs additional checks for matching.
        :type predicate: function
    )");

    optional_type.def(
        py::init(
            [](const std::vector<std::string>& type_names, const ov::OutputVector& inputs, const Predicate& predicate) {
                return std::make_shared<ov::pass::pattern::op::Optional>(get_types(type_names), inputs, predicate);
            }),
        py::arg("type_names"),
        py::arg("inputs"),
        py::arg("predicate"),
        R"(
        Create Optional with the given node type, input node and predicate.

        :param type_names: node type. For example: ["opset8.Abs", "opset8.Relu"]
        :type type_names: List[str]

        :param inputs: input node's output list.
        :type inputs: List[openvino.runtime.Output]

        :param predicate: Function that performs additional checks for matching.
        :type predicate: function
    )");

    optional_type.def(
        py::init([](const std::vector<std::string>& type_names, const ov::NodeVector& inputs, const Predicate& pred) {
            return std::make_shared<ov::pass::pattern::op::Optional>(get_types(type_names),
                                                                     ov::as_output_vector(inputs),
                                                                     pred);
        }),
        py::arg("type_names"),
        py::arg("inputs"),
        py::arg("predicate"),
        R"(
        Create Optional with the given node type, input node and predicate.

        :param type_names: node type. For example: ["opset8.Abs", "opset8.Relu"]
        :type type_names: List[str]

        :param inputs: input node list
        :type inputs: List[openvino.runtime.Node]

        :param predicate: Function that performs additional checks for matching.
        :type predicate: function
    )");

    optional_type.def("__repr__", [](const ov::pass::pattern::op::Optional& self) {
        return Common::get_simple_repr(self);
    });
}

inline void reg_predicates(py::module m) {
    m.def("consumers_count", &ov::pass::pattern::consumers_count);
    m.def("has_static_dim", &ov::pass::pattern::has_static_dim);
    m.def("has_static_dims", &ov::pass::pattern::has_static_dims);
    m.def("has_static_shape", &ov::pass::pattern::has_static_shape);
    m.def("has_static_rank", &ov::pass::pattern::has_static_rank);
    m.def("rank_equals", &ov::pass::pattern::rank_equals);
    m.def("type_matches", &ov::pass::pattern::type_matches);
    m.def("type_matches_any", &ov::pass::pattern::type_matches_any);
}

void reg_passes_pattern_ops(py::module m) {
    reg_pattern_any_input(m);
    reg_pattern_wrap_type(m);
    reg_pattern_or(m);
    reg_pattern_optional(m);
    reg_predicates(m);
}
