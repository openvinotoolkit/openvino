// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <fstream>
#include <memory>
#include <ostream>
#include <string>
#include <type_traits>
#include <vector>

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "openvino/pass/pass.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace pass {

namespace detail {

// to_code convert value into literal/constexpr/initializer_list/factory_calls in C++ source code
inline std::string to_code(bool value) {
    return value ? "true" : "false";
}
inline std::string to_code(const std::string& value) {
    return std::string("\"") + value + "\"";
}
inline std::string to_code(const element::Type& value) {
    return std::string("element::") + value.to_string();
}
inline std::string to_code(const ov::Shape& value) {
    std::stringstream ss;
    ss << "ov::Shape({";
    for (auto& d : value)
        ss << d << ",";
    ss << "})";
    return ss.str();
}
inline std::string to_code(int value) {
    if (INT_MAX == value) {
        return "INT_MAX";
    }
    if (INT_MIN == value) {
        return "INT_MIN";
    }
    return std::to_string(value);
}
inline std::string to_code(int64_t value) {
    if (LLONG_MAX == value) {
        return "LLONG_MAX";
    }
    if (LLONG_MIN == value) {
        return "LLONG_MIN";
    }
    const char* suffix = "LL";
    if (value == static_cast<int64_t>(static_cast<int>(value))) {
        // save suffix since most values can be expressed as int
        // this produces more readable code
        suffix = "";
    }
    return std::to_string(value) + suffix;
}
inline std::string to_code(uint64_t value) {
    if (ULLONG_MAX == value) {
        return "ULLONG_MAX";
    }
    const char* suffix = "uLL";
    if (value == static_cast<uint64_t>(static_cast<int>(value))) {
        // save suffix since most values can be expressed as int
        // this produces more readable code
        suffix = "";
    }
    return std::to_string(value) + suffix;
}
inline std::string to_code(int8_t value) {
    return std::to_string(static_cast<int>(value));
}
inline std::string to_code(uint8_t value) {
    return std::to_string(static_cast<int>(value));
}

template <typename T>
std::string to_code_float(T value) {
    if (std::isnan(value)) {
        return "NAN";
    } else if (std::isinf(value)) {
        return (value > 0 ? "INFINITY" : "-INFINITY");
    } else if (value == FLT_MIN) {
        return "FLT_MIN";
    } else if (value == -FLT_MIN) {
        return "-FLT_MIN";
    } else if (value == FLT_MAX) {
        return "FLT_MAX";
    } else if (value == -FLT_MAX) {
        return "-FLT_MAX";
    }
    auto strv = std::to_string(value);
    if (strv.find(".") == std::string::npos && strv.find("e") == std::string::npos)
        strv += ".0";
    if (std::is_same<T, float>::value)
        strv += "f";
    return strv;
}

inline std::string to_code(float value) {
    return to_code_float(value);
}
inline std::string to_code(double value) {
    return to_code_float(value);
}
template <typename T>
std::string to_code(const std::vector<T>& values, bool no_braces = false, int maxsize = 80) {
    std::stringstream ss;
    if (!no_braces)
        ss << "{";
    const char* sep = "";
    for (auto& v : values) {
        if (ss.tellp() > maxsize) {
            ss << "... (" << values.size() << " in total)";
            break;
        }
        ss << sep << to_code(v);
        sep = ",";
    }
    if (!no_braces)
        ss << "}";
    return ss.str();
}

template <typename T = void>
std::string to_code(std::shared_ptr<ov::op::v0::Constant> constop, bool force_braces = false) {
    bool no_braces = (constop->get_shape().size() == 0) && (!force_braces);
    auto ele_type = constop->get_element_type();
    if (ele_type == element::Type_t::f32) {
        return to_code(constop->get_vector<float>(), no_braces);
    } else if (ele_type == element::Type_t::i8) {
        return to_code(constop->get_vector<int8_t>(), no_braces);
    } else if (ele_type == element::Type_t::u8) {
        return to_code(constop->get_vector<uint8_t>(), no_braces);
    } else if (ele_type == element::Type_t::i32) {
        return to_code(constop->get_vector<int32_t>(), no_braces);
    } else if (ele_type == element::Type_t::i64) {
        return to_code(constop->get_vector<int64_t>(), no_braces);
    }

    // general case
    std::stringstream ss;
    if (!no_braces)
        ss << "{";
    auto ele_size = shape_size(constop->get_shape());
    if (ele_size < 9) {
        const char* sep = "";
        for (auto v : constop->get_value_strings()) {
            ss << sep << v;
            sep = ", ";
        }
    } else {
        ss << "...";
    }
    if (!no_braces)
        ss << "}";
    return ss.str();
}

class OstreamAttributeVisitor : public ov::AttributeVisitor {
    std::ostream& os;
    const char* sep = "";

public:
    OstreamAttributeVisitor(std::ostream& os) : os(os) {}

    void append_attribute(const std::string& name, const std::string& value) {
        os << sep << "{\"" << name << "\", " << value << "}";
        sep = ", ";
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<void>& adapter) override {
        if (auto a = ov::as_type<ov::AttributeAdapter<std::set<std::string>>>(&adapter)) {
            const auto& strset = a->get();
            std::vector<std::string> values(strset.begin(), strset.end());
            append_attribute(name, to_code(values));
        } else if (auto a = ov::as_type<ov::AttributeAdapter<std::vector<ov::element::Type>>>(&adapter)) {
            append_attribute(name, to_code(a->get()));
        } else if (auto a = ov::as_type<ov::AttributeAdapter<ov::PartialShape>>(&adapter)) {
            const auto& value = a->get();
            append_attribute(name, value.to_string());
        } else if (auto a = ov::as_type<ov::AttributeAdapter<std::shared_ptr<ov::op::util::Variable>>>(&adapter)) {
            const auto& vinfo = a->get()->get_info();
            std::stringstream ss;
            ss << vinfo.variable_id << vinfo.data_shape << vinfo.data_type;
            append_attribute(name, ss.str());
        } else {
            append_attribute(name, "?");
        }
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<bool>& adapter) override {
        append_attribute(name, to_code(adapter.get()));
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::string>& adapter) override {
        append_attribute(name, to_code(adapter.get()));
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<int64_t>& adapter) override {
        append_attribute(name, to_code(adapter.get()));
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<double>& adapter) override {
        append_attribute(name, to_code(adapter.get()));
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<int32_t>& adapter) override {
        append_attribute(name, to_code(adapter.get()));
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<float>& adapter) override {
        append_attribute(name, to_code(adapter.get()));
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<int>>& adapter) override {
        append_attribute(name, to_code(adapter.get()));
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<int64_t>>& adapter) override {
        append_attribute(name, to_code(adapter.get()));
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<uint64_t>>& adapter) override {
        append_attribute(name, to_code(adapter.get()));
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<float>>& adapter) override {
        append_attribute(name, to_code(adapter.get()));
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<std::string>>& adapter) override {
        append_attribute(name, to_code(adapter.get()));
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::shared_ptr<ov::Model>>& adapter) override {
        append_attribute(name, "Model");
    }
};

template <typename UNUSED_T = void>
void dump_cpp_style(std::ostream& os, const std::shared_ptr<ov::Model>& model) {
    const ov::Model& f = *model;
    std::string prefix = "";
    std::string tag = "";
    std::string sep = "";
    os << prefix;
    for (auto op : f.get_results()) {
        os << sep << op->get_name();
        sep = ",";
    }
    os << " " << f.get_friendly_name() << "(\n" << prefix;
    for (auto op : f.get_parameters()) {
        os << "    " << tag << op->get_friendly_name() << ",\n" << prefix;
    }
    os << ") {\n";

    // collect all scalar & short 1D vectors for literal-style display
    std::map<std::shared_ptr<ov::Node>, std::string> literal_consts;
    for (auto op : f.get_ordered_ops()) {
        if (auto constop = ov::as_type_ptr<op::v0::Constant>(op)) {
            // only i32/f32 type const literal can be parsed by C++ compiler
            if (constop->get_output_element_type(0) != ov::element::i32 &&
                constop->get_output_element_type(0) != ov::element::i64 &&
                constop->get_output_element_type(0) != ov::element::f32)
                continue;
            auto shape = constop->get_shape();
            if (shape.size() > 1)
                continue;
            if (shape_size(constop->get_shape()) > 64)
                continue;
            literal_consts[op] = to_code(constop);
        }
    }

    auto get_output_values_info = [](std::shared_ptr<ov::Node>& op) {
        std::stringstream ss;
        const char* sep = "";
        for (size_t i = 0; i < op->get_output_size(); i++) {
            ss << sep << op->get_output_element_type(i) << op->get_output_partial_shape(i);
            sep = " ";
        }
        return ss.str();
    };

    // change name convension
    std::map<ov::Node*, std::string> opname;
    std::map<std::string, int> opname_count;
    for (auto op : f.get_ordered_ops()) {
        auto name = op->get_friendly_name();
        std::replace(name.begin(), name.end(), '\\', '_');
        std::replace(name.begin(), name.end(), '/', '_');
        std::replace(name.begin(), name.end(), '.', '_');
        std::replace(name.begin(), name.end(), '[', '_');
        std::replace(name.begin(), name.end(), ']', '_');
        std::replace(name.begin(), name.end(), '-', 'n');
        if (name[0] >= '0' && name[0] <= '9') {
            const auto& type_info = op->get_type_info();
            name.insert(0, type_info.name);
        }
        int idx = 0;
        if (opname_count.count(name)) {
            idx = opname_count[name] + 1;
        }
        opname_count[name] = idx;

        if (idx)
            name += std::to_string(idx);

        opname[op.get()] = name;
    }

    for (auto op : f.get_ordered_ops()) {
        if (literal_consts.count(op))
            continue;

        const auto& type_info = op->get_type_info();
        auto version_info = std::string(type_info.get_version());
        auto type = version_info + "::" + type_info.name;
        auto& rt_info = op->get_rt_info();
        if (rt_info.count("opset") && rt_info["opset"] == "type_relaxed_opset") {
            type = std::string("ov::op::TypeRelaxed<") + type + ">";
        }
        auto name = opname[op.get()];
        os << prefix << "    ";

        if (auto constop = ov::as_type_ptr<op::v0::Constant>(op)) {
            os << "auto " << name << " = makeConst(" << to_code(op->get_output_element_type(0)) << ", "
               << to_code(op->get_output_shape(0)) << ", " << to_code(constop, true) << ");" << std::endl;
        } else {
            os << "auto " << name << " = makeOP<" << type << ">({";
            // input args
            sep = "";
            for (size_t i = 0; i < op->get_input_size(); i++) {
                auto vout = op->get_input_source_output(i);
                auto iop = vout.get_node_shared_ptr();
                if (iop->get_output_size() > 1) {
                    auto out_port = vout.get_index();
                    os << sep << tag << opname[iop.get()] << "->output(" << out_port << ")";
                } else {
                    if (literal_consts.count(iop))
                        os << sep << tag << literal_consts[iop];
                    else
                        os << sep << tag << opname[iop.get()];
                }
                sep = ", ";
            }
            os << "}";

            // attributes as AnyMap
            std::stringstream ss2;
            OstreamAttributeVisitor osvis(ss2);
            op->visit_attributes(osvis);
            auto str_attr = ss2.str();
            if (str_attr.size())
                os << ", {" << str_attr << "}";
            os << ");   //  tensor_array<" << get_output_values_info(op) << "> " << op->get_friendly_name();

            os << "(";
            sep = "";
            for (size_t i = 0; i < op->get_input_size(); i++) {
                auto vout = op->get_input_source_output(i);
                auto iop = vout.get_node_shared_ptr();
                os << sep << tag << iop->get_friendly_name();
                if (iop->get_output_size() > 1) {
                    auto out_port = vout.get_index();
                    os << "[" << out_port << "]";
                }
                sep = ", ";
            }
            os << ")" << std::endl;
        }

        // recursively output subgraphs
        if (auto msubgraph = ov::as_type_ptr<op::util::MultiSubGraphOp>(op)) {
            auto cnt = msubgraph->get_internal_subgraphs_size();
            for (size_t i = 0; i < cnt; i++) {
                os << "    MultiSubGraphOp " << tag << msubgraph->get_friendly_name() << "[" << i << "]" << std::endl;
                dump_cpp_style(os, msubgraph->get_function(i));
            }
        }
    }
    os << prefix << "}\n";
}

}  // namespace detail

class OPENVINO_API PrintModel : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::pass::PrintModel");

    PrintModel(std::string file_name) {
        static int dump_index = 0;
        m_file_name = std::string("modelprint_") + std::to_string(dump_index) + "_" + file_name;
        dump_index++;
    }
    ~PrintModel() {}

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override {
        if (m_file_name.empty())
            return false;

        for (auto& node : model->get_ordered_ops()) {
            ov::op::util::process_subgraph(*this, node);
        }

        std::ofstream ofs(m_file_name);
        if (!ofs) {
            return false;
        }
        detail::dump_cpp_style(ofs, model);
        ofs.close();
        return true;
    }

protected:
    std::string m_file_name;
};
}  // namespace pass
}  // namespace ov
