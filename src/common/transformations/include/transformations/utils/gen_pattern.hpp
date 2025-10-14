// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#ifdef _MSC_VER
#    pragma warning(disable : 4244)
#endif

#include <climits>
#include <sstream>

#include "openvino/core/log_util.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/opsets/opset1_decl.hpp"
#include "openvino/opsets/opset2_decl.hpp"
#include "openvino/opsets/opset3_decl.hpp"
#include "openvino/opsets/opset4_decl.hpp"
#include "openvino/opsets/opset5_decl.hpp"
#include "openvino/opsets/opset6_decl.hpp"
#include "openvino/opsets/opset7_decl.hpp"
#include "openvino/opsets/opset8_decl.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/log.hpp"

namespace ov {
namespace gen_pattern {

#ifdef CPU_DEBUG_CAPS
template <typename... Args>
static inline void _verbose_log(Args&&... args) {
    std::stringstream ss;
    int dummy[] = {(ss << std::forward<Args>(args) << " ", 0)...};
    (void)(dummy);
    ss << std::endl;
    std::cout << ss.str();
}

static bool matcher_verbose_enabled() {
    static const bool enabled = std::getenv("GENP_VERBOSE") ? (atoi(std::getenv("GENP_VERBOSE")) != 0) : false;
    return enabled;
}

#    define _VERBOSE_LOG(...)          \
        if (matcher_verbose_enabled()) \
        _verbose_log(__VA_ARGS__)
#else
#    define _VERBOSE_LOG(...)
#endif

namespace detail {
// AttrAny is simple wrapper of Any to provide some constructor
// to take advantage of C++ implicit conversion to allow:
//   - attribute expressed using initializer_list.
struct AttrAny {
    ov::Any any;

    // empty attribute, means empty vector, and error for scalar
    AttrAny() {}

    AttrAny(const ov::element::Type& v) : any(v) {}
    AttrAny(const ov::PartialShape& v) : any(v) {}
    AttrAny(const ov::Dimension& v) : any(v) {}
    AttrAny(bool v) : any(v) {}
    AttrAny(int v) : any(v) {}
    AttrAny(float v) : any(v) {}
    AttrAny(double v) : any(v) {}
    AttrAny(long v) : any(static_cast<int64_t>(v)) {}
    AttrAny(long long v) : any(static_cast<int64_t>(v)) {}
    AttrAny(const char* v) : any(v) {}
    AttrAny(const std::string& v) : any(v) {}

    AttrAny(const std::vector<int64_t>& v) : any(v) {}

    AttrAny(std::initializer_list<int> values) : any(std::vector<int>(values)) {}
    AttrAny(std::initializer_list<long> values) : any(std::vector<int64_t>(values.begin(), values.end())) {}
    AttrAny(std::initializer_list<float> values) : any(std::vector<float>(values)) {}
    AttrAny(std::initializer_list<double> values) : any(std::vector<double>(values)) {}
    AttrAny(std::initializer_list<long long> values) : any(std::vector<int64_t>(values.begin(), values.end())) {}

    AttrAny(std::initializer_list<std::string> values) : any(std::vector<std::string>(values)) {}
    AttrAny(std::initializer_list<const char*> values) : any(std::vector<const char*>(values)) {}

    std::string as_string() {
        if (any.is<const char*>())
            return any.as<const char*>();
        return any.as<std::string>();
    }
    bool as_bool() {
        if (any.is<int>())
            return any.as<int>();
        return any.as<bool>();
    }
    double as_double() {
        if (any.is<float>())
            return any.as<float>();
        if (any.is<int>())
            return any.as<int>();
        return any.as<double>();
    }
    int64_t as_int64_t() {
        if (any.is<int>())
            return any.as<int>();
        return any.as<int64_t>();
    }

    template <typename T>
    std::vector<T> as_vector() {
        if (any.empty())
            return {};
        if (!std::is_same<T, int>::value) {
            if (any.is<std::initializer_list<int>>()) {
                auto ivec = any.as<std::initializer_list<int>>();
                return std::vector<T>(ivec.begin(), ivec.end());
            }
            if (any.is<std::vector<int>>()) {
                auto vec = any.as<std::vector<int>>();
                return std::vector<T>(vec.begin(), vec.end());
            }
        }
        if (!std::is_same<T, float>::value) {
            if (any.is<std::initializer_list<float>>()) {
                auto ivec = any.as<std::initializer_list<float>>();
                return std::vector<T>(ivec.begin(), ivec.end());
            }
            if (any.is<std::vector<float>>()) {
                auto vec = any.as<std::vector<float>>();
                return std::vector<T>(vec.begin(), vec.end());
            }
        }
        if (any.is<std::initializer_list<T>>()) {
            auto ivec = any.as<std::initializer_list<T>>();
            return std::vector<T>(ivec.begin(), ivec.end());
        }
        return any.as<std::vector<T>>();
    }

    template <typename T>
    std::vector<T> as_T_vector() {
        if (any.empty())
            return {};
        if (any.is<T>()) {
            auto to_vec = [](std::initializer_list<T> v) {
                return std::vector<T>(v);
            };
            return to_vec({any.as<T>()});
        }
        if (any.is<std::initializer_list<T>>()) {
            auto ivec = any.as<std::initializer_list<T>>();
            return std::vector<T>(ivec.begin(), ivec.end());
        }
        return any.as<std::vector<T>>();
    }

    std::vector<std::string> as_str_vector() {
        if (any.empty())
            return {};
        if (any.is<std::vector<const char*>>()) {
            auto vec = any.as<std::vector<const char*>>();
            return std::vector<std::string>(vec.begin(), vec.end());
        }
        return any.as<std::vector<std::string>>();
    }

    template <typename T>
    T cast_to() {
        if (any.is<bool>())
            return static_cast<T>(any.as<bool>());
        if (any.is<int>())
            return static_cast<T>(any.as<int>());
        if (any.is<long>())
            return static_cast<T>(any.as<long>());
        if (any.is<long long>())
            return static_cast<T>(any.as<long long>());
        if (any.is<int32_t>())
            return static_cast<T>(any.as<int32_t>());
        if (any.is<int64_t>())
            return static_cast<T>(any.as<int64_t>());
        if (any.is<float>())
            return static_cast<T>(any.as<float>());
        if (any.is<double>())
            return static_cast<T>(any.as<double>());
        if (any.is<int8_t>())
            return static_cast<T>(any.as<int8_t>());
        if (any.is<uint8_t>())
            return static_cast<T>(any.as<uint8_t>());
        return any.as<T>();
    }

    template <typename T>
    bool equal_to(const std::vector<T>& rhs) {
        if (any.empty())
            return rhs.empty();
        auto& vec = any.as<std::vector<T>>();
        return std::equal(vec.begin(), vec.end(), rhs.begin());
    }

    template <typename T, typename CT0, typename... CTs>
    bool equal_to(const std::vector<T>& rhs) {
        if (any.empty())
            return rhs.empty();

        if (any.is<std::vector<CT0>>()) {
            auto& vec = any.as<std::vector<CT0>>();
            return vec.size() == rhs.size() && std::equal(vec.begin(), vec.end(), rhs.begin());
        }
        return equal_to<T, CTs...>(rhs);
    }

    template <typename T>
    typename std::enable_if<std::is_arithmetic<T>::value, bool>::type equal_to(const T& rhs) {
        return rhs == any.as<T>();
    }

    template <typename T, typename CT0, typename... CTs>
    typename std::enable_if<std::is_arithmetic<T>::value, bool>::type equal_to(const T& rhs) {
        if (any.is<CT0>()) {
            auto& value = any.as<CT0>();
            return rhs == static_cast<T>(value);
        }
        return equal_to<T, CTs...>(rhs);
    }
};

using AttrMap = std::map<std::string, AttrAny>;

class AttrSetter : public ov::AttributeVisitor {
public:
    AttrMap& m_attr_map;
    std::vector<std::string> m_missing_attrs;

    AttrSetter(AttrMap& attrs) : m_attr_map(attrs) {}

    const std::vector<std::string>& get_missing_attrs() {
        return m_missing_attrs;
    }

    bool should_skip(const std::string& name) {
        if (m_attr_map.count(name) == 0) {
            // attributes not specified is recorded as missing
            m_missing_attrs.push_back(name);
            return true;
        }

        if (m_attr_map[name].any.empty()) {
            // input is set to empty, meaning default value is used.
            return true;
        }
        return false;
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<std::string>& value) override {
        if (should_skip(name))
            return;
        value.set(m_attr_map[name].as_string());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<bool>& value) override {
        if (should_skip(name))
            return;
        value.set(m_attr_map[name].as_bool());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<void>& adapter) override {
        if (should_skip(name))
            return;
        auto& any = m_attr_map[name].any;
        if (auto a = ov::as_type<ov::AttributeAdapter<ov::element::Type>>(&adapter)) {
            static_cast<ov::element::Type&>(*a) = any.as<ov::element::Type>();
        } else if (auto a = ov::as_type<ov::AttributeAdapter<ov::PartialShape>>(&adapter)) {
            a->set(any.as<ov::PartialShape>());
        } else if (auto a = ov::as_type<ov::AttributeAdapter<ov::Dimension>>(&adapter)) {
            a->set(any.as<ov::Dimension>());
        } else if (auto a = ov::as_type<ov::AttributeAdapter<ov::Shape>>(&adapter)) {
            a->set(m_attr_map[name].as_vector<int64_t>());
        } else if (auto a = ov::as_type<ov::AttributeAdapter<ov::Strides>>(&adapter)) {
            a->set(m_attr_map[name].as_vector<int64_t>());
        } else if (auto a = ov::as_type<ov::AttributeAdapter<std::vector<size_t>>>(&adapter)) {
#if defined(__APPLE__) || defined(__EMSCRIPTEN__)
            static_cast<std::vector<size_t>&>(*a) = m_attr_map[name].as_vector<size_t>();
#else
            a->set(m_attr_map[name].as_vector<size_t>());
#endif
        } else if (auto a = ov::as_type<ov::AttributeAdapter<ov::AxisSet>>(&adapter)) {
            a->set(m_attr_map[name].as_vector<int64_t>());
            //} else if (auto a = ov::as_type<ValueAccessor<std::string>>(&adapter)) {
            //    a->set(m_attr_map[name].as_string());
        } else if (auto a = ov::as_type<ov::AttributeAdapter<ov::op::TopKMode>>(&adapter)) {
            a->set(m_attr_map[name].as_string());
        } else if (auto a = ov::as_type<ov::AttributeAdapter<ov::CoordinateDiff>>(&adapter)) {
            a->set(m_attr_map[name].as_vector<int64_t>());
        } else if (auto a = ov::as_type<ov::AttributeAdapter<ov::element::TypeVector>>(&adapter)) {
            a->set(m_attr_map[name].as_T_vector<ov::element::Type>());
        } else if (auto a = dynamic_cast<ov::AttributeAdapter<std::shared_ptr<ov::op::util::Variable>>*>(&adapter)) {
            ov::op::util::VariableInfo var_info;
            var_info.variable_id = m_attr_map[name].as_string();
            auto variable = std::make_shared<ov::op::util::Variable>(var_info);
            a->set(variable);
        } else {
            OPENVINO_THROW("unsupported AttributeAdapter for attribute : ", name);
        }
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<double>& value) override {
        if (should_skip(name))
            return;
        value.set(m_attr_map[name].as_double());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<int64_t>& value) override {
        if (should_skip(name))
            return;
        value.set(m_attr_map[name].as_int64_t());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<int32_t>>& value) override {
        if (should_skip(name))
            return;
        value.set(m_attr_map[name].as_vector<int32_t>());
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<int64_t>>& value) override {
        if (should_skip(name))
            return;
        value.set(m_attr_map[name].as_vector<int64_t>());
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<float>>& value) override {
        if (should_skip(name))
            return;
        value.set(m_attr_map[name].as_vector<float>());
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<std::string>>& value) override {
        if (should_skip(name))
            return;
        value.set(m_attr_map[name].as_str_vector());
    }
};

// A glue/syntax-sugar type which allows more types to be used as input to makePattern()
struct PatternNode {
    std::shared_ptr<Node> node;
    int output_port = -1;

    operator ov::Output<ov::Node>() const {
        return get_output();
    }

    ov::Output<ov::Node> get_output() const {
        if (output_port >= 0)
            return node->output(output_port);
        return node->get_default_output();
    }

    PatternNode(const Output<Node>& out)
        : node(out.get_node_shared_ptr()),
          output_port(static_cast<int>(out.get_index())) {}

    PatternNode() {
        node = ov::pass::pattern::any_input(ov::pass::pattern::has_static_rank());
    }
    PatternNode(ov::Rank rank) {
        node = ov::pass::pattern::any_input([rank](const Output<Node>& value) {
            if (!rank.compatible(value.get_partial_shape().rank())) {
                _VERBOSE_LOG("*mismatched PatternNode rank ", value, " expecting ", rank);
                return false;
            }
            return true;
        });
    }

    PatternNode(const std::shared_ptr<Node>& node) : node(node) {}
    PatternNode(const std::shared_ptr<ov::op::v0::Parameter>& node) : node(node) {}
    PatternNode(const std::shared_ptr<ov::pass::pattern::op::Or>& pattern)
        : node(std::dynamic_pointer_cast<Node>(pattern)) {}

    // scalar constant (treated as wildcard for single-element-constant with any rank)
    PatternNode(int v) : node(std::make_shared<ov::op::v0::Constant>(element::from<int>(), Shape({}), v)) {}
    PatternNode(float v) : node(std::make_shared<ov::op::v0::Constant>(element::from<float>(), Shape({}), v)) {}
    PatternNode(long long v) : node(std::make_shared<ov::op::v0::Constant>(element::from<int64_t>(), Shape({}), v)) {}

    PatternNode(std::initializer_list<int> v) {
        node = ConstVector(std::vector<int>(v));
    }
    PatternNode(std::initializer_list<float> v) {
        node = ConstVector(std::vector<float>(v));
    }
    PatternNode(std::initializer_list<long> v) {
        node = ConstVector(std::vector<int64_t>(v.begin(), v.end()));
    }
    PatternNode(std::initializer_list<long long> v) {
        node = ConstVector(std::vector<int64_t>(v.begin(), v.end()));
    }

    // 1d const tensor or scalar
    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value, bool>::type = true>
    static std::shared_ptr<Node> ConstVector(const std::vector<T>& vec) {
        return std::make_shared<ov::op::v0::Constant>(element::from<T>(), Shape({vec.size()}), vec);
    }
};
}  // namespace detail

//==================================================================================================

// unknown const
template <typename T>
std::shared_ptr<Node> makeConst(const ov::element::Type& type,
                                const ov::Shape& shape,
                                std::initializer_list<T> values) {
    return std::make_shared<ov::op::v0::Constant>(type, shape, std::vector<T>(values));
}

template <typename T>
std::shared_ptr<Node> makeConst(const ov::element::Type& type, const ov::Shape& shape, const std::vector<T>& values) {
    return std::make_shared<ov::op::v0::Constant>(type, shape, values);
}

template <class T>
std::shared_ptr<Node> makeOP(const std::vector<detail::PatternNode>& inputs,
                             detail::AttrMap attrmap = {},
                             const char* friendly_name = nullptr) {
    std::shared_ptr<Node> node = std::make_shared<T>();

    OutputVector args;
    for (auto& in : inputs)
        args.push_back(in.get_output());
    node->set_arguments(args);

    detail::AttrSetter visitor(attrmap);
    node->visit_attributes(visitor);

    auto missing_attrs = visitor.get_missing_attrs();

    // when some attribute is missing, the returned
    // node is suitable for pattern matching only.
    OPENVINO_ASSERT(missing_attrs.size() == 0,
                    "missing ",
                    missing_attrs.size(),
                    " attributes : ",
                    missing_attrs[0],
                    "...");

    if (friendly_name)
        node->set_friendly_name(friendly_name);
    node->constructor_validate_and_infer_types();
    return node;
}

}  // namespace gen_pattern
}  // namespace ov
