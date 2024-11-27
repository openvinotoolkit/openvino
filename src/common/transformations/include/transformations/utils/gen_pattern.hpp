// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#ifdef _MSC_VER
#    pragma warning(disable : 4244)
#endif

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset2.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/opsets/opset4.hpp"
#include "openvino/opsets/opset5.hpp"
#include "openvino/opsets/opset6.hpp"
#include "openvino/opsets/opset7.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

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
static bool matcher_verbose_enabled() {
    return false;
}

#    define _VERBOSE_LOG(...)
#endif

namespace detail {
inline std::vector<std::string> split_string(const std::string& s, const std::string& delimiter) {
    std::vector<std::string> ret;
    size_t pos = 0, pos_next;
    std::string token;
    while ((pos_next = s.find(delimiter, pos)) != std::string::npos) {
        token = s.substr(pos, pos_next - pos);
        ret.push_back(token);
        pos = pos_next + 1;
    }
    // return whole string if no delimiter if found
    token = s.substr(pos, pos_next);
    ret.push_back(token);
    return ret;
}

template <typename T>
std::string vec2str(const std::vector<T>& vec, int cnt_limit = 9) {
    std::stringstream ss;
    ss << "{";
    const char* sep = "";
    for (auto& v : vec) {
        cnt_limit--;
        if (cnt_limit == 0) {
            ss << sep << "...";
            break;
        }
        ss << sep << v;
        sep = ",";
    }
    ss << "}";
    return ss.str();
}
}  // namespace detail

struct values_info {
    values_info(const char* pattern_list = nullptr) {
        if (pattern_list == nullptr || pattern_list[0] == 0) {
            all_type_pshape.clear();
            return;
        }
        auto pattern_vector = detail::split_string(pattern_list, " ");
        for (auto& pattern : pattern_vector) {
            if (pattern[0] == '[') {
                all_type_pshape.emplace_back(ov::element::dynamic, ov::PartialShape(pattern));
            } else {
                auto sep = pattern.find("[");
                if (sep != std::string::npos) {
                    // ele_type[p_shape]
                    all_type_pshape.emplace_back(ov::element::Type(pattern.substr(0, sep)),
                                                 ov::PartialShape(pattern.substr(sep)));
                } else {
                    // ele_type
                    all_type_pshape.emplace_back(ov::element::Type(pattern), ov::PartialShape::dynamic());
                }
            }
        }
    }

    size_t size() {
        return all_type_pshape.size();
    }
    const std::pair<ov::element::Type, ov::PartialShape>& operator[](int index) {
        return all_type_pshape[index];
    }

    //-------------------------------------------------------------
    bool predicate(const ov::Output<ov::Node>& value) const {
        if (all_type_pshape.empty())
            return true;
        auto index = value.get_index();
        if (index >= all_type_pshape.size()) {
            _VERBOSE_LOG("* mismatched vtype : value from output port ",
                         index,
                         ", but only ",
                         all_type_pshape.size(),
                         " ports are expected!");
            return false;
        }
        auto& item = all_type_pshape[index];
        if (!item.first.compatible(value.get_element_type()) || !item.second.compatible(value.get_partial_shape())) {
            _VERBOSE_LOG("* mismatched vtype between value & pattern : ",
                         value.get_element_type(),
                         value.get_partial_shape(),
                         "vs",
                         item.first,
                         item.second);
            return false;
        }
        return true;
    }

    std::string to_string() {
        std::stringstream ss;
        const char* sep = "";
        for (auto& t : all_type_pshape) {
            ss << sep << t.first << t.second;
            sep = ";";
        }
        return ss.str();
    }

    std::vector<std::pair<ov::element::Type, ov::PartialShape>> all_type_pshape;
};

// Symbol : a constant that unknown at the pattern's building time
//          but collected and validated after pattern was matched
//          with some sub-graph values.
class Symbol {
private:
    struct Entity {
        const char* name = "?";
        char op;
        double literal_const_value;
        std::shared_ptr<Entity> lhs;
        std::shared_ptr<Entity> rhs;
        // _,+,-,*,/
        // l : literal const
        // n : named symbol
        double eval(const std::map<const void*, double>& value_map) const {
            switch (op) {
            case 'l':
                return literal_const_value;
            case 'n':
                return value_map.at(this);
            case '+':
                return lhs->eval(value_map) + rhs->eval(value_map);
            case '-':
                return lhs->eval(value_map) - rhs->eval(value_map);
            case '*':
                return lhs->eval(value_map) * rhs->eval(value_map);
            case '/':
                return lhs->eval(value_map) / rhs->eval(value_map);
            case '_':
                return -lhs->eval(value_map);
            case 'r':
                return std::sqrt(lhs->eval(value_map));
            default:
                assert(false);
                return std::numeric_limits<double>::quiet_NaN();
            }
        }
    };
    std::shared_ptr<Entity> entity;

public:
    Symbol() {
        entity = std::make_shared<Entity>();
        entity->op = 'n';
    }
    Symbol(const char* name) {
        entity = std::make_shared<Entity>();
        entity->op = 'n';
        entity->name = name;
    }
    Symbol(const int value) {
        entity = std::make_shared<Entity>();
        entity->op = 'l';
        entity->literal_const_value = value;
    }
    Symbol(char op, const Symbol& lhs, const Symbol& rhs) {
        entity = std::make_shared<Entity>();
        entity->op = op;
        entity->lhs = lhs.entity;
        entity->rhs = rhs.entity;
    }
    double eval(const std::map<const void*, double>& value_map) const {
        return entity->eval(value_map);
    }
    bool is_independent_var() const {
        return entity->op == 'n';
    }
    int is_literal_const() const {
        return entity->op == 'l';
    }
    char get_op() const {
        return entity->op;
    }
    void* get_id() const {
        return entity.get();
    }
    const char* get_name() const {
        return entity->name;
    }
    bool operator<(const Symbol& rhs) const {
        return get_id() < rhs.get_id();
    }
};

inline Symbol operator-(const Symbol& lhs) {
    return Symbol('_', lhs, lhs);
}
inline Symbol operator+(const Symbol& lhs, const Symbol& rhs) {
    return Symbol('+', lhs, rhs);
}
inline Symbol operator-(const Symbol& lhs, const Symbol& rhs) {
    return Symbol('-', lhs, rhs);
}
inline Symbol operator*(const Symbol& lhs, const Symbol& rhs) {
    return Symbol('*', lhs, rhs);
}
inline Symbol operator/(const Symbol& lhs, const Symbol& rhs) {
    return Symbol('/', lhs, rhs);
}
inline Symbol sqrt(Symbol lhs) {
    return Symbol('r', lhs, lhs);
}

namespace detail {

using SymbolObservationVector = std::vector<std::pair<Symbol, double>>;

template <typename T>
void add_symbol_observed(SymbolObservationVector& sov, const Symbol& sym, const T& value) {
    auto v = static_cast<double>(value);
    OPENVINO_ASSERT(static_cast<T>(v) == value);  // ensure there is no precison lost in double
    sov.push_back(std::make_pair(sym, v));
}

// AttrAny is simple wrapper of Any to provide some constructor
// to take advantage of C++ implicit conversion to allow:
//   - attribute expressed using initializer_list.
//   - symbol to be used as attributes
struct AttrAny {
    ov::Any any;

    // empty attribute, means empty vector, and error for scalar
    AttrAny() {}

    AttrAny(const Symbol& v) : any(v) {}
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

    // template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type = true>
    // AttrAny(const T& v) : any(v) {}

    // template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type = true>
    // AttrAny(const std::vector<T>& v) : any(v) {}

    AttrAny(const std::vector<int64_t>& v) : any(v) {}

    // template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type = true>
    // AttrAny(std::initializer_list<T> values) : any(std::vector<T>(values)) {}
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

        if (m_attr_map[name].any.is<Symbol>()) {
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

// for arithmetic data type, Attr matcher will success as long as the actuall attributes
// is equal to the casted attributes from pattern w/o requiring exact type match.
class AttrMatcher : public ov::AttributeVisitor {
public:
    AttrMap& m_attr_map;
    std::vector<std::string> m_missing_attrs;
    SymbolObservationVector* m_psov;
    bool m_all_matched;

    AttrMatcher(AttrMap& attrs, SymbolObservationVector* psov = nullptr)
        : m_attr_map(attrs),
          m_psov(psov),
          m_all_matched(true) {}

    bool matched() {
        return m_all_matched;
    }

    const std::vector<std::string>& get_missing_attrs() {
        return m_missing_attrs;
    }

    bool should_skip(const std::string& name, bool allow_symbol = false) {
        if (m_attr_map.count(name) == 0) {
            m_missing_attrs.push_back(name);
            return true;
        }

        if (!allow_symbol) {
            OPENVINO_ASSERT(!m_attr_map[name].any.is<Symbol>(), "Symbol is not allowed.");
        }
        return false;
    }

    void add_match_result(const std::string& name, bool is_matched) {
        if (!is_matched) {
            _VERBOSE_LOG(" attribute '", name, "' mismatch.");
        }
        m_all_matched = m_all_matched && is_matched;
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<std::string>& value) override {
        if (should_skip(name))
            return;
        add_match_result(name, value.get() == m_attr_map[name].as_string());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<bool>& value) override {
        if (should_skip(name))
            return;
        add_match_result(name, m_attr_map[name].equal_to<bool, int, float>(value.get()));
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<double>& value) override {
        if (should_skip(name))
            return;
        add_match_result(name, m_attr_map[name].equal_to<double, int, float>(value.get()));
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<int32_t>>& value) override {
        if (should_skip(name))
            return;
        add_match_result(name, m_attr_map[name].equal_to<int, int64_t>(value.get()));
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<int64_t>>& value) override {
        if (should_skip(name))
            return;
        add_match_result(name, m_attr_map[name].equal_to<int64_t, int>(value.get()));
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<float>>& value) override {
        if (should_skip(name))
            return;
        add_match_result(name, m_attr_map[name].equal_to<float, int>(value.get()));
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<std::string>>& value) override {
        if (should_skip(name))
            return;
        add_match_result(name, m_attr_map[name].equal_to<std::string, const char*>(value.get()));
    }

    // only integer is allowed to be of symbol type
    void on_adapter(const std::string& name, ov::ValueAccessor<int32_t>& value) override {
        if (should_skip(name, true))
            return;
        auto& any = m_attr_map[name].any;
        if (any.is<Symbol>()) {
            if (m_psov) {
                // collect symbol reference and do comparison later
                add_symbol_observed(*m_psov, any.as<Symbol>(), value.get());
            }
            return;
        }
        add_match_result(name, m_attr_map[name].cast_to<int32_t>() == value.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<int64_t>& value) override {
        if (should_skip(name, true))
            return;
        auto& any = m_attr_map[name].any;
        if (any.is<Symbol>()) {
            if (m_psov) {
                // collect symbol reference and do comparison later
                add_symbol_observed(*m_psov, any.as<Symbol>(), value.get());
            }
            return;
        }
        add_match_result(name, m_attr_map[name].cast_to<int64_t>() == value.get());
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<void>& adapter) override {
        if (should_skip(name))
            return;
        OPENVINO_ASSERT(m_attr_map.count(name) > 0);
        auto& any = m_attr_map[name].any;
        bool is_matched = true;
        if (auto a = ov::as_type<ov::AttributeAdapter<ov::element::Type>>(&adapter)) {
            is_matched = (static_cast<ov::element::Type&>(*a) == any.as<ov::element::Type>());
        } else if (auto a = ov::as_type<ov::AttributeAdapter<PartialShape>>(&adapter)) {
            is_matched = (a->get() == any.as<PartialShape>());
        } else if (auto a = ov::as_type<ov::AttributeAdapter<Dimension>>(&adapter)) {
            is_matched = (a->get() == any.as<Dimension>());
        } else if (auto a = ov::as_type<ov::AttributeAdapter<ov::Shape>>(&adapter)) {
            is_matched = m_attr_map[name].equal_to<int64_t, int>(a->get());
        } else if (auto a = ov::as_type<ov::AttributeAdapter<ov::Strides>>(&adapter)) {
            is_matched = m_attr_map[name].equal_to<int64_t, int>(a->get());
        } else if (auto a = ov::as_type<ov::AttributeAdapter<std::vector<size_t>>>(&adapter)) {
#if defined(__APPLE__) || defined(__EMSCRIPTEN__)
            is_matched = m_attr_map[name].equal_to<size_t, int>(static_cast<std::vector<size_t>&>(*a));
#else
            is_matched = m_attr_map[name].equal_to<size_t, int>(a->get());
#endif
        } else if (auto a = ov::as_type<ov::AttributeAdapter<ov::AxisSet>>(&adapter)) {
            is_matched = m_attr_map[name].equal_to<int64_t, int>(a->get());
        } else if (auto a = ov::as_type<ov::AttributeAdapter<ov::op::TopKSortType>>(&adapter)) {
            is_matched = (a->get() == any.as<std::string>());
        } else if (auto a = ov::as_type<ov::AttributeAdapter<ov::op::TopKMode>>(&adapter)) {
            is_matched = (a->get() == any.as<std::string>());
        } else if (auto a = ov::as_type<ov::AttributeAdapter<ov::CoordinateDiff>>(&adapter)) {
            is_matched = m_attr_map[name].equal_to<int64_t, int>(a->get());
        } else {
            OPENVINO_THROW("AttrMatcher met unsupported AttributeAdapter ", name);
        }
        add_match_result(name, is_matched);
    }
};

// GraphRewrite::apply_matcher_passes() has special logic to make matching faster
// which relies on pattern::op::Pattern subclassing, thus we have to derive
// from it. but we didn't use the predicate facility.
class GenericPattern : public ov::pass::pattern::op::Pattern {
public:
    OPENVINO_RTTI("GenericPattern");

    explicit GenericPattern(const DiscreteTypeInfo& type_info,
                            const OutputVector& args,
                            const detail::AttrMap& attrs,
                            const char* vt)
        : ov::pass::pattern::op::Pattern(args),
          m_type_info(type_info),
          m_attrs(attrs),
          m_vt(vt),
          m_signature() {
        static int global_id = 0;
        if (matcher_verbose_enabled()) {
            // generate signature & friendlyname for verbose debugging log
            auto id = global_id++;
            std::stringstream ss;
            ss << "P" << id << "<" << type_info.get_version() << "::" << type_info.name << ">";
            ss << "(";
            const char* sep = "";
            for (auto& i : args) {
                ss << sep << i.get_node()->get_friendly_name();
                sep = ",";
            }
            ss << ")";
            m_signature = ss.str();
            set_friendly_name(std::string("P") + std::to_string(id));
        }
        // default output for most OP types
        set_output_type(0, element::Type_t::dynamic, PartialShape::dynamic());
    }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& /* new_args */) const override {
        OPENVINO_THROW("Uncopyable");
    }

    bool match_value(ov::pass::pattern::Matcher* matcher,
                     const Output<Node>& pattern_value,
                     const Output<Node>& graph_value) override {
        static std::string level;
        // strictly requires pattern & graph value to come from output port with same index,
        // this is absolute necessary when pattern contains split node connections.
        if (pattern_value.get_index() != graph_value.get_index()) {
            _VERBOSE_LOG(level, "X output index mismatch: ", pattern_value.get_index(), "!=", graph_value.get_index());
            return false;
        }

        auto value_node = graph_value.get_node_shared_ptr();
        if (!value_node->get_type_info().is_castable(m_type_info)) {
            _VERBOSE_LOG(level, "X OP type mismatch: ", m_signature, " vs ", graph_value);
            return false;
        }

        if (!m_vt.predicate(graph_value)) {
            _VERBOSE_LOG(level, "X value info mismatch: ", m_signature, " vs ", graph_value);
            return false;
        }

        if (!m_attrs.empty()) {
            detail::AttrMatcher visitor(m_attrs);
            value_node->visit_attributes(visitor);
            if (!visitor.matched()) {
                _VERBOSE_LOG(level, "X OP attrs mismatch: ", m_signature, " vs ", graph_value);
                return false;
            }
        }

        auto& pattern_map = matcher->get_pattern_value_map();
        pattern_map[shared_from_this()] = graph_value;
        matcher->add_node(graph_value);

        if (get_input_size() == 0)
            return true;

        if (matcher_verbose_enabled())
            level.push_back('\t');
        bool ret = matcher->match_arguments(pattern_value.get_node(), graph_value.get_node_shared_ptr());
        if (matcher_verbose_enabled()) {
            level.pop_back();
            _VERBOSE_LOG(level, ret ? "O" : "X", m_signature, " vs ", graph_value);
        }
        return ret;
    }

private:
    const DiscreteTypeInfo& m_type_info;
    detail::AttrMap m_attrs;
    values_info m_vt;
    std::string m_signature;
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

    PatternNode(values_info vt) {
        node = ov::pass::pattern::any_input([vt](const Output<Node>& value) {
            if (!vt.predicate(value)) {
                _VERBOSE_LOG("*mismatched PatternNode ", value);
                return false;
            }
            _VERBOSE_LOG(" matched PatternNode ", value);
            return true;
        });
    }
    PatternNode(const std::shared_ptr<Node>& node) : node(node) {}
    PatternNode(const std::shared_ptr<ov::op::v0::Parameter>& node) : node(node) {}
    PatternNode(const std::shared_ptr<ov::pass::pattern::op::Or>& pattern)
        : node(std::dynamic_pointer_cast<Node>(pattern)) {}

    // 1D-vector & scalar of symbol
    PatternNode(std::initializer_list<Symbol> v) {
        // initializer_list of Symbol ls special, need to be recorded
        // and eval/check in the callback after whole match is complete,
        // where all observed actual constant values are known, first
        // we will go over all symbols and collect actual value for individual
        // symbol(named symbol), and then we go over all derived symbols and
        // evaluate their predicated values and compare against what observed,
        // and check if they all match.
        // node = ConstVector(std::vector<float>(v), nullptr);
        node = ov::pass::pattern::wrap_type<opset1::Constant>();

        auto& rt_info = node->get_rt_info();
        rt_info["symbolic_const_value"] = std::vector<Symbol>(v);
    }
    PatternNode(const std::vector<Symbol>& v) {
        node = ov::pass::pattern::wrap_type<opset1::Constant>();
        auto& rt_info = node->get_rt_info();
        rt_info["symbolic_const_value"] = v;
    }

    PatternNode(Symbol v) {
        node = ov::pass::pattern::wrap_type<opset1::Constant>();
        auto& rt_info = node->get_rt_info();
        rt_info["symbolic_const_value"] = std::vector<Symbol>({v});
    }

    // scalar constant (treated as wildcard for single-element-constant with any rank)
    PatternNode(int v) : node(std::make_shared<ov::op::v0::Constant>(element::from<int>(), Shape({}), v)) {}
    PatternNode(float v) : node(std::make_shared<ov::op::v0::Constant>(element::from<float>(), Shape({}), v)) {}

    PatternNode(std::initializer_list<int> v, values_info vi = nullptr) {
        node = ConstVector(std::vector<int>(v), vi);
    }
    PatternNode(std::initializer_list<float> v, values_info vi = nullptr) {
        node = ConstVector(std::vector<float>(v), vi);
    }
    PatternNode(std::initializer_list<long> v, values_info vi = nullptr) {
        node = ConstVector(std::vector<int64_t>(v.begin(), v.end()), vi);
    }
    PatternNode(std::initializer_list<long long> v, values_info vi = nullptr) {
        node = ConstVector(std::vector<int64_t>(v.begin(), v.end()), vi);
    }

    // 1d const tensor or scalar
    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value, bool>::type = true>
    static std::shared_ptr<Node> ConstVector(const std::vector<T>& vec, values_info vi = nullptr) {
        if (vi.size() > 0)
            return std::make_shared<ov::op::v0::Constant>(vi[0].first, vi[0].second.to_shape(), vec);
        // initializer_list w/o value_info means to create normal 1D vector
        return std::make_shared<ov::op::v0::Constant>(element::from<T>(), Shape({vec.size()}), vec);
    }
};

/*
template <typename T>
static bool vector_equal_to_any(const std::vector<T>& v0, detail::AttrAny& any) {
    auto v1 = any.cast_to_vector<T>();
    if (v0.size() != v1.size())
        return false;
    return std::equal(v0.begin(), v0.end(), v1.begin());
}

template <typename T>
static bool scalar_equal_to_any(const T& v0, detail::AttrAny& any) {
    if (any.is<int>()) {
        return v0 == any.as<int>();
    } else if (any.is<float>()) {
        return v0 == any.as<float>();
    }
    return v0 == any.as<T>();
}
*/

}  // namespace detail

//==================================================================================================

inline std::shared_ptr<Node> GenInput(values_info vt = nullptr) {
    return ov::pass::pattern::any_input([vt](const Output<Node>& value) {
        if (!vt.predicate(value)) {
            _VERBOSE_LOG("*mismatched GenInput ", value);
            return false;
        }
        _VERBOSE_LOG(" matched GenInput ", value);
        return true;
    });
}

inline std::shared_ptr<Node> makePattern() {
    detail::PatternNode g;
    return g.node;
}

inline std::shared_ptr<Node> makePattern(ov::Rank rank) {
    detail::PatternNode g(rank);
    return g.node;
}

inline std::shared_ptr<Node> makePattern(values_info vt) {
    detail::PatternNode g(vt);
    return g.node;
}

// unknown const
inline std::shared_ptr<Node> makeConst(const ov::element::Type& type,
                                       const ov::PartialShape& pshape,
                                       std::function<bool(ov::op::v0::Constant& node)> pred) {
    return ov::pass::pattern::wrap_type<ov::op::v0::Constant>([type, pshape, pred](const Output<Node>& value) {
        auto cnode = ov::as_type_ptr<opset1::Constant>(value.get_node_shared_ptr());
        if (!cnode)
            return false;

        if (!type.compatible(value.get_element_type()) || !pshape.compatible(value.get_partial_shape())) {
            return false;
        }
        if (pred && !pred(*cnode)) {
            return false;
        }
        return true;
    });
}

template <typename T>
std::shared_ptr<Node> makeConst(const ov::element::Type& type,
                                const ov::Shape& shape,
                                std::initializer_list<T> values) {
    return std::make_shared<ov::op::v0::Constant>(type, shape, std::vector<T>(values));
}

inline std::shared_ptr<Node> makeConst(const std::vector<Symbol>& v) {
    auto node = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto& rt_info = node->get_rt_info();
    rt_info["symbolic_const_value"] = v;
    return node;
}

template <typename T>
std::shared_ptr<Node> makeConst(const ov::element::Type& type, const ov::Shape& shape, const std::vector<T>& values) {
    return std::make_shared<ov::op::v0::Constant>(type, shape, values);
}

template <class T>
std::shared_ptr<Node> makePattern(const std::vector<detail::PatternNode>& inputs,
                                  detail::AttrMap attrmap = {},
                                  const char* vt = nullptr,
                                  const char* friendly_name = nullptr) {
    OutputVector args;
    for (auto& in : inputs)
        args.push_back(in.get_output());

    // pattern nodes are better for pattern matching because
    //  - it can be generic/incomplete, so normal OP node is not working properly
    //  - it has predicate to correctly decide which branch to take (in Or pattern)
    auto pattern_node = std::make_shared<detail::GenericPattern>(T::get_type_info_static(), args, attrmap, vt);

    if (friendly_name)
        pattern_node->set_friendly_name(friendly_name);

    return pattern_node;
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

    // when some attribute is missing or is symbol, the returned
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

template <typename T>
std::shared_ptr<Node> GenConst_tril(values_info vt) {
    return ov::pass::pattern::wrap_type<opset1::Constant>([vt](const Output<Node>& value) {
        auto s1 = as_type_ptr<opset1::Constant>(value.get_node_shared_ptr());
        if (!s1) {
            _VERBOSE_LOG("*mismatched GenConst_tril op type: opset1::Constant vs", value);
            return false;
        }

        if (!vt.predicate(value)) {
            _VERBOSE_LOG("*mismatched GenConst_tril values_info:", value);
            return false;
        }

        // ignore higher dimensions, require lowerst 2D to be lower triangular
        auto shape = s1->get_output_shape(0);
        auto rank = shape.size();
        if (rank < 2) {
            _VERBOSE_LOG("*mismatched GenConst_tril rank < 2 (rank=", rank, ")");
            return false;
        }
        if (shape[rank - 1] != shape[rank - 2]) {
            _VERBOSE_LOG("*mismatched GenConst_tril shape[-1] != shape[-2] : ",
                         shape[rank - 1],
                         " != ",
                         shape[rank - 2]);
            return false;
        }
        // NxN const matrix
        auto N = shape[rank - 1];
        std::vector<T> output_vector = s1->cast_vector<T>();
        // check if it's unit lower triangular matrix
        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < N; j++) {
                if (static_cast<bool>(output_vector[i * N + j]) != static_cast<bool>(j <= i))
                    return false;
            }
        }
        return true;
    });
}

inline std::shared_ptr<Node> operator|(const Output<Node>& lhs, const Output<Node>& rhs) {
    return std::make_shared<ov::pass::pattern::op::Or>(OutputVector{lhs, rhs});
}

inline std::shared_ptr<Node> operator|(const std::shared_ptr<Node>& lhs, const std::shared_ptr<Node>& rhs) {
    return std::make_shared<ov::pass::pattern::op::Or>(
        OutputVector{lhs->get_default_output(), rhs->get_default_output()});
}

inline std::shared_ptr<Node> GenStridedSlice(detail::PatternNode data,
                                             detail::PatternNode start,
                                             detail::PatternNode stop,
                                             detail::PatternNode step,
                                             size_t axis) {
    std::vector<int64_t> begin_mask(axis + 1, 1);
    std::vector<int64_t> end_mask(axis + 1, 1);
    std::vector<int64_t> new_axis_mask;
    std::vector<int64_t> shrink_axis_mask;
    std::vector<int64_t> ellipsis_mask;

    begin_mask[axis] = 0;
    end_mask[axis] = 0;

    auto opt2 = makePattern<opset1::StridedSlice>({data, start, stop, step},
                                                  {{"begin_mask", begin_mask},
                                                   {"end_mask", end_mask},
                                                   {"new_axis_mask", new_axis_mask},
                                                   {"shrink_axis_mask", shrink_axis_mask},
                                                   {"ellipsis_mask", ellipsis_mask}});
    return opt2;
}

inline std::shared_ptr<Node> GenSlice(detail::PatternNode data, Symbol start, Symbol stop, Symbol step, size_t axis) {
    auto opt1 = makePattern<opset8::Slice>({data, {start}, {stop}, {step}, {static_cast<int>(axis)}});

    std::vector<Symbol> vbegin(axis + 1, Symbol(0));
    std::vector<Symbol> vend(axis + 1, Symbol(0));
    std::vector<Symbol> vstride(axis + 1, Symbol(1));

    vbegin[axis] = start;
    vend[axis] = stop;
    vstride[axis] = step;

    detail::PatternNode begin(vbegin);
    detail::PatternNode end(vend);
    detail::PatternNode stride(vstride);

    std::vector<int64_t> begin_mask(axis + 1, 1);
    std::vector<int64_t> end_mask(axis + 1, 1);
    std::vector<int64_t> new_axis_mask;
    std::vector<int64_t> shrink_axis_mask;
    std::vector<int64_t> ellipsis_mask;

    begin_mask[axis] = 0;
    end_mask[axis] = 0;

    auto opt2 = makePattern<opset1::StridedSlice>({data, begin, end, stride},
                                                  {{"begin_mask", begin_mask},
                                                   {"end_mask", end_mask},
                                                   {"new_axis_mask", new_axis_mask},
                                                   {"shrink_axis_mask", shrink_axis_mask},
                                                   {"ellipsis_mask", ellipsis_mask}});
    return opt1 | opt2;
}

//==================================================================================================
class PatternValidator {
public:
    PatternValidator(ov::pass::pattern::Matcher& m) {
        m_is_valid = validate(m);
    }

    double& operator[](const char* symbol_name) {
        return m_symbol_values[symbol_name];
    }

    operator bool() {
        if (!m_is_valid) {
            _VERBOSE_LOG("PatternValidator failed.");
        }
        return m_is_valid;
    }

    bool validate(ov::pass::pattern::Matcher& m) {
        detail::SymbolObservationVector sov;

        auto& pvmap = m.get_pattern_value_map();
        for (auto& pv : pvmap) {
            auto pnode = pv.first;
            auto value_node = pv.second.get_node_shared_ptr();
            auto& rt_info = pnode->get_rt_info();

            if (auto pattern_node = std::dynamic_pointer_cast<ov::pass::pattern::op::Pattern>(pnode)) {
                // pattern_node has no attribute and it has been matched in its predicate
                if (rt_info.count("symbolic_const_value")) {
                    // symbolic constant node, a symbol reference is observed
                    auto& symbols = rt_info["symbolic_const_value"].as<std::vector<Symbol>>();
                    auto constop = std::dynamic_pointer_cast<op::v0::Constant>(value_node);
                    if (!constop) {
                        _VERBOSE_LOG("symbolic_const_value unexpected OP: ", value_node->get_friendly_name());
                        return false;
                    }
                    auto ele_cnt = shape_size(constop->get_shape());
                    auto ele_type = constop->get_element_type();

                    if (ele_cnt != symbols.size()) {
                        _VERBOSE_LOG("symbolic_const_value expect ",
                                     symbols.size(),
                                     " but got ",
                                     ele_cnt,
                                     " from ",
                                     value_node->get_friendly_name());
                        return false;
                    }

                    if (ele_type == ov::element::i32 || ele_type == ov::element::i64 || ele_type == ov::element::f16 ||
                        ele_type == ov::element::f32) {
                        auto observed = constop->cast_vector<double>();
                        for (size_t i = 0; i < symbols.size(); i++)
                            detail::add_symbol_observed(sov, symbols[i], observed[i]);
                    } else {
                        _VERBOSE_LOG("Unexpect element type ", ele_type, " from ", value_node->get_friendly_name());
                        return false;
                    }
                }
                continue;
            }
            if (auto pconst_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(pnode)) {
                // const_node needs to match type/shape/value
                auto vconst_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(value_node);
                if (!vconst_node) {
                    _VERBOSE_LOG("expecting Constant op, but got ", value_node);
                    return false;
                }

                // for constant node matched in pattern, a scalar constant is considered to
                // be compatible with any shape with 1 element, like {}, {1,1}, {1,1,...}
                const auto& expected_shape = pconst_node->get_output_shape(0);
                if (expected_shape.size() == 0) {
                    if (shape_size(vconst_node->get_output_shape(0)) != 1) {
                        _VERBOSE_LOG("expecting a single element const, but got ", vconst_node);
                        return false;
                    }
                } else {
                    if (expected_shape != vconst_node->get_output_shape(0)) {
                        _VERBOSE_LOG("expecting Constant of shape ", expected_shape, " but got ", vconst_node);
                        return false;
                    }
                }

                if (pconst_node->get_output_element_type(0) != vconst_node->get_output_element_type(0)) {
                    // signed integer compare is relaxed, as long as tey are equal when both up-casted to int64_t
                    if (pconst_node->get_output_element_type(0).is_integral() &&
                        pconst_node->get_output_element_type(0).is_signed() &&
                        vconst_node->get_output_element_type(0).is_integral() &&
                        vconst_node->get_output_element_type(0).is_signed()) {
                        auto p_values = pconst_node->cast_vector<int64_t>();
                        auto v_values = vconst_node->cast_vector<int64_t>();
                        if (p_values == v_values) {
                            continue;
                        }
                    }

                    if (pconst_node->get_output_element_type(0).is_real() &&
                        vconst_node->get_output_element_type(0).is_real()) {
                        auto p_values = pconst_node->cast_vector<float>();
                        auto v_values = vconst_node->cast_vector<float>();
                        if (p_values == v_values) {
                            continue;
                        }
                    }

                    _VERBOSE_LOG("expecting Constant of type ",
                                 pconst_node->get_output_element_type(0),
                                 " but got ",
                                 vconst_node);
                    return false;
                }

                auto byte_size =
                    shape_size(vconst_node->get_output_shape(0)) * vconst_node->get_output_element_type(0).size();
                if (std::memcmp(pconst_node->get_data_ptr(), vconst_node->get_data_ptr(), byte_size) != 0) {
                    _VERBOSE_LOG("Constant value mismatch on ", pconst_node, " vs ", vconst_node);
                    return false;
                }
                continue;
            }

            // compare attributes between them
            // assume that there is no Symbol in the attributes, we need to fetch each attributes
            // from
            if (rt_info.count("__attrs__") == 0) {
                _VERBOSE_LOG(" attr compare failed: __attrs__ not found for ", pnode->get_friendly_name());
                return false;
            }

            // attr not specified is treated as not-care and ignored
            // attr with symbol

            detail::AttrMap& attr_map = rt_info["__attrs__"].as<detail::AttrMap>();
            detail::AttrMatcher visitor(attr_map, &sov);
            value_node->visit_attributes(visitor);
            if (!visitor.matched()) {
                _VERBOSE_LOG(" attr compare failed: ",
                             pnode->get_friendly_name(),
                             " vs ",
                             value_node->get_friendly_name());
                return false;
            }
        }

        // check symbol consistency & return independent symbols
        // assign independent symbols & check literals
        std::map<const void*, double> symbol_value_map;
        for (auto& ref : sov) {
            auto& sym = ref.first;
            auto& value = ref.second;

            if (sym.is_independent_var()) {
                auto id = sym.get_id();
                if (symbol_value_map.count(id)) {
                    if (symbol_value_map[id] != value) {
                        _VERBOSE_LOG(" in-consistency between multiple references of same symbol : ",
                                     symbol_value_map[id],
                                     " != ",
                                     value);
                        return false;
                    }
                } else {
                    symbol_value_map[id] = value;
                    m_symbol_values[sym.get_name()] = value;
                    _VERBOSE_LOG("Independent Symbol: ", sym.get_name(), " = ", value);
                }
            }

            if (sym.is_literal_const()) {
                auto literal = sym.eval(symbol_value_map);
                if (literal != value) {
                    _VERBOSE_LOG(" mismatch between literal symbol & value : ", literal, " != ", value);
                    return false;
                }
                // no need to put literal into value map to eval them.
            }
        }

        // derive/eval dependent symbol's value and check against observed
        for (auto& ref : sov) {
            auto& sym = ref.first;
            if (!sym.is_literal_const() && !sym.is_independent_var()) {
                auto derived = sym.eval(symbol_value_map);
                auto value = ref.second;
                bool is_match;

                if (std::trunc(value) == value) {
                    // observed integer
                    is_match = (derived == value);
                } else {
                    auto abs_diff = std::abs(derived - value);
                    auto avg = 0.5f * std::abs(derived + value);
                    if (avg != 0) {
                        is_match = abs_diff < avg * 1e-7;  // relative error less than threshold
                    } else {
                        is_match = (derived == value);
                    }
                }
                if (!is_match) {
                    _VERBOSE_LOG(" mismatch between derived & value : ",
                                 std::setprecision(std::numeric_limits<float>::max_digits10),
                                 derived,
                                 " != ",
                                 std::setprecision(std::numeric_limits<float>::max_digits10),
                                 value);
                    return false;
                }
            }
        }
        _VERBOSE_LOG("PatternValidator validate success!");
        return true;
    }

private:
    std::map<std::string, double> m_symbol_values;
    bool m_is_valid;
};

}  // namespace gen_pattern
}  // namespace ov
