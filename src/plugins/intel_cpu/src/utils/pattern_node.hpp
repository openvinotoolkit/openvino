// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <openvino/opsets/opset8.hpp>
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
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov {
namespace intel_cpu {

extern const int _matcher_verbose;

template <typename... Args>
static inline void _verbose_log(Args&&... args) {
    if (!_matcher_verbose)
        return;
    std::stringstream ss;
    int dummy[] = {(ss << std::forward<Args>(args) << " ", 0)...};
    (void)(dummy);
    ss << std::endl;
    std::cout << ss.str();
}

#define _VERBOSE_LOG(...) _verbose_log(__VA_ARGS__)

//#define _VERBOSE_LOG(...)


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

struct values_info {
    values_info(const char* pattern_list = nullptr) {
        if (pattern_list == nullptr || pattern_list[0] == 0) {
            all_type_pshape.emplace_back(ov::element::dynamic, ov::PartialShape::dynamic(ov::Dimension::dynamic()));
            return;
        }
        auto pattern_vector = split_string(pattern_list, " ");
        for (auto& pattern : pattern_vector) {
            if (pattern[0] == '[') {
                all_type_pshape.emplace_back(ov::element::dynamic, ov::PartialShape(pattern));
            } else {
                auto sep = pattern.find("[");
                assert(sep != std::string::npos);
                all_type_pshape.emplace_back(ov::element::Type(pattern.substr(0, sep)),
                                             ov::PartialShape(pattern.substr(sep)));
            }
        }
    }

    bool predicate(const ov::Output<ov::Node>& value) const {
        auto index = value.get_index();
        auto& item = all_type_pshape[index];
        if (!item.first.compatible(value.get_element_type()) || !item.second.compatible(value.get_partial_shape())) {
            _VERBOSE_LOG("* mismatched vtype between value & pattern : ",
                        value.get_element_type(),
                        value.get_partial_shape(),
                        " vs ",
                        item.first,
                        item.second);
            return false;
        }
        return true;
    }

    size_t get_output_size() {
        return all_type_pshape.size();
    }

    std::vector<std::pair<ov::element::Type, ov::PartialShape>> all_type_pshape;
};

struct attr {
    attr() = default;
    attr(const char* name, const char* v) : name(name) {
        type = 0;
        value.str = v;
    }
    attr(const char* name, int v) : name(name) {
        type = 1;
        value.i32 = v;
    }
    attr(const char* name, float v) : name(name) {
        type = 2;
        value.f32 = v;
    }
    attr(const char* name, double v) : name(name) {
        type = 2;
        value.f32 = v;
    }
    attr(const char* name, std::initializer_list<int64_t> vec_i64) : name(name) {
        type = 3;
        vec_i64 = vec_i64;
    }
    bool predicate(int v) const {
        bool ret = (type == 1 && v == value.i32);
        return ret;
    }
    bool predicate(int64_t v) const {
        bool ret = (type == 1 && v == value.i32);
        return ret;
    }
    bool predicate(float v) const {
        bool ret = (type == 2 && v == value.f32);
        return ret;
    }
    bool predicate(double v) const {
        bool ret = (type == 2 && v == value.f32);
        return ret;
    }
    bool predicate(const std::string& v) const {
        bool ret = (type == 0 && v == value.str);
        return ret;
    }
    bool predicate(const std::vector<int64_t>& vec_i64) const {
        bool ret = (type == 3 && vec_i64 == vec_i64);
        return ret;
    }
    std::string to_string() const {
        std::stringstream ss;
        ss << name << ":";
        if (type == 0)
            ss << value.str;
        if (type == 1)
            ss << value.i32;
        if (type == 2)
            ss << value.f32;
        return ss.str();
    }
    const char* name;
    union {
        const char* str;
        int i32;
        float f32;
    } value;
    std::vector<int64_t> vec_i64;
    int type;
};

bool attr_compatible(ov::Node& node, const std::vector<attr>& attr);

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
        double eval(const std::map<void*, double>& value_map) {
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
    double eval(const std::map<void*, double>& value_map) {
        return entity->eval(value_map);
    }
    bool is_independent_var() {
        return entity->op == 'n';
    }
    int is_literal_const() {
        return entity->op == 'l';
    }
    char get_op() {
        return entity->op;
    }
    void* get_id() {
        return entity.get();
    }
    const char* get_name() {
        return entity->name;
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

// A glue/syntax-sugar type which allows more types to be used as input to GenPattern()
struct GenPatternNode {
    std::shared_ptr<Node> node;

    operator ov::Output<ov::Node> () {
        return node->get_default_output();
    }

    GenPatternNode() {
        node = ov::pass::pattern::any_input(ov::pass::pattern::has_static_rank());
    }
    GenPatternNode(ov::Rank rank) {
        node = ov::pass::pattern::any_input([rank](const Output<Node>& value) {
            if (!rank.compatible(value.get_partial_shape().rank())) {
                _VERBOSE_LOG("*mismatched GenPatternNode rank ", value);
                return false;
            }
            _VERBOSE_LOG(" matched GenPatternNode rank ", value);
            return true;
        });
    }

    GenPatternNode(values_info vt) {
        node = ov::pass::pattern::any_input([vt](const Output<Node>& value) {
            if (!vt.predicate(value)) {
                _VERBOSE_LOG("*mismatched GenPatternNode ", value);
                return false;
            }
            _VERBOSE_LOG(" matched GenPatternNode ", value);
            return true;
        });
    }
    GenPatternNode(const std::shared_ptr<Node>& node) : node(node) {}
    GenPatternNode(const std::shared_ptr<ov::pass::pattern::op::Or>& pattern) : node(std::dynamic_pointer_cast<Node>(pattern)) {}
    GenPatternNode(const Output<Node>& out) : node(out.get_node_shared_ptr()) {}
    GenPatternNode(int v) {
        node = ConstVector(std::vector<int>{v}, "i32[]");
    }
    GenPatternNode(float v) {
        node = ConstVector(std::vector<float>{v}, "f32[]");
    }

    GenPatternNode(std::initializer_list<int> v) {
        node = ConstVector(std::vector<int>(v), nullptr);
    }
    GenPatternNode(std::initializer_list<float> v) {
        node = ConstVector(std::vector<float>(v), nullptr);
    }

    // 1D-vector & scalar of symbol
    GenPatternNode(std::initializer_list<Symbol> v) {
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
    GenPatternNode(const std::vector<Symbol>& v) {
        node = ov::pass::pattern::wrap_type<opset1::Constant>();
        auto& rt_info = node->get_rt_info();
        rt_info["symbolic_const_value"] = v;
    }

    GenPatternNode(Symbol v) {
        node = ov::pass::pattern::wrap_type<opset1::Constant>();
        auto& rt_info = node->get_rt_info();
        rt_info["symbolic_const_value"] = std::vector<Symbol>({v});
    }

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value, bool>::type = true>
    GenPatternNode(std::initializer_list<T> v, values_info vt) {
        node = ConstVector(std::vector<T>(v), vt);
    }

    // 1d const tensor or scalar
    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value, bool>::type = true>
    static std::shared_ptr<Node> ConstVector(const std::vector<T>& vec, values_info vt) {
        auto pred = [vec, vt](const Output<Node>& value) {
            if (!vt.predicate(value)) {
                _VERBOSE_LOG("*mismatched ConstVector ", value);
                return false;
            }
            auto s1 = as_type_ptr<opset1::Constant>(value.get_node_shared_ptr());
            auto shape = s1->get_output_shape(0);
            if (shape_size(shape) != vec.size()) {
                _VERBOSE_LOG("*mismatched shape_size between pattern & value : ", shape_size(shape), " vs ", vec.size());
                _VERBOSE_LOG("*mismatched ConstVector ", value);
                return false;
            }
            std::vector<T> actual = s1->cast_vector<T>();
            if (actual != vec) {
                _VERBOSE_LOG("*mismatched actual value between pattern & value : ",
                            vec2str(vec),
                            " vs ",
                            vec2str(actual));
                _VERBOSE_LOG("*mismatched ConstVector ", value);
                return false;
            }
            _VERBOSE_LOG(" matched ConstVector ", value);
            return true;
        };
        return ov::pass::pattern::wrap_type<opset1::Constant>({}, pred);
    }
};

template <typename T, typename std::enable_if<std::is_arithmetic<T>::value, bool>::type = true>
std::shared_ptr<Node> GenConst(std::initializer_list<T> v, values_info vt = nullptr) {
    GenPatternNode g(v, vt);
    return g.node;
}

inline std::shared_ptr<Node> GenPattern() {
    GenPatternNode g;
    return g.node;
}

inline std::shared_ptr<Node> GenPattern(ov::Rank rank) {
    GenPatternNode g(rank);
    return g.node;
}

inline std::shared_ptr<Node> GenPattern(values_info vt) {
    GenPatternNode g(vt);
    return g.node;
}

template <class... Args>
std::shared_ptr<Node> GenPattern(const std::vector<GenPatternNode>& inputs,
                                 values_info vt = nullptr,
                                 const std::vector<attr>& attrs = {},
                                 const char * friendly_name = "") {
    OutputVector ovs;
    for (auto& i : inputs) {
        ovs.push_back(i.node);
    }

    auto pattern_node = ov::pass::pattern::wrap_type<Args...>(ovs, [vt, attrs, friendly_name](const Output<Node>& value) {
        if (!vt.predicate(value)) {
            _VERBOSE_LOG("*mismatched GenPattern ", friendly_name, "  vt ", value);
            return false;
        }

        // match parent node with attribute a0/a1/...
        if (!attrs.empty() && !attr_compatible(*value.get_node_shared_ptr(), attrs)) {
            _VERBOSE_LOG("*mismatched GenPattern ", friendly_name, " attr ", value);
            return false;
        }
        _VERBOSE_LOG(" matched GenPattern ", friendly_name, " == ", value);
        return true;
    });

    pattern_node->set_friendly_name(friendly_name);

    auto output_size = vt.get_output_size();
    if (output_size > 1)
        pattern_node->set_output_size(output_size);

    return pattern_node;
}

template<typename T>
std::shared_ptr<Node> GenConst_tril(values_info vt) {
    return ov::pass::pattern::wrap_type<opset1::Constant>({}, [vt](const Output<Node>& value) {
        if (!vt.predicate(value)) {
            _VERBOSE_LOG("*mismatched GenConst_tril vt ", value);
            return false;
        }
        // ignore higher dimensions, require lowerst 2D to be lower triangular
        auto s1 = as_type_ptr<opset1::Constant>(value.get_node_shared_ptr());
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
    return std::make_shared<ov::pass::pattern::op::Or>(OutputVector{lhs->get_default_output(), rhs->get_default_output()});
}

std::shared_ptr<Node> GenSlice(GenPatternNode data, Symbol start, Symbol stop, Symbol step, int axis, const char * friendly_name = "");

bool validate_matched_symbols(ov::pass::pattern::Matcher& m, std::map<std::string, double>& symbol_name2value);

}  // namespace intel_cpu
}  // namespace ov