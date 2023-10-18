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
namespace intel_cpu {

#ifdef CPU_DEBUG_CAPS

template <typename... Args>
static inline void _verbose_log(Args&&... args) {
    std::stringstream ss;
    int dummy[] = {(ss << std::forward<Args>(args) << " ", 0)...};
    (void)(dummy);
    ss << std::endl;
    std::cout << ss.str();
}

extern const int _matcher_verbose;
#    define _VERBOSE_LOG(...) \
        if (_matcher_verbose) \
        _verbose_log(__VA_ARGS__)
#else
#    define _VERBOSE_LOG(...)
#endif

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

struct values_info {
    values_info(const char* pattern_list = nullptr) {
        if (pattern_list == nullptr || pattern_list[0] == 0) {
            all_type_pshape.clear();
            return;
        }
        auto pattern_vector = split_string(pattern_list, " ");
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

    bool predicate(const ov::Output<ov::Node>& value) const {
        if (all_type_pshape.empty())
            return true;
        auto index = value.get_index();
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

    size_t get_output_size() {
        return all_type_pshape.size();
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
    const char* get_name() const {
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

struct attr {
    const char* name;
    union {
        const char* str;
        int i32;
        float f32;
    } value;
    Symbol sym;
    std::vector<int64_t> vec_i64;
    enum class Type {
        STR = 0,
        I32 = 1,
        F32 = 2,
        VI64 = 3,
        SYM = 4,
        NONE = 999,
    } type;

    double predicate_with;
    bool is_predicate_set = false;

    attr() = default;
    attr(const char* name, const char* v) : name(name) {
        type = Type::STR;
        value.str = v;
    }
    attr(const char* name, int v) : name(name) {
        type = Type::I32;
        value.i32 = v;
    }
    attr(const char* name, float v) : name(name) {
        type = Type::F32;
        value.f32 = v;
    }
    attr(const char* name, double v) : name(name) {
        type = Type::F32;
        value.f32 = v;
    }
    attr(const char* name, std::initializer_list<int64_t> v) : name(name) {
        type = Type::VI64;
        vec_i64 = v;
    }
    attr(const char* name, Symbol v) : name(name) {
        type = Type::SYM;
        sym = v;
    }

    bool predicate(int v) {
        if (lazy_predicate(v))
            return true;
        return (type == Type::I32 && v == value.i32);
    }
    bool predicate(int64_t v) {
        if (lazy_predicate(v))
            return true;
        return (type == Type::I32 && v == value.i32);
    }
    bool predicate(float v) {
        if (lazy_predicate(v))
            return true;
        return (type == Type::F32 && v == value.f32);
    }
    bool predicate(double v) {
        if (lazy_predicate(v))
            return true;
        return (type == Type::F32 && v == value.f32);
    }
    bool predicate(const std::string& v) {
        return (type == Type::STR && v == value.str);
    }
    bool predicate(const std::vector<int64_t>& v) {
        return (type == Type::VI64 && v == vec_i64);
    }
    std::string to_string() const {
        std::stringstream ss;
        ss << name << ":";
        if (type == Type::STR)
            ss << value.str;
        if (type == Type::I32)
            ss << value.i32;
        if (type == Type::F32)
            ss << value.f32;
        if (type == Type::VI64)
            ss << vec2str(vec_i64);
        if (type == Type::SYM)
            ss << sym.get_name();
        return ss.str();
    }

private:
    bool lazy_predicate(double v) {
        if (type != Type::SYM)
            return false;
        predicate_with = v;
        is_predicate_set = true;
        return true;
    }
};

bool attr_compatible(ov::Node& node, std::vector<attr>& attr);

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

class GenericPattern : public ov::pass::pattern::op::Pattern {
public:
    OPENVINO_RTTI("GenericPattern");

    explicit GenericPattern(const OutputVector& patterns = {}) : ov::pass::pattern::op::Pattern(patterns) {
        set_output_type(0, element::Type_t::dynamic, PartialShape::dynamic());
    }

    // this allows code inside pred to access pattern node itself
    void set_predicate(ov::pass::pattern::op::ValuePredicate pred) {
        m_predicate = pred;
    }

    bool match_value(ov::pass::pattern::Matcher* matcher,
                     const Output<Node>& pattern_value,
                     const Output<Node>& graph_value) override {
        if (m_predicate(graph_value)) {
            auto& pattern_map = matcher->get_pattern_value_map();
            pattern_map[shared_from_this()] = graph_value;
            matcher->add_node(graph_value);
            return (get_input_size() == 0
                        ? true
                        : matcher->match_arguments(pattern_value.get_node(), graph_value.get_node_shared_ptr()));
        }
        return false;
    }
};

// A glue/syntax-sugar type which allows more types to be used as input to GenPattern()
struct GenPatternNode {
    std::shared_ptr<Node> node;

    operator ov::Output<ov::Node>() {
        return node->get_default_output();
    }

    GenPatternNode() {
        node = ov::pass::pattern::any_input(ov::pass::pattern::has_static_rank());
    }
    GenPatternNode(ov::Rank rank) {
        node = ov::pass::pattern::any_input([rank](const Output<Node>& value) {
            if (!rank.compatible(value.get_partial_shape().rank())) {
                _VERBOSE_LOG("*mismatched GenPatternNode rank ", value, " expecting ", rank);
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
    GenPatternNode(const std::shared_ptr<ov::pass::pattern::op::Or>& pattern)
        : node(std::dynamic_pointer_cast<Node>(pattern)) {}
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
        auto friendly_name = vt.to_string() + vec2str(vec, 9);
        auto pnode = std::make_shared<GenericPattern>();
        pnode->set_friendly_name(friendly_name);
        pnode->set_predicate([vec, vt, friendly_name](const Output<Node>& value) {
            if (!value.get_node_shared_ptr()->get_type_info().is_castable(opset1::Constant::get_type_info_static())) {
                _VERBOSE_LOG("*mismatched ConstVector type:", friendly_name, "vs", value);
                return false;
            }
            if (!vt.predicate(value)) {
                _VERBOSE_LOG("*mismatched ConstVector value info:", friendly_name, "vs", value);
                return false;
            }
            auto s1 = as_type_ptr<opset1::Constant>(value.get_node_shared_ptr());
            auto shape = s1->get_output_shape(0);
            if (shape_size(shape) != vec.size()) {
                _VERBOSE_LOG("*mismatched shape_size between pattern & value:", shape_size(shape), "vs", vec.size());
                _VERBOSE_LOG("*mismatched ConstVector", friendly_name, "vs", value);
                return false;
            }
            std::vector<T> actual = s1->cast_vector<T>();
            if (actual != vec) {
                _VERBOSE_LOG("*mismatched actual value between pattern & value :", vec2str(vec), "vs", vec2str(actual));
                _VERBOSE_LOG("*mismatched ConstVector", value);
                return false;
            }
            _VERBOSE_LOG(" matched ConstVector", friendly_name, " == ", value);
            return true;
        });

        return pnode;
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

template <class T, bool check_vt = false>
std::shared_ptr<Node> GenPattern(const std::vector<GenPatternNode>& inputs,
                                 values_info vt = nullptr,
                                 std::vector<attr> attrs = {}) {
    auto* p_type_info = &(T::get_type_info_static());

    OutputVector output_vectors;
    for (auto& i : inputs) {
        output_vectors.push_back(i.node);
    }
    auto pattern_node = std::make_shared<GenericPattern>(output_vectors);

#ifdef CPU_DEBUG_CAPS
    std::stringstream ss;
    ss << p_type_info->get_version() << "::" << p_type_info->name << " " << vt.to_string();

    pattern_node->set_friendly_name(ss.str());

    ss << "(";
    const char* sep = "";
    for (auto& i : inputs) {
        ss << sep << i.node->get_friendly_name();
        sep = ",";
    }
    ss << ")";
    auto friendly_name = ss.str();
#else
    auto friendly_name = "";
#endif

    // attributes may also contain symbol, so record it into rt_info
    auto& rt_info = pattern_node->get_rt_info();
    rt_info["pattern_attrs"] = std::vector<attr>(attrs);

    pattern_node->set_predicate([p_type_info, vt, pattern_node, friendly_name](const Output<Node>& value) {
        if (!value.get_node_shared_ptr()->get_type_info().is_castable(*p_type_info)) {
            _VERBOSE_LOG("*mismatched GenPattern OP type: ", friendly_name, "vs", value);
            return false;
        }

        if (check_vt && !vt.predicate(value)) {
            _VERBOSE_LOG("*mismatched GenPattern value info: ", friendly_name, "vs", value);
            return false;
        }

        // match parent node with attribute a0/a1/...
        auto& rt_info = pattern_node->get_rt_info();
        if (rt_info.count("pattern_attrs")) {
            auto& attrs = rt_info["pattern_attrs"].as<std::vector<attr>>();
            if (!attrs.empty() && !attr_compatible(*value.get_node_shared_ptr(), attrs)) {
                _VERBOSE_LOG("*mismatched GenPattern attr: ", friendly_name, "vs", value);
                return false;
            }
        }
        _VERBOSE_LOG(" matched GenPattern ", friendly_name, " == ", value);
        return true;
    });

    auto output_size = vt.get_output_size();
    if (output_size > 1)
        pattern_node->set_output_size(output_size);

    return pattern_node;
}

template <typename T>
std::shared_ptr<Node> GenConst_tril(values_info vt) {
    auto pnode = std::make_shared<GenericPattern>();
    pnode->set_predicate([vt](const Output<Node>& value) {
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
    return pnode;
}

inline std::shared_ptr<Node> operator|(const Output<Node>& lhs, const Output<Node>& rhs) {
    return std::make_shared<ov::pass::pattern::op::Or>(OutputVector{lhs, rhs});
}

inline std::shared_ptr<Node> operator|(const std::shared_ptr<Node>& lhs, const std::shared_ptr<Node>& rhs) {
    return std::make_shared<ov::pass::pattern::op::Or>(
        OutputVector{lhs->get_default_output(), rhs->get_default_output()});
}

std::shared_ptr<Node> GenSlice(GenPatternNode data, Symbol start, Symbol stop, Symbol step, size_t axis);

bool validate_matched_symbols(ov::pass::pattern::Matcher& m, std::map<std::string, double>& symbol_name2value);

}  // namespace intel_cpu
}  // namespace ov