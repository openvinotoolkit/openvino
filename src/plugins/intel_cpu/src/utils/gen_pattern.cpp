// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gen_pattern.hpp"

#include <iomanip>
#include <map>

#include "utils/general_utils.h"

namespace ov {
namespace intel_cpu {

#ifdef CPU_DEBUG_CAPS
const int _matcher_verbose = std::getenv("GENP_VERBOSE") ? (atoi(std::getenv("GENP_VERBOSE"))) : 0;
#endif

class AttributePredicate : public ngraph::AttributeVisitor {
    std::map<std::string, attr*> attr_map;
    std::map<std::string, bool> attr_match;

public:
    AttributePredicate(std::vector<attr>& attr) {
        for (auto& a : attr) {
            attr_map[a.name] = &a;
            attr_match[a.name] = false;
        }
    }

    bool all_matched(bool verbose = false) {
        bool ret = true;
        for (auto& a : attr_match) {
            if (!a.second) {
                _VERBOSE_LOG("     AttributePredicate: failed at ", attr_map[a.first]->to_string());
            }
            ret = ret && a.second;
        }
        return ret;
    }

    template <class Container>
    inline std::string join(const Container& strs) {
        std::stringstream ss;
        ss << "{" << ov::intel_cpu::join(strs, ',') << "}";
        return ss.str();
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<void>& adapter) override {
        auto it = attr_map.find(name);
        if (it == attr_map.end())
            return;
        if (auto a = ov::as_type<ov::AttributeAdapter<ov::PartialShape>>(&adapter)) {
            const auto& value = a->get();
            attr_match[name] = it->second->predicate(value.to_string());
        } else if (auto a = ov::as_type<ov::AttributeAdapter<ov::Shape>>(&adapter)) {
            ov::PartialShape value(a->get());
            attr_match[name] = it->second->predicate(value.to_string());
        } else if (auto a = ov::as_type<ov::AttributeAdapter<std::vector<ov::element::Type>>>(&adapter)) {
            attr_match[name] = it->second->predicate(join(a->get()));
        } else {
            std::cout << "...." << name << ":" << it->second->to_string() << " vs ???" << std::endl;
            attr_match[name] = false;
        }
        /*
        if (auto a = ov::as_type<ov::AttributeAdapter<std::set<std::string>>>(&adapter)) {
            const auto& value = join(a->get());
            append_attribute(name.c_str(), value.c_str());
        } else if (auto a = ov::as_type<ov::AttributeAdapter<std::vector<ov::element::Type>>>(&adapter)) {
            const auto& value = join(a->get());
            append_attribute(name.c_str(), value.c_str());
        } else {
            append_attribute(name.c_str(), "?");
        }
        */
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<bool>& adapter) override {
        auto it = attr_map.find(name);
        if (it == attr_map.end())
            return;
        attr_match[name] = it->second->predicate(adapter.get());
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::string>& adapter) override {
        auto it = attr_map.find(name);
        if (it == attr_map.end())
            return;
        attr_match[name] = it->second->predicate(adapter.get());
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<int>& adapter) override {
        auto it = attr_map.find(name);
        if (it == attr_map.end())
            return;
        attr_match[name] = it->second->predicate(adapter.get());
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<float>& adapter) override {
        auto it = attr_map.find(name);
        if (it == attr_map.end())
            return;
        attr_match[name] = it->second->predicate(adapter.get());
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<int64_t>& adapter) override {
        auto it = attr_map.find(name);
        if (it == attr_map.end())
            return;
        attr_match[name] = it->second->predicate(adapter.get());
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<double>& adapter) override {
        auto it = attr_map.find(name);
        if (it == attr_map.end())
            return;
        attr_match[name] = it->second->predicate(adapter.get());
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<int64_t>>& adapter) override {
        auto it = attr_map.find(name);
        if (it == attr_map.end())
            return;
        attr_match[name] = it->second->predicate(adapter.get());
    }
    /*
        void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<int>>& adapter) override {
            const auto& value = join(adapter.get());
            append_attribute(name.c_str(), value.c_str());
        }

        void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<uint64_t>>& adapter) override {
            const auto& value = join(adapter.get());
            append_attribute(name.c_str(), value.c_str());
        }

        void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<float>>& adapter) override {
            const auto& value = join(adapter.get());
            append_attribute(name.c_str(), value.c_str());
        }

        void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<std::string>>& adapter) override {
            const auto& value = join(adapter.get());
            append_attribute(name.c_str(), value.c_str());
        }

        template<class Container>
        inline std::string join(const Container& strs) {
            std::stringstream ss;
            ss << "[" << ov::intel_cpu::join(strs, ',') << "]";
            return ss.str();
        }
    */
};

bool attr_compatible(ov::Node& node, std::vector<attr>& attr) {
    AttributePredicate vis(attr);
    node.visit_attributes(vis);
    return vis.all_matched(true);
}

// Symbol may be references by pattern in values of constant node & attributes of normal node
// and the references maybe a direct reference or an indirect expression
struct SymbolReference {
    Symbol sym;

    // observations in matched subgraph
    std::shared_ptr<ov::Node> node;
    double value;
    bool is_integer;
    int32_t offset = -1;
    std::string attr_name;

    std::string info() {
        auto node_name = node->get_friendly_name();
        if (offset >= 0)
            return node_name + "[" + std::to_string(offset) + "]";
        return node_name + "." + attr_name;
    }

    SymbolReference(Symbol sym,
                    std::shared_ptr<ov::Node> node,
                    double value,
                    int32_t offset,
                    std::string attr_name = {})
        : sym(sym),
          node(node),
          value(value),
          is_integer(std::floor(value) == value),
          offset(offset),
          attr_name(attr_name) {}

    SymbolReference(Symbol sym,
                    std::shared_ptr<ov::Node> node,
                    int32_t value,
                    int32_t offset,
                    std::string attr_name = {})
        : sym(sym),
          node(node),
          value(value),
          is_integer(true),
          offset(offset),
          attr_name(attr_name) {}
};

static bool collect_symbol_references(std::vector<SymbolReference>& svs,
                                      ov::pass::pattern::PatternValueMap& pvmap,
                                      std::shared_ptr<ov::Node> pattern_node) {
    //  recursively collect from parent node
    // some pattern node like Or, it's not designed to matching any one, but their parents node may be.
    for (size_t i = 0; i < pattern_node->get_input_size(); i++) {
        collect_symbol_references(svs, pvmap, pattern_node->input_value(i).get_node_shared_ptr());
    }

    if (!pvmap.count(pattern_node)) {
        // not matched is not a failure.
        return true;
    }

    auto& rt_info = pattern_node->get_rt_info();
    if (rt_info.count("symbolic_const_value")) {
        auto& symbols = rt_info["symbolic_const_value"].as<std::vector<Symbol>>();
        auto matched_node = pvmap[pattern_node].get_node_shared_ptr();
        auto constop = std::dynamic_pointer_cast<op::v0::Constant>(matched_node);
        if (constop) {
            auto ele_cnt = shape_size(constop->get_shape());
            auto ele_type = constop->get_element_type();

            if (ele_cnt != symbols.size()) {
                return false;
            }

            if (ele_type == ov::element::i32) {
                auto observed = constop->get_vector<int32_t>();
                if (observed.size() != symbols.size())
                    return false;
                for (size_t i = 0; i < symbols.size(); i++) {
                    svs.emplace_back(symbols[i], matched_node, observed[i], i);
                }
            } else if (ele_type == ov::element::f32) {
                auto observed = constop->get_vector<float>();
                if (observed.size() != symbols.size())
                    return false;
                for (size_t i = 0; i < symbols.size(); i++) {
                    svs.emplace_back(symbols[i], matched_node, observed[i], i);
                }
            } else {
                return false;
            }
        }
    }
    if (rt_info.count("pattern_attrs")) {
        auto& vec_attrs = rt_info["pattern_attrs"].as<std::vector<attr>>();
        auto matched_node = pvmap[pattern_node].get_node_shared_ptr();
        for (auto& attr : vec_attrs) {
            if (attr.type != attr::Type::SYM)
                continue;
            if (attr.is_predicate_set)
                svs.emplace_back(attr.sym, matched_node, attr.predicate_with, -1, attr.name);
            else
                return false;
        }
    }
    return true;
}

bool validate_matched_symbols(ov::pass::pattern::Matcher& m, std::map<std::string, double>& symbol_name2value) {
    auto& pvmap = m.get_pattern_value_map();
    auto root_pattern = m.get_pattern();

    // collect symbols and their observed value
    std::vector<SymbolReference> sym_refs;
    if (!collect_symbol_references(sym_refs, pvmap, root_pattern)) {
        _VERBOSE_LOG(" collect_symbol_references failed.");
        return false;
    }

    // assign independent symbols & check literals
    std::vector<Symbol> independent_vars;
    std::map<void*, double> symbol_value_map;
    for (auto& ref : sym_refs) {
        auto& sym = ref.sym;
        if (sym.is_literal_const()) {
            auto literal = sym.eval(symbol_value_map);
            if (literal != ref.value) {
                _VERBOSE_LOG(" mismatch between literal symbol & value : ",
                             literal,
                             " vs ",
                             ref.value,
                             " from ",
                             ref.info());
                return false;
            }
            // no need to put literal into value map to eval them.
        }

        if (sym.is_independent_var()) {
            auto id = sym.get_id();
            if (symbol_value_map.count(id)) {
                if (symbol_value_map[id] != ref.value) {
                    _VERBOSE_LOG(" in-consistency between multiple references of same symbol : ",
                                 symbol_value_map[id],
                                 " vs ",
                                 ref.value,
                                 " from ",
                                 ref.info());
                    return false;
                }
            } else {
                symbol_value_map[id] = ref.value;
                independent_vars.emplace_back(sym);
                symbol_name2value[sym.get_name()] = ref.value;
            }
        }
    }

#ifdef CPU_DEBUG_CAPS
    if (_matcher_verbose) {
        if (independent_vars.size()) {
            std::cout << "Independent Symbols : ";
            for (auto& sym : independent_vars) {
                std::cout << sym.get_name() << "=" << sym.eval(symbol_value_map) << ", ";
            }
            std::cout << std::endl;
        }
    }
#endif

    // derive/eval dependent symbol's value and check against observed
    for (auto& ref : sym_refs) {
        auto& sym = ref.sym;
        if (!sym.is_literal_const() && !sym.is_independent_var()) {
            auto derived = sym.eval(symbol_value_map);
            bool is_match;
            if (ref.is_integer) {
                is_match = (derived == ref.value);
            } else {
                auto abs_diff = std::abs(static_cast<float>(derived) - static_cast<float>(ref.value));
                auto avg = 0.5f * std::abs(static_cast<float>(derived) + static_cast<float>(ref.value));
                if (avg != 0) {
                    is_match = abs_diff < avg * 1e-7;  // relative error less than threshold
                } else {
                    is_match = static_cast<float>(derived) == static_cast<float>(ref.value);
                }
            }
            if (!is_match) {
                _VERBOSE_LOG(" mismatch between derived & value : ",
                             std::setprecision(std::numeric_limits<float>::max_digits10),
                             derived,
                             " vs ",
                             std::setprecision(std::numeric_limits<float>::max_digits10),
                             ref.value,
                             " from ",
                             ref.info());
                return false;
            }
        }
    }

    return true;
}

std::shared_ptr<Node> GenSlice(GenPatternNode data, Symbol start, Symbol stop, Symbol step, size_t axis) {
    auto opt1 = GenPattern<opset8::Slice>({data, {start}, {stop}, {step}, {static_cast<int>(axis)}});

    std::vector<Symbol> vbegin(axis + 1, Symbol(0));
    std::vector<Symbol> vend(axis + 1, Symbol(0));
    std::vector<Symbol> vstride(axis + 1, Symbol(1));

    vbegin[axis] = start;
    vend[axis] = stop;
    vstride[axis] = step;

    GenPatternNode begin(vbegin);
    GenPatternNode end(vend);
    GenPatternNode stride(vstride);

    OutputVector inputs{data, begin, end, stride};
    auto opt2 = std::make_shared<GenericPattern>(inputs);

#ifdef CPU_DEBUG_CAPS
    std::stringstream ss;
    ss << "opset1::StridedSlice";
    opt2->set_friendly_name(ss.str());

    ss << "(";
    const char* sep = "";
    for (auto& i : inputs) {
        ss << sep << i.get_node()->get_friendly_name();
        sep = ",";
    }
    ss << ")";
    auto friendly_name = ss.str();
#else
    auto friendly_name = "";
#endif

    opt2->set_predicate([axis, friendly_name](const Output<Node>& value) {
        (void)friendly_name;
        auto s1 = as_type_ptr<opset1::StridedSlice>(value.get_node_shared_ptr());
        if (!s1) {
            _VERBOSE_LOG(" mismatch StridedSlice OP type: ", friendly_name, "vs", value);
            return false;
        }

        if (!s1->get_new_axis_mask().empty() || !s1->get_shrink_axis_mask().empty() ||
            !s1->get_ellipsis_mask().empty()) {
            _VERBOSE_LOG(" mismatch StridedSlice new/shrink/ellipsis mask: ", friendly_name, "vs", value);
            return false;
        }

        auto& begin_mask = s1->get_begin_mask();
        auto& end_mask = s1->get_end_mask();
        auto mask_size = begin_mask.size();
        if (begin_mask.size() != end_mask.size()) {
            _VERBOSE_LOG(" mismatch StridedSlice begin/end mask size: ", friendly_name, "vs", value);
            return false;
        }

        if (mask_size < axis + 1) {
            _VERBOSE_LOG(" mismatch StridedSlice too small mask size: ", friendly_name, "vs", value);
            return false;
        }

        for (size_t i = 0; i < mask_size; i++) {
            auto expect_mask = (i == axis) ? 0 : 1;
            if (begin_mask[i] != expect_mask || end_mask[i] != expect_mask) {
                _VERBOSE_LOG(" mismatch StridedSlice unexpected mask: ", friendly_name, "vs", value);
                return false;
            }
        }
        _VERBOSE_LOG(" matched StridedSlice ", friendly_name, "==", value);
        return true;
    });
    return opt1 | opt2;
}

}  // namespace intel_cpu
}  // namespace ov