// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pattern_node.hpp"

#include <map>

namespace ov {
namespace intel_cpu {

const int _matcher_verbose = std::getenv("MATCHER_VERBOSE") ? (atoi(std::getenv("MATCHER_VERBOSE"))) : 0;

class AttributePredicate : public ngraph::AttributeVisitor {
    std::map<std::string, attr> attr_map;
    std::map<std::string, bool> attr_match;

public:
    AttributePredicate(const std::vector<attr>& attr) {
        for (auto& a : attr) {
            attr_map[a.name] = a;
            attr_match[a.name] = false;
        }
    }

    bool all_matched(bool verbose = false) {
        bool ret = true;
        for (auto& a : attr_match) {
            if (!a.second) {
                auto& attr = attr_map[a.first];
                _VERBOSE_LOG("     AttributePredicate: failed at ", attr.to_string());
            }
            ret = ret && a.second;
        }
        return ret;
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<void>& adapter) override {
        auto it = attr_map.find(name);
        if (it == attr_map.end())
            return;
        if (auto a = ov::as_type<ov::AttributeAdapter<ov::PartialShape>>(&adapter)) {
            const auto& value = a->get();
            attr_match[name] = it->second.predicate(value.to_string());
        } else if (auto a = ov::as_type<ov::AttributeAdapter<ov::Shape>>(&adapter)) {
            ov::PartialShape value(a->get());
            attr_match[name] = it->second.predicate(value.to_string());
        } else {
            std::cout << "...." << name << ":" << it->second.to_string() << " vs ???" << std::endl;
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
        attr_match[name] = it->second.predicate(adapter.get());
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::string>& adapter) override {
        auto it = attr_map.find(name);
        if (it == attr_map.end())
            return;
        attr_match[name] = it->second.predicate(adapter.get());
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<int>& adapter) override {
        auto it = attr_map.find(name);
        if (it == attr_map.end())
            return;
        attr_match[name] = it->second.predicate(adapter.get());
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<float>& adapter) override {
        auto it = attr_map.find(name);
        if (it == attr_map.end())
            return;
        attr_match[name] = it->second.predicate(adapter.get());
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<int64_t>& adapter) override {
        auto it = attr_map.find(name);
        if (it == attr_map.end())
            return;
        attr_match[name] = it->second.predicate(adapter.get());
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<double>& adapter) override {
        auto it = attr_map.find(name);
        if (it == attr_map.end())
            return;
        attr_match[name] = it->second.predicate(adapter.get());
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<int64_t>>& adapter) override {
        auto it = attr_map.find(name);
        if (it == attr_map.end())
            return;
        attr_match[name] = it->second.predicate(adapter.get());
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

bool attr_compatible(ov::Node& node, const std::vector<attr>& attr) {
    AttributePredicate vis(attr);
    node.visit_attributes(vis);
    return vis.all_matched(true);
}

struct SymbolReference {
    Symbol sym;

    // observations in matched subgraph
    std::shared_ptr<ov::Node> node;
    int offset;
    double value;
    bool is_integer;

    SymbolReference(Symbol sym, std::shared_ptr<ov::Node> node, int offset, float value)
        : sym(sym),
          node(node),
          offset(offset),
          value(value),
          is_integer(false) {}

    SymbolReference(Symbol sym, std::shared_ptr<ov::Node> node, int offset, int32_t value)
        : sym(sym),
          node(node),
          offset(offset),
          value(value),
          is_integer(true) {}
};

static bool collect_symbol_references(std::vector<SymbolReference>& svs,
                                      ov::pass::pattern::PatternValueMap& pvmap,
                                      std::shared_ptr<ov::Node> node) {
    // std::cout << "-------" << node->get_friendly_name() << std::endl;
    //  recursively collect from parent node
    for (size_t i = 0; i < node->get_input_size(); i++) {
        if (!collect_symbol_references(svs, pvmap, node->input_value(i).get_node_shared_ptr())) {
            return false;
        }
    }

    auto& rt_info = node->get_rt_info();
    if (rt_info.count("symbolic_const_value")) {
        auto& symbols = rt_info["symbolic_const_value"].as<std::vector<Symbol>>();
        auto matched_node = pvmap[node].get_node_shared_ptr();
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
                    svs.emplace_back(symbols[i], matched_node, i, observed[i]);
                }
            } else if (ele_type == ov::element::f32) {
                auto observed = constop->get_vector<float>();
                if (observed.size() != symbols.size())
                    return false;
                for (size_t i = 0; i < symbols.size(); i++) {
                    svs.emplace_back(symbols[i], matched_node, i, observed[i]);
                }
            } else {
                return false;
            }
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
                             ref.node,
                             "[",
                             ref.offset,
                             "]");
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
                                 ref.node,
                                 "[",
                                 ref.offset,
                                 "]");
                }
            } else {
                symbol_value_map[id] = ref.value;
                independent_vars.emplace_back(sym);
                symbol_name2value[sym.get_name()] = ref.value;
            }
        }
    }

    if (_matcher_verbose) {
        if (independent_vars.size()) {
            std::cout << "Independent Symbols : ";
            for (auto& sym : independent_vars) {
                std::cout << sym.get_name() << "=" << sym.eval(symbol_value_map) << ", ";
            }
            std::cout << std::endl;
        }
    }

    // derive/eval dependent symbol's value and check against observed
    for (auto& ref : sym_refs) {
        auto& sym = ref.sym;
        if (!sym.is_literal_const() && !sym.is_independent_var()) {
            auto derived = sym.eval(symbol_value_map);
            bool is_match;
            if (ref.is_integer) {
                is_match = (derived == ref.value);
            } else {
                is_match = static_cast<float>(derived) == static_cast<float>(ref.value);
            }
            if (!is_match) {
                _VERBOSE_LOG(" mismatch between derived & value : ",
                             derived,
                             " vs ",
                             ref.value,
                             " from ",
                             ref.node,
                             "[",
                             ref.offset,
                             "]");
                return false;
            }
        }
    }

    return true;
}

std::shared_ptr<Node> GenSlice(GenPatternNode data,
                               Symbol start,
                               Symbol stop,
                               Symbol step,
                               int axis,
                               const char* friendly_name) {
    auto opt1 = GenPattern<opset8::Slice>({data, {start}, {stop}, {step}, {axis}}, nullptr, {}, friendly_name);

    opt1->set_friendly_name(std::string(friendly_name) + "_opt1");

    std::vector<Symbol> vbegin(axis + 1, Symbol(0));
    std::vector<Symbol> vend(axis + 1, Symbol(0));
    std::vector<Symbol> vstride(axis + 1, Symbol(1));

    vbegin[axis] = start;
    vend[axis] = stop;
    vstride[axis] = step;

    GenPatternNode begin(vbegin);
    GenPatternNode end(vend);
    GenPatternNode stride(vstride);

    auto opt2 = ov::pass::pattern::wrap_type<opset1::StridedSlice>(
        {data, begin, end, stride},
        [friendly_name, axis](const Output<Node>& value) {
            auto s1 = as_type_ptr<opset1::StridedSlice>(value.get_node_shared_ptr());
            if (!s1->get_new_axis_mask().empty() || !s1->get_shrink_axis_mask().empty() ||
                !s1->get_ellipsis_mask().empty()) {
                _VERBOSE_LOG(" mismatch GenSlice new/shrink/ellipsis mask: ", friendly_name, " vs ", s1);
                return false;
            }

            auto& begin_mask = s1->get_begin_mask();
            auto& end_mask = s1->get_end_mask();
            auto mask_size = begin_mask.size();
            if (begin_mask.size() != end_mask.size()) {
                _VERBOSE_LOG(" mismatch GenSlice begin/end mask size: ", friendly_name, " vs ", s1);
                return false;
            }

            if (mask_size < axis + 1) {
                _VERBOSE_LOG(" mismatch GenSlice too small mask size: ", friendly_name, " vs ", s1);
                return false;
            }

            for (int i = 0; i < mask_size; i++) {
                auto expect_mask = (i == axis) ? 0 : 1;
                if (begin_mask[i] != expect_mask || end_mask[i] != expect_mask) {
                    _VERBOSE_LOG(" mismatch GenSlice unexpected mask: ", friendly_name, " vs ", s1);
                    return false;
                }
            }
            return true;
        });
    opt2->set_friendly_name(std::string(friendly_name) + "_opt2");
    return opt1 | opt2;
}

}  // namespace intel_cpu
}  // namespace ov