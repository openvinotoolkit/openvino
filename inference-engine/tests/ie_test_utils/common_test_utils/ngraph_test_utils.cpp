// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_test_utils.hpp"

#include <cassert>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <vector>

#include <ngraph/function.hpp>
#include <ngraph/op/util/op_types.hpp>
#include <ngraph/op/util/sub_graph_base.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pass/visualize_tree.hpp>

namespace {
bool isTypeRelaxed(const std::string& type) {
    return type.find_first_of("TypeRelaxed") == 0;
}

bool compareTypeInfo(const ngraph::DiscreteTypeInfo& info1, const ngraph::DiscreteTypeInfo& info2) {
    if (!isTypeRelaxed(info1.name) && !isTypeRelaxed(info2.name) &&
        (info1.version != info2.version)) {
        return false;
    }

    const std::string info1Name =
        isTypeRelaxed(info1.name) && (info1.parent != nullptr) ? info1.parent->name : info1.name;
    const std::string info2Name =
        isTypeRelaxed(info2.name) && (info2.parent != nullptr) ? info2.parent->name : info2.name;
    return info1Name == info2Name;
}

template <typename Node>
bool compare_rt_keys(const Node& node1, const Node& node2) {
    const auto& first_node_rt_info = node1->get_rt_info();
    const auto& second_node_rt_info = node2->get_rt_info();

    if (first_node_rt_info.empty() && second_node_rt_info.empty()) {
        return true;
    }

    if (first_node_rt_info.size() != second_node_rt_info.size()) {
        return false;
    }

    auto first_node_rt_info_it = first_node_rt_info.begin();
    auto second_node_rt_info_it = second_node_rt_info.begin();
    while (first_node_rt_info_it != first_node_rt_info.end()) {
        if (first_node_rt_info_it->first != second_node_rt_info_it->first) {
            return false;
        }
        ++first_node_rt_info_it;
        ++second_node_rt_info_it;
    }

    return true;
}

bool less_by_name(
    const std::shared_ptr<ngraph::op::v0::Result>& l,
    const std::shared_ptr<ngraph::op::v0::Result>& r) {
    return l->get_friendly_name() < r->get_friendly_name();
}

template <typename T>
std::string to_str(const T& v) {
    return std::to_string(v);
}

std::pair<bool, std::string> error(std::string s) {
    return {false, std::move(s)};
}

std::string typeInfoToStr(const ngraph::Node::type_info_t& typeInfo) {
    return std::string(typeInfo.name) + "/" + to_str(typeInfo.version);
}

template <typename Node>
std::string name(const Node& n) {
    return n->get_friendly_name();
}

using AttrName = std::string;

class Result {
public:
    explicit Result(std::string m = {}) : m_message(std::move(m)) {}

    const std::string& message() const {
        return m_message;
    }

    explicit operator bool() const {
        return m_message.empty();
    }

    Result& operator+=(const std::string& msg) {
        m_message.append(1, '\n').append(msg);
        return *this;
    }

private:
    std::string m_message;
};

class MemoryChunk {
public:
    using Data = std::vector<unsigned char>;
    MemoryChunk(Data data) : m_data{std::move(data)} {}

    Data::const_pointer data() const {
        return m_data.data();
    }

    size_t size() const {
        return m_data.size();
    }

private:
    Data m_data;
};

template <typename AttrValue>
class AttributeStorage {
public:
    bool insert_value(AttrName name, AttrValue value) {
        return m_attributes.insert({std::move(name), std::move(value)}).second;
    }

    const AttrValue* get_value(const AttrName& name) const {
        const auto found = m_attributes.find(name);
        if (found != end(m_attributes)) {
            return std::addressof(found->second);
        }
        return {};
    }

    std::size_t get_attributes_number() const {
        return m_attributes.size();
    }

private:
    std::map<AttrName, AttrValue> m_attributes;
};

class Storage : private AttributeStorage<MemoryChunk>,
                private AttributeStorage<bool>,
                private AttributeStorage<std::string>,
                private AttributeStorage<int8_t>,
                private AttributeStorage<int16_t>,
                private AttributeStorage<int32_t>,
                private AttributeStorage<int64_t>,
                private AttributeStorage<uint8_t>,
                private AttributeStorage<uint16_t>,
                private AttributeStorage<uint32_t>,
                private AttributeStorage<uint64_t>,
                private AttributeStorage<float>,
                private AttributeStorage<double>,
                private AttributeStorage<std::vector<int8_t>>,
                private AttributeStorage<std::vector<int16_t>>,
                private AttributeStorage<std::vector<int32_t>>,
                private AttributeStorage<std::vector<int64_t>>,
                private AttributeStorage<std::vector<uint8_t>>,
                private AttributeStorage<std::vector<uint16_t>>,
                private AttributeStorage<std::vector<uint32_t>>,
                private AttributeStorage<std::vector<uint64_t>>,
                private AttributeStorage<std::vector<float>>,
                private AttributeStorage<std::vector<double>>,
                private AttributeStorage<std::vector<std::string>> {
public:
    template <typename AttrValue>
    const AttributeStorage<AttrValue>& storage() const {
        return *static_cast<const AttributeStorage<AttrValue>*>(this);
    }
    template <typename AttrValue>
    AttributeStorage<AttrValue>& storage() {
        return *static_cast<AttributeStorage<AttrValue>*>(this);
    }
};

class ReadAndStoreAttributes : public ngraph::AttributeVisitor, protected Storage {
public:
    void on_adapter(const std::string& name, ngraph::ValueAccessor<void>& adapter) override {
        std::cout << "store   attr [ ERR ]: " << name << " [drop `void` comparison which is '"
                  << adapter.get_type_info().name << "']" << std::endl;
        ///
        /// TODO what is not cover by rest of overloads?
        ///
        /// AttributeAdapter<std::vector<std::shared_ptr<ngraph::op::util::SubGraphOp::InputDescription>>>
        /// AttributeAdapter<std::vector<std::shared_ptr<ngraph::op::util::SubGraphOp::OutputDescription>>
        ///
        ///
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<void*>& adapter) override {
        const auto beg = static_cast<unsigned char*>(adapter.get_ptr());
        const auto end = beg + adapter.size();
        insert(name, MemoryChunk{MemoryChunk::Data(beg, end)});
    }

#define ON_ADAPTER(TYPE)                                                                      \
    void on_adapter(const std::string& name, ngraph::ValueAccessor<TYPE>& adapter) override { \
        insert(name, adapter.get());                                                          \
    }

    ON_ADAPTER(bool)
    ON_ADAPTER(std::string)
    ON_ADAPTER(int8_t)
    ON_ADAPTER(int16_t)
    ON_ADAPTER(int32_t)
    ON_ADAPTER(int64_t)
    ON_ADAPTER(uint8_t)
    ON_ADAPTER(uint16_t)
    ON_ADAPTER(uint32_t)
    ON_ADAPTER(uint64_t)
    ON_ADAPTER(float)
    ON_ADAPTER(double)
    ON_ADAPTER(std::vector<int8_t>)
    ON_ADAPTER(std::vector<int16_t>)
    ON_ADAPTER(std::vector<int32_t>)
    ON_ADAPTER(std::vector<int64_t>)
    ON_ADAPTER(std::vector<uint8_t>)
    ON_ADAPTER(std::vector<uint16_t>)
    ON_ADAPTER(std::vector<uint32_t>)
    ON_ADAPTER(std::vector<uint64_t>)
    ON_ADAPTER(std::vector<float>)
    ON_ADAPTER(std::vector<double>)
    ON_ADAPTER(std::vector<std::string>)

#undef ON_ADAPTER

    void on_adapter(
        const std::string& name,
        ngraph::ValueAccessor<std::shared_ptr<ngraph::Function>>& adapter) override {
        std::cout << "store   attr [ ERR ]: " << name << " [drop `Function` comparison]"
                  << std::endl;
    }

    template <typename AttrValue>
    const AttrValue* get(const AttrName& name) const {
        return storage<AttrValue>().get_value(name);
    }

    template <typename AttrValue>
    bool insert(AttrName name, AttrValue value) {
        return storage<AttrValue>().insert_value(std::move(name), std::move(value));
    }

    size_t attributes_number() const {
        return storage<MemoryChunk>().get_attributes_number() +
               storage<bool>().get_attributes_number() +
               storage<std::string>().get_attributes_number() +
               storage<int8_t>().get_attributes_number() +
               storage<int16_t>().get_attributes_number() +
               storage<int32_t>().get_attributes_number() +
               storage<int64_t>().get_attributes_number() +
               storage<uint8_t>().get_attributes_number() +
               storage<uint16_t>().get_attributes_number() +
               storage<uint32_t>().get_attributes_number() +
               storage<uint64_t>().get_attributes_number() +
               storage<float>().get_attributes_number() +
               storage<double>().get_attributes_number() +
               storage<std::vector<int8_t>>().get_attributes_number() +
               storage<std::vector<int16_t>>().get_attributes_number() +
               storage<std::vector<int32_t>>().get_attributes_number() +
               storage<std::vector<int64_t>>().get_attributes_number() +
               storage<std::vector<uint8_t>>().get_attributes_number() +
               storage<std::vector<uint16_t>>().get_attributes_number() +
               storage<std::vector<uint32_t>>().get_attributes_number() +
               storage<std::vector<uint64_t>>().get_attributes_number() +
               storage<std::vector<float>>().get_attributes_number() +
               storage<std::vector<double>>().get_attributes_number() +
               storage<std::vector<std::string>>().get_attributes_number();
    }
};

template <typename Value>
static constexpr bool in_range(Value v, std::pair<Value, Value> range) {
    return range.first <= v && v < range.second;
}

template <typename Value>
struct Equal {
    static bool equal_value(const Value& lhs, const Value& rhs) {
        return lhs == rhs;
    }
};

template <>
struct Equal<float> {
    static bool equal_value(float lhs, float rhs) {
        return in_range(lhs - rhs, {-0.01, 0.01});
    }
};

template <>
struct Equal<double> {
    static bool equal_value(double lhs, double rhs) {
        return in_range(lhs - rhs, {-0.01, 0.01});
    }
};

template <>
struct Equal<std::vector<double>> {
    static bool equal_value(const std::vector<double>& lhs, const std::vector<double>& rhs) {
        return lhs.size() == rhs.size() &&
               std::equal(begin(lhs), end(lhs), begin(rhs), Equal<double>::equal_value);
    }
};

template <>
struct Equal<std::vector<float>> {
    static bool equal_value(const std::vector<float>& lhs, const std::vector<float>& rhs) {
        return lhs.size() == rhs.size() &&
               std::equal(begin(lhs), end(lhs), begin(rhs), Equal<float>::equal_value);
    }
};

class ReadAndCompareAttributes : public ngraph::AttributeVisitor {
public:
    ReadAndCompareAttributes(const ReadAndStoreAttributes& ref) : m_attr_ref(ref) {}

    enum Tribool { FALSE, INDETERMINATE, TRUE };

    void on_adapter(const std::string& name, ngraph::ValueAccessor<void>& adapter) override {
        std::cout << "compare attr [ ERR ]: " << name << " [drop `void` comparison]" << std::endl;
    }

#define ON_ADAPTER(TYPE)                                                                      \
    void on_adapter(const std::string& name, ngraph::ValueAccessor<TYPE>& adapter) override { \
        verify(name, adapter);                                                                \
    }

    ON_ADAPTER(void*)
    ON_ADAPTER(bool)
    ON_ADAPTER(std::string)
    ON_ADAPTER(int8_t)
    ON_ADAPTER(int16_t)
    ON_ADAPTER(int32_t)
    ON_ADAPTER(int64_t)
    ON_ADAPTER(uint8_t)
    ON_ADAPTER(uint16_t)
    ON_ADAPTER(uint32_t)
    ON_ADAPTER(uint64_t)
    ON_ADAPTER(float)
    ON_ADAPTER(double)
    ON_ADAPTER(std::vector<int8_t>)
    ON_ADAPTER(std::vector<int16_t>)
    ON_ADAPTER(std::vector<int32_t>)
    ON_ADAPTER(std::vector<int64_t>)
    ON_ADAPTER(std::vector<uint8_t>)
    ON_ADAPTER(std::vector<uint16_t>)
    ON_ADAPTER(std::vector<uint32_t>)
    ON_ADAPTER(std::vector<uint64_t>)
    ON_ADAPTER(std::vector<float>)
    ON_ADAPTER(std::vector<double>)
    ON_ADAPTER(std::vector<std::string>)

#undef ON_ADAPTER

    void on_adapter(
        const std::string& name,
        ngraph::ValueAccessor<std::shared_ptr<ngraph::Function>>& adapter) override {
        std::cout << "compare attr [ ERR ]: " << name << " [drop `Function` comparison]"
                  << std::endl;
    }

    bool all_attr_was_compared() const {
        return m_visited_attributes.size() == m_attr_ref.attributes_number();
    }

    const Result& resutl() const {
        return m_result;
    }

private:
    template <typename AttrValue>
    const AttrValue* extract_attr_val_and_mark_as_read(const std::string& name) {
        m_visited_attributes.insert(name);
        const auto ref_value = m_attr_ref.get<AttrValue>(name);
        if (!ref_value) {
            m_result += "missing attribute name: " + name;
        }
        return ref_value;
    }

    template <typename AttrValue>
    void verify(const std::string& name, ngraph::ValueAccessor<AttrValue>& adapter) {
        const AttrValue* ref_value = extract_attr_val_and_mark_as_read<AttrValue>(name);
        if (!ref_value) {
            return;
        }

        if (!Equal<AttrValue>::equal_value(*ref_value, adapter.get())) {
            m_result += "mismatch in value: " + name;
            return;
        }
    }

    void verify(const std::string& name, ngraph::ValueAccessor<void*>& adapter) {
        const MemoryChunk* ref_value = extract_attr_val_and_mark_as_read<MemoryChunk>(name);
        if (!ref_value) {
            return;
        }

        if (adapter.size() != ref_value->size() ||
            std::memcmp(ref_value->data(), adapter.get_ptr(), ref_value->size()) != 0) {
            m_result += "mismatch in value: " + name;
            return;
        }
    }

    const ReadAndStoreAttributes& m_attr_ref;
    Result m_result;
    std::set<AttrName> m_visited_attributes;
};

class CompareNodesAttributes {
public:
    CompareNodesAttributes() : m_compare_attr(m_store_attr) {}

    ReadAndStoreAttributes& get_ref_reder() {
        return m_store_attr;
    }

    ReadAndCompareAttributes& get_cmp_reader() {
        return m_compare_attr;
    }

    bool equal() const {
        return m_compare_attr.all_attr_was_compared() && static_cast<bool>(m_compare_attr.resutl());
    }

    friend std::string to_string(const CompareNodesAttributes& c) {
        const auto& result = c.m_compare_attr.resutl();
        return !!result ? std::string{"Node equal (I guess)"} : result.message();
    }

private:
    ReadAndStoreAttributes m_store_attr;
    ReadAndCompareAttributes m_compare_attr;
};

}  // namespace

std::pair<bool, std::string> compare_functions(
    const std::shared_ptr<ngraph::Function>& f1,
    const std::shared_ptr<ngraph::Function>& f2,
    const bool compareConstValues,
    const bool compareNames,
    const bool compareRuntimeKeys,
    const bool comparePrecisions,
    const bool compareAttribures) {
    /*
     * This function compares two nGraph functions and requires them to have exactly one output
     * + Check nodes types
     * + Check number of inputs
     * + Check shapes
     * + Check parent ports
     * - Do not check nodes attributes (requires visitor mechanism to be completed)
     */

    auto f1_results = f1->get_results();
    auto f2_results = f2->get_results();

    std::sort(f1_results.begin(), f1_results.end(), less_by_name);
    std::sort(f2_results.begin(), f2_results.end(), less_by_name);

    if (f1_results.size() != f2_results.size()) {
        return error(
            "Number of results is different: " + to_str(f1_results.size()) + " and " +
            to_str(f2_results.size()));
    }

    const auto& f1_sinks = f1->get_sinks();
    const auto& f2_sinks = f2->get_sinks();
    if (f1_sinks.size() != f2_sinks.size()) {
        return error(
            "Number of sinks is different: " + to_str(f1_sinks.size()) + " and " +
            to_str(f2_sinks.size()));
    }

    std::ostringstream err_log;

    using ComparedNodes = std::pair<ngraph::Node*, ngraph::Node*>;
    std::queue<ComparedNodes> q;
    std::unordered_set<ngraph::Node*> used;

    for (size_t i = 0; i < f1_results.size(); ++i) {
        if (compareNames) {
            if (name(f1_results[i]->get_input_node_shared_ptr(0)) !=
                name(f2_results[i]->get_input_node_shared_ptr(0))) {
                return error(
                    "Different output names: " + name(f1_results[i]->get_input_node_shared_ptr(0)) +
                    " and " + name(f2_results[i]->get_input_node_shared_ptr(0)));
            }
        }
        q.push({f1_results[i].get(), f2_results[i].get()});
        used.insert(f1_results[i].get());
    }

    while (!q.empty()) {
        auto node1 = q.front().first;
        auto node2 = q.front().second;
        q.pop();

        auto type_info1 = node1->get_type_info();
        auto type_info2 = node2->get_type_info();

        if (!compareTypeInfo(type_info1, type_info2)) {
            return error(typeInfoToStr(type_info1) + " != " + typeInfoToStr(type_info2));
        }

        auto subgraph1 = dynamic_cast<ngraph::op::util::SubGraphOp*>(node1);
        auto subgraph2 = dynamic_cast<ngraph::op::util::SubGraphOp*>(node2);

        if (subgraph1 && subgraph2) {
            auto res = compare_functions(
                subgraph1->get_function(), subgraph2->get_function(), compareConstValues,
                compareNames, compareRuntimeKeys, comparePrecisions);
            if (!res.first) {
                return res;
            }
        }

        const auto& dependencies_1 = node1->get_control_dependencies();
        const auto& dependencies_2 = node2->get_control_dependencies();

        if (dependencies_1.size() != dependencies_2.size()) {
            return error(
                "Number of dependencies is different: " + to_str(dependencies_1.size()) + " for " +
                name(node1) + " and " + to_str(dependencies_2.size()) + " for " + name(node2));
        }

        if (node1->inputs().size() != node2->inputs().size()) {
            return error(
                "Number of inputs is different: " + to_str(node1->inputs().size()) + " for " +
                name(node1) + " and " + to_str(node2->inputs().size()) + " for " + name(node2));
        }

        if (node1->outputs().size() != node2->outputs().size()) {
            return error(
                "Number of outputs is different: " + to_str(node1->inputs().size()) + " for " +
                name(node1) + " and " + to_str(node2->inputs().size()) + " for " + name(node2));
        }

        for (int i = 0; i < node1->inputs().size(); ++i) {
            if (compareConstValues) {
                using Constant = ngraph::opset1::Constant;
                auto const1 = ngraph::as_type_ptr<Constant>(node1->get_input_node_shared_ptr(i));
                auto const2 = ngraph::as_type_ptr<Constant>(node2->get_input_node_shared_ptr(i));

                const auto equal = [](std::shared_ptr<Constant> c1, std::shared_ptr<Constant> c2) {
                    const auto& c1v = c1->cast_vector<double>();
                    const auto& c2v = c2->cast_vector<double>();

                    return c1v.size() == c2v.size() && std::equal(
                                                           begin(c1v), end(c1v), begin(c2v),
                                                           [](const double& s1, const double& s2) {
                                                               return std::abs(s1 - s2) < 0.001;
                                                           });
                };

                if (const1 && const2 && !equal(const1, const2)) {
                    err_log << "Different Constant values detected\n"
                            << node1->description() << " Input(" << i << ") and "
                            << node2->description() << " Input(" << i << ")" << std::endl;
                }
            }

            if (comparePrecisions) {
                if (node1->input(i).get_element_type() != node2->input(i).get_element_type()) {
                    err_log << "Different element type detected\n"
                            << name(node1) << " Input(" << i << ") "
                            << node1->input(i).get_element_type() << " and " << name(node2)
                            << " Input(" << i << ") " << node2->input(i).get_element_type()
                            << std::endl;
                }
            }

            if (!node1->input(i).get_partial_shape().same_scheme(
                    node2->input(i).get_partial_shape())) {
                err_log << "Different shape detected\n"
                        << name(node1) << " Input(" << i << ") "
                        << node1->input(i).get_partial_shape() << " and " << name(node2)
                        << " Input(" << i << ") " << node2->input(i).get_partial_shape()
                        << std::endl;
            }

            if (node1->get_input_source_output(i).get_index() !=
                node2->get_input_source_output(i).get_index()) {
                auto idx1 = node1->get_input_source_output(i).get_index();
                auto idx2 = node2->get_input_source_output(i).get_index();
                err_log << "Different ports detected\n"
                        << name(node1) << " Input(" << i << ") connected to parent port " << idx1
                        << " and " << name(node2) << " Input(" << i << ") connected to parent port "
                        << idx2 << std::endl;
            }

            if (compareRuntimeKeys && !compare_rt_keys(node1, node2)) {
                err_log << "Different runtime info detected\n"
                        << name(node1) << " and " << name(node2) << " not equal runtime info."
                        << std::endl;
            }

            if (!used.count(node1->input_value(i).get_node())) {
                q.push({node1->input_value(i).get_node(), node2->input_value(i).get_node()});
                used.insert(node1->input_value(i).get_node());
            }
        }

        for (int i = 0; i < node1->outputs().size(); ++i) {
            if (!node1->output(i).get_partial_shape().same_scheme(
                    node2->output(i).get_partial_shape())) {
                err_log << "Different shape detected\n"
                        << name(node1) << " Output(" << i << ") "
                        << node1->output(i).get_partial_shape() << " and " << name(node2)
                        << " Output(" << i << ") " << node2->output(i).get_partial_shape()
                        << std::endl;
            }
        }

        if (compareAttribures) {
            CompareNodesAttributes compare_nodes;
            // std::cout << "proceed node1: " << name(node1) << std::endl;
            node1->visit_attributes(compare_nodes.get_ref_reder());
            // std::cout << "proceed node2: " << name(node1) << std::endl;
            node2->visit_attributes(compare_nodes.get_cmp_reader());
            if (!compare_nodes.equal()) {
                return {false, to_string(compare_nodes)};
            }
        }
    }
    return {err_log.str().empty(), err_log.str()};
}
void check_rt_info(const std::shared_ptr<ngraph::Function>& f) {
    static const std::vector<std::string> attrs_to_check{"Variant::RuntimeAttribute::FusedNames"};

    std::ostringstream err_log;
    for (auto& op : f->get_ops()) {
        if (ngraph::op::is_constant(op)) continue;

        const auto& rt_info = op->get_rt_info();
        for (const auto& attr_name : attrs_to_check) {
            if (!rt_info.count(attr_name)) {
                err_log << "Node: " << op->get_friendly_name() << " has no attribute: " << attr_name
                        << std::endl;
            }
        }
    }

    auto err_msg = err_log.str();
    if (!err_msg.empty()) {
        throw ngraph::ngraph_error(err_msg);
    }
}

NGRAPH_RTTI_DEFINITION(TestOpMultiOut, "TestOp", 0);
