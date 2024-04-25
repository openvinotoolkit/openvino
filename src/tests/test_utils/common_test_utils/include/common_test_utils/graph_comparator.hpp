// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <limits>
#include <memory>
#include <queue>

#include "openvino/core/dimension.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/op/util/sub_graph_base.hpp"
#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/runtime/string_aligned_buffer.hpp"

class FunctionsComparator {
public:
    enum CmpValues {
        NONE = 0,
        NODES = 1 << 0,
        CONST_VALUES = 1 << 1,
        NAMES = 1 << 2,
        RUNTIME_KEYS = 1 << 3,
        PRECISIONS = 1 << 4,
        ATTRIBUTES = 1 << 5,
        TENSOR_NAMES = 1 << 6,
        ACCURACY = 1 << 7,
        SUBGRAPH_DESCRIPTORS = 1 << 8,
        CONSUMERS_COUNT = 1 << 9
    };

    struct Result {
        bool valid;
        std::string message;

        static Result ok(std::string msg = {}) {
            return {true, std::move(msg)};
        }
        static Result error(std::string msg) {
            return {false, std::move(msg)};
        }
    };

    static FunctionsComparator no_default() noexcept {
        return FunctionsComparator{NONE};
    }
    static FunctionsComparator with_default() noexcept {
        auto fc = FunctionsComparator::no_default();
        fc.enable(NODES);
        fc.enable(PRECISIONS);
        fc.enable(TENSOR_NAMES);
        fc.enable(SUBGRAPH_DESCRIPTORS);
        return fc;
    }

    FunctionsComparator& enable(CmpValues f) noexcept {
        m_comparison_flags = static_cast<CmpValues>(m_comparison_flags | f);
        return *this;
    }

    FunctionsComparator& disable(CmpValues f) noexcept {
        m_comparison_flags = static_cast<CmpValues>(m_comparison_flags & ~f);
        return *this;
    }

    bool should_compare(CmpValues f) const noexcept {
        return m_comparison_flags & f;
    }
    Result compare(const std::shared_ptr<ov::Model>& f, const std::shared_ptr<ov::Model>& f_ref) const;

    Result operator()(const std::shared_ptr<ov::Model>& f, const std::shared_ptr<ov::Model>& f_ref) const {
        return compare(f, f_ref);
    }

    void set_accuracy_thresholds(float abs_threshold, float rel_threshold) {
        m_abs_threshold = abs_threshold;
        m_rel_threshold = rel_threshold;
    }

private:
    explicit FunctionsComparator(CmpValues f) noexcept : m_comparison_flags(f) {}
    CmpValues m_comparison_flags;
    float m_abs_threshold = 1e-7f;
    float m_rel_threshold = 1e-7f;
};

///
/// \deprecated
/// \brief compare_functions is obsolete function use FunctionsComparator instead.
///
inline std::pair<bool, std::string> compare_functions(const std::shared_ptr<ov::Model>& f,
                                                      const std::shared_ptr<ov::Model>& f_ref,
                                                      const bool compareConstValues = false,
                                                      const bool compareNames = false,
                                                      const bool compareRuntimeKeys = false,
                                                      const bool comparePrecisions = true,
                                                      const bool compareAttributes = false,
                                                      const bool compareTensorNames = true) {
    auto fc = FunctionsComparator::no_default();

    using Cmp = FunctionsComparator::CmpValues;
    fc.enable(Cmp::NODES);
    if (compareConstValues)
        fc.enable(Cmp::CONST_VALUES);
    if (compareNames)
        fc.enable(Cmp::NAMES);
    if (compareRuntimeKeys)
        fc.enable(Cmp::RUNTIME_KEYS);
    if (comparePrecisions)
        fc.enable(Cmp::PRECISIONS);
    if (compareAttributes)
        fc.enable(Cmp::ATTRIBUTES);
    if (compareTensorNames)
        fc.enable(Cmp::TENSOR_NAMES);

    const auto r = fc(f, f_ref);
    return {r.valid, r.message};
}

void check_rt_info(const std::shared_ptr<ov::Model>& f);

namespace ov {
namespace pass {
class InjectionPass : public ov::pass::ModelPass {
public:
    using injection_callback = std::function<void(std::shared_ptr<ov::Model>)>;

    explicit InjectionPass(injection_callback callback) : ModelPass(), m_callback(std::move(callback)) {}

    bool run_on_model(const std::shared_ptr<ov::Model>& f) override {
        m_callback(f);
        return false;
    }

private:
    injection_callback m_callback;
};

class UniqueNamesHolder {
    using names_t = std::unordered_set<std::string>;
    std::unordered_map<Node*, names_t> m_result_tensor_names;
    std::unordered_map<Node*, std::pair<std::string, bool>> m_result_node_names;

    size_t m_index{0};
    bool m_soft_names_comparison{false};
    bool m_result_friendly_names_check{true};

    std::string generate_tensor_name() {
        return "tensor_" + std::to_string(m_index++);
    }

    std::string generate_friendly_name() {
        return "node_" + std::to_string(m_index++);
    }

public:
    using Ptr = std::shared_ptr<UniqueNamesHolder>;

    UniqueNamesHolder() = default;

    void init_names(std::shared_ptr<ov::Model> f) {
        // initialize function with unique friendly and tensor names
        for (auto node : f->get_ordered_ops()) {
            const auto& node_name = generate_friendly_name();
            // this expression means that user didn't set friendly name and it was generated automatically
            if (node->get_friendly_name() == node->get_name()) {
                node->set_friendly_name(node_name);
            }

            for (auto output : node->outputs()) {
                const auto& tensor_name = generate_tensor_name();
                if (output.get_names().empty()) {
                    output.set_names({tensor_name});
                }
            }
        }

        // save result input tensor names and friendly name for future comparison
        for (auto r : f->get_results()) {
            const auto& tensor_names = r->input_value(0).get_names();
            m_result_tensor_names[r.get()].insert(tensor_names.begin(), tensor_names.end());
            m_result_node_names[r.get()] = {r->input_value(0).get_node()->get_friendly_name(),
                                            r->input_value(0).get_node()->outputs().size() != 1};
            // As get_ordered_ops doesn't guaranty that the order of Result ops is the same
            // we explicitly update Result names to have them in increasing order that
            // helps FunctionComparator to compare Functions with multiple Results.
            r->set_friendly_name(generate_friendly_name());
        }
    }

    void check_unique_names(std::shared_ptr<ov::Model> f) {
        // Check that all tensor names and friendly names are unique
        names_t unique_tensor_names, unique_friendly_names;
        for (auto node : f->get_ordered_ops()) {
            if (unique_friendly_names.count(node->get_friendly_name())) {
                std::stringstream ss;
                ss << "Node: " << node->get_type_info() << " with name " << node->get_friendly_name() << " ";
                ss << "has non unique friendly name.";
                OPENVINO_THROW(ss.str());
            }
            unique_friendly_names.insert(node->get_friendly_name());

            if (as_type_ptr<ov::op::v0::Result>(node))
                continue;
            for (auto output : node->outputs()) {
                const auto& tensor_names = output.get_names();
                if (std::any_of(tensor_names.begin(), tensor_names.end(), [&](const std::string& name) {
                        return unique_tensor_names.count(name);
                    })) {
                    std::stringstream ss;
                    ss << "Node: " << node->get_type_info() << " with name " << node->get_friendly_name() << " ";
                    ss << "has non unique tensor name.";
                    OPENVINO_THROW(ss.str());
                }
                unique_tensor_names.insert(tensor_names.begin(), tensor_names.end());
            }
        }

        for (auto r : f->get_results()) {
            // Check that old tensor names for results were preserved
            const auto& ref_tensor_names = m_result_tensor_names.at(r.get());
            const auto& cur_tensor_names = r->input_value(0).get_names();
            for (const auto& ref_name : ref_tensor_names) {
                if (cur_tensor_names.count(ref_name) == 0) {
                    std::stringstream ss;
                    auto node = r->input_value(0).get_node();
                    ss << "Tensor name: " << ref_name << " is missing in " << node->get_type_info() << " ";
                    ss << "output(" << r->input_value(0).get_index() << ")";
                    OPENVINO_THROW(ss.str());
                }
            }

            if (m_result_friendly_names_check) {
                // Check that result input node names are preserved
                bool is_multi_output = m_result_node_names.at(r.get()).second;
                const auto& ref_node_name = m_result_node_names.at(r.get()).first;
                const auto& cur_node_name = r->input_value(0).get_node()->get_friendly_name();
                if (is_multi_output || m_soft_names_comparison) {
                    if (cur_node_name.find(ref_node_name) == std::string::npos) {
                        std::stringstream ss;
                        ss << "Output node names mismatch: " << cur_node_name << " and " << ref_node_name
                           << " (reference)";
                        OPENVINO_THROW(ss.str());
                    }
                } else if (cur_node_name != ref_node_name) {
                    std::stringstream ss;
                    ss << "Output node names are different: " << cur_node_name << " and " << ref_node_name
                       << " (reference)";
                    OPENVINO_THROW(ss.str());
                }
            }
        }
    }

    void enable_soft_names_comparison() {
        m_soft_names_comparison = true;
    }
    void disable_result_friendly_names_check() {
        m_result_friendly_names_check = false;
    }
};

class InitUniqueNames : public ov::pass::ModelPass {
    UniqueNamesHolder::Ptr m_unh;

public:
    InitUniqueNames(UniqueNamesHolder::Ptr unh) : m_unh(unh) {}
    bool run_on_model(const std::shared_ptr<ov::Model>& f) override {
        m_unh->init_names(f);
        return false;
    }
};

class CheckUniqueNames : public ov::pass::ModelPass {
    UniqueNamesHolder::Ptr m_unh;

public:
    CheckUniqueNames(UniqueNamesHolder::Ptr unh,
                     bool soft_names_comparison = false,
                     bool result_friendly_names_check = true)
        : m_unh(unh) {
        if (soft_names_comparison)
            m_unh->enable_soft_names_comparison();
        if (!result_friendly_names_check)
            m_unh->disable_result_friendly_names_check();
    }
    bool run_on_model(const std::shared_ptr<ov::Model>& f) override {
        m_unh->check_unique_names(f);
        return false;
    }
};

}  // namespace pass
}  // namespace ov

class Comparator {
public:
    using CmpValues = FunctionsComparator::CmpValues;
    using Result = FunctionsComparator::Result;
    using ComparedNodes = std::pair<ov::Node*, ov::Node*>;

    explicit Comparator(CmpValues f, float abs_threshold = 1e-7f, float rel_threshold = 1e-7f)
        : m_comparison_flags(f),
          m_abs_threshold(abs_threshold),
          m_rel_threshold(rel_threshold) {}

    Result compare(const std::shared_ptr<ov::Model>& f, const std::shared_ptr<ov::Model>& f_ref);

    Result compare(ov::Node* node1, ov::Node* node2) {
        std::stringstream errors;
        const auto result = compare(node1, node2, errors);
        if (!result.valid) {
            return result;
        }
        const auto msg = errors.str();
        return msg.empty() ? Result::ok() : Result::error(msg);
    }

    CmpValues get_comparison_flags() const {
        return m_comparison_flags;
    }

    void compare_inputs(ov::Node* node1, ov::Node* node2, std::ostream& err_log);

    void compare_outputs(ov::Node* node1, ov::Node* node2, std::ostream& err_log);

    void compare_nodes(ov::Node* node1, ov::Node* node2, std::ostream& err_log);

private:
    bool should_compare(CmpValues f) const noexcept {
        return m_comparison_flags & f;
    }

    ///
    /// \param err_log - will be fill by minor errors if happen
    /// \return only fatality error if some minor one appears it will be add to err_log
    ///
    Result compare(ov::Node* node1, ov::Node* node2, std::ostream& err_log);

    void add_nodes_inputs_to_queue(ov::Node* node1, ov::Node* node2);

    //-- DATA --
    CmpValues m_comparison_flags;

    std::queue<ComparedNodes> q;
    std::unordered_set<ov::Node*> used;

    float m_abs_threshold = 1e-7f;
    float m_rel_threshold = 1e-7f;
};

inline namespace tools {
template <typename T>
std::string to_str(const T& v) {
    using std::to_string;
    return to_string(v);
}
template <typename Node>
std::string name(const Node& n) {
    return n->get_friendly_name();
}
}  // namespace tools
namespace attributes {

namespace detail {

using AttrName = std::string;

class Result {
public:
    explicit Result(std::string m = {}) : m_message(std::move(m)) {}

    const std::string& message() const {
        return m_message;
    }

    bool has_error() const {
        return !m_message.empty();
    }

    Result& operator+=(const std::string& msg) {
        m_message.append(m_break_line_no, '\n').append(msg);
        m_break_line_no = 1;
        return *this;
    }

private:
    std::string m_message;
    int m_break_line_no{0};
};

using SubGraphOpInputDescription = std::vector<std::shared_ptr<ov::op::util::SubGraphOp::InputDescription>>;

using SubGraphOpOutputDescription = std::vector<std::shared_ptr<ov::op::util::SubGraphOp::OutputDescription>>;

using SpecialBodyPorts = ov::op::v5::Loop::SpecialBodyPorts;

namespace storage {

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
                private AttributeStorage<std::vector<std::string>>,
                private AttributeStorage<std::shared_ptr<ov::Model>>,
                private AttributeStorage<SubGraphOpInputDescription>,
                private AttributeStorage<SubGraphOpOutputDescription>,
                private AttributeStorage<ov::op::util::FrameworkNodeAttrs>,
                private AttributeStorage<std::shared_ptr<ov::op::util::Variable>>,
                private AttributeStorage<ov::PartialShape>,
                private AttributeStorage<ov::Dimension>,
                private AttributeStorage<std::shared_ptr<ov::StringAlignedBuffer>> {
public:
    template <typename AttrValue>
    const AttributeStorage<AttrValue>& storage() const {
        return *static_cast<const AttributeStorage<AttrValue>*>(this);
    }
    template <typename AttrValue>
    AttributeStorage<AttrValue>& storage() {
        return *static_cast<AttributeStorage<AttrValue>*>(this);
    }

    size_t stored_attributes_number() const {
        return storage<MemoryChunk>().get_attributes_number() + storage<bool>().get_attributes_number() +
               storage<std::string>().get_attributes_number() + storage<int8_t>().get_attributes_number() +
               storage<int16_t>().get_attributes_number() + storage<int32_t>().get_attributes_number() +
               storage<int64_t>().get_attributes_number() + storage<uint8_t>().get_attributes_number() +
               storage<uint16_t>().get_attributes_number() + storage<uint32_t>().get_attributes_number() +
               storage<uint64_t>().get_attributes_number() + storage<float>().get_attributes_number() +
               storage<double>().get_attributes_number() + storage<std::vector<int8_t>>().get_attributes_number() +
               storage<std::vector<int16_t>>().get_attributes_number() +
               storage<std::vector<int32_t>>().get_attributes_number() +
               storage<std::vector<int64_t>>().get_attributes_number() +
               storage<std::vector<uint8_t>>().get_attributes_number() +
               storage<std::vector<uint16_t>>().get_attributes_number() +
               storage<std::vector<uint32_t>>().get_attributes_number() +
               storage<std::vector<uint64_t>>().get_attributes_number() +
               storage<std::vector<float>>().get_attributes_number() +
               storage<std::vector<double>>().get_attributes_number() +
               storage<std::vector<std::string>>().get_attributes_number() +
               storage<std::shared_ptr<ov::Model>>().get_attributes_number() +
               storage<SubGraphOpInputDescription>().get_attributes_number() +
               storage<SubGraphOpOutputDescription>().get_attributes_number() +
               storage<ov::op::util::FrameworkNodeAttrs>().get_attributes_number() +
               storage<std::shared_ptr<ov::op::util::Variable>>().get_attributes_number() +
               storage<ov::PartialShape>().get_attributes_number() + storage<ov::Dimension>().get_attributes_number() +
               storage<std::shared_ptr<ov::StringAlignedBuffer>>().get_attributes_number();
    }
};

}  // namespace storage

class ReadAndStoreAttributes : public ov::AttributeVisitor, protected storage::Storage {
public:
    void on_adapter(const std::string& name, ov::ValueAccessor<void>& adapter) override;

#define ON_ADAPTER(TYPE) \
    void on_adapter(const std::string& name, ov::ValueAccessor<TYPE>& adapter) override { insert(name, adapter.get()); }

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
    ON_ADAPTER(std::shared_ptr<ov::Model>)

#undef ON_ADAPTER

    template <typename AttrValue>
    const AttrValue* get(const AttrName& name) const {
        return storage<AttrValue>().get_value(name);
    }

    template <typename AttrValue>
    bool insert(AttrName name, AttrValue value) {
        return storage<AttrValue>().insert_value(std::move(name), std::move(value));
    }

    size_t attributes_number() const {
        return stored_attributes_number();
    }

    const Result read_result() const {
        return m_read_result;
    }

private:
    Result m_read_result;
};

namespace equal {

template <typename Value>
struct Equal {
    static bool equal_value(const Value& lhs, const Value& rhs) {
        return lhs == rhs;
    }
};

template <>
struct Equal<ov::bfloat16> {
    static bool equal_value(ov::bfloat16 lhs, ov::bfloat16 rhs) {
        if (lhs.to_bits() == rhs.to_bits()) {
            return true;
        }
        return std::abs(lhs - rhs) < 1e-3;
    }
};

template <>
struct Equal<ov::float16> {
    static bool equal_value(ov::float16 lhs, ov::float16 rhs) {
        if (lhs.to_bits() == rhs.to_bits()) {
            return true;
        }
        return std::abs(lhs - rhs) < 1e-3;
    }
};

template <>
struct Equal<float> {
    static bool equal_value(float lhs, float rhs) {
        if (std::isfinite(lhs) && std::isfinite(rhs)) {
            return std::abs(lhs - rhs) < 1e-4;
        }
        return (std::isinf(lhs) && std::isinf(rhs)) || (std::isnan(lhs) && std::isnan(rhs));
    }
};

template <>
struct Equal<double> {
    static bool equal_value(double lhs, double rhs) {
        if (std::isfinite(lhs) && std::isfinite(rhs)) {
            return std::abs(lhs - rhs) < 1e-5;
        }
        return (std::isinf(lhs) && std::isinf(rhs)) || (std::isnan(lhs) && std::isnan(rhs));
    }
};

template <typename T>
struct Equal<std::vector<T>> {
    static bool equal_value(const std::vector<T>& lhs, const std::vector<T>& rhs) {
        return lhs.size() == rhs.size() && std::equal(begin(lhs), end(lhs), begin(rhs), Equal<T>::equal_value);
    }
};

template <>
struct Equal<SubGraphOpInputDescription::value_type> {
    static bool equal_value(SubGraphOpInputDescription::const_reference lhs,
                            SubGraphOpInputDescription::const_reference rhs) {
        const auto& lhs_type_info = lhs->get_type_info();
        const auto& rhs_type_info = rhs->get_type_info();
        if (lhs_type_info != rhs_type_info) {
            return false;
        }
        using SubGraphOp = ov::op::util::SubGraphOp;
        if (lhs_type_info == SubGraphOp::SliceInputDescription::get_type_info_static()) {
            const auto& l_input = static_cast<const SubGraphOp::SliceInputDescription&>(*lhs);
            const auto& r_input = static_cast<const SubGraphOp::SliceInputDescription&>(*rhs);
            return l_input.m_start == r_input.m_start && l_input.m_stride == r_input.m_stride &&
                   l_input.m_part_size == r_input.m_part_size && l_input.m_end == r_input.m_end &&
                   l_input.m_axis == r_input.m_axis;
        } else if (lhs_type_info == SubGraphOp::MergedInputDescription::get_type_info_static()) {
            return true;
        } else if (lhs_type_info == SubGraphOp::InvariantInputDescription::get_type_info_static()) {
            return true;
        }
        return false;
    }
};

template <>
struct Equal<SubGraphOpInputDescription> {
    static bool equal_value(const SubGraphOpInputDescription& lhs, const SubGraphOpInputDescription& rhs) {
        if (lhs.size() != rhs.size()) {
            return false;
        }
        return std::is_permutation(begin(lhs),
                                   end(lhs),
                                   begin(rhs),
                                   Equal<SubGraphOpInputDescription::value_type>::equal_value);
    }
};

template <>
struct Equal<SubGraphOpOutputDescription::value_type> {
    static bool equal_value(SubGraphOpOutputDescription::const_reference lhs,
                            SubGraphOpOutputDescription::const_reference rhs) {
        const auto& lhs_type_info = lhs->get_type_info();
        const auto& rhs_type_info = rhs->get_type_info();
        if (lhs_type_info != rhs_type_info) {
            return false;
        }
        using SubGraphOp = ov::op::util::SubGraphOp;
        if (lhs_type_info == SubGraphOp::ConcatOutputDescription::get_type_info_static()) {
            const auto& l_output = static_cast<const SubGraphOp::ConcatOutputDescription&>(*lhs);
            const auto& r_output = static_cast<const SubGraphOp::ConcatOutputDescription&>(*rhs);
            return l_output.m_start == r_output.m_start && l_output.m_stride == r_output.m_stride &&
                   l_output.m_part_size == r_output.m_part_size && l_output.m_end == r_output.m_end &&
                   l_output.m_axis == r_output.m_axis;
        } else if (lhs_type_info == SubGraphOp::BodyOutputDescription::get_type_info_static()) {
            const auto& l_output = static_cast<const SubGraphOp::BodyOutputDescription&>(*lhs);
            const auto& r_output = static_cast<const SubGraphOp::BodyOutputDescription&>(*rhs);
            return l_output.m_iteration == r_output.m_iteration;
        }
        return false;
    }
};

template <>
struct Equal<SubGraphOpOutputDescription> {
    static bool equal_value(const SubGraphOpOutputDescription& lhs, const SubGraphOpOutputDescription& rhs) {
        if (lhs.size() != rhs.size()) {
            return false;
        }
        return std::is_permutation(begin(lhs),
                                   end(lhs),
                                   begin(rhs),
                                   Equal<SubGraphOpOutputDescription::value_type>::equal_value);
    }
};

template <>
struct Equal<std::shared_ptr<ov::op::util::Variable>> {
    static bool equal_value(const std::shared_ptr<ov::op::util::Variable>& lhs,
                            const std::shared_ptr<ov::op::util::Variable>& rhs) {
        return lhs->get_info() == rhs->get_info();
    }
};

template <>
struct Equal<uint8_t*> {
    static constexpr uint8_t BITS_IN_BYTE_COUNT = 8;

    static inline uint8_t extract_bit(uint8_t val, uint8_t bit) {
        return (val >> bit) & 0x01;
    }

    static bool equal_value(const uint8_t* lhs, const uint8_t* rhs, size_t lhs_bit_size, size_t rhs_bit_size) {
        if (lhs_bit_size != rhs_bit_size)
            return false;

        for (size_t bit_idx = 0; bit_idx < lhs_bit_size; bit_idx++) {
            const size_t byte_idx = bit_idx / BITS_IN_BYTE_COUNT;

            const uint8_t bit_in_byte_idx = 7 - (bit_idx % BITS_IN_BYTE_COUNT);

            if (extract_bit(lhs[byte_idx], bit_in_byte_idx) != extract_bit(rhs[byte_idx], bit_in_byte_idx)) {
                return false;
            }
        }

        return true;
    }
};

using Constant = ov::op::v0::Constant;
template <>
struct Equal<std::shared_ptr<Constant>> {
    static bool equal_value(const std::shared_ptr<Constant>& lhs, const std::shared_ptr<Constant>& rhs) {
        const auto lhs_t = lhs->get_element_type();
        const auto rhs_t = rhs->get_element_type();
        if (lhs_t != rhs_t) {
            return false;
        }

        switch (lhs_t) {
        case ov::element::Type_t::u1: {
            const auto lhs_v = static_cast<const uint8_t*>(lhs->get_data_ptr());
            const auto rhs_v = static_cast<const uint8_t*>(rhs->get_data_ptr());
            const auto lhs_bit_size = shape_size(lhs->get_shape());
            const auto rhs_bit_size = shape_size(rhs->get_shape());
            return Equal<uint8_t*>::equal_value(lhs_v, rhs_v, lhs_bit_size, rhs_bit_size);
        }
        case ov::element::Type_t::bf16: {
            auto lhs_v = lhs->cast_vector<ov::bfloat16>();
            auto rhs_v = rhs->cast_vector<ov::bfloat16>();
            return Equal<std::vector<ov::bfloat16>>::equal_value(lhs_v, rhs_v);
            break;
        }
        case ov::element::Type_t::f16: {
            const auto& lhs_v = lhs->cast_vector<ov::float16>();
            const auto& rhs_v = rhs->cast_vector<ov::float16>();
            return Equal<std::vector<ov::float16>>::equal_value(lhs_v, rhs_v);
            break;
        }
        case ov::element::Type_t::f32: {
            const auto& lhs_v = lhs->cast_vector<float>();
            const auto& rhs_v = rhs->cast_vector<float>();
            return Equal<std::vector<float>>::equal_value(lhs_v, rhs_v);
            break;
        }
        case ov::element::Type_t::string: {
            const auto& lhs_v = lhs->cast_vector<std::string>();
            const auto& rhs_v = rhs->cast_vector<std::string>();
            return Equal<std::vector<std::string>>::equal_value(lhs_v, rhs_v);
            break;
        }
        default: {
            const auto& lhs_v = lhs->cast_vector<double>();
            const auto& rhs_v = rhs->cast_vector<double>();
            return Equal<std::vector<double>>::equal_value(lhs_v, rhs_v);
            break;
        }
        }
        return false;
    }
};

template <>
struct Equal<std::shared_ptr<ov::Dimension>> {
    static bool equal_value(const std::shared_ptr<ov::Dimension>& dim1, const std::shared_ptr<ov::Dimension>& dim2) {
        return dim1 == dim2;
    }
};

template <>
struct Equal<std::shared_ptr<ov::PartialShape>> {
    static bool equal_value(const std::shared_ptr<ov::PartialShape>& shape1,
                            const std::shared_ptr<ov::PartialShape>& shape2) {
        return shape1 == shape2;
    }
};

}  // namespace equal

namespace str {
template <typename...>
struct Void_t {
    using type = void;
};

template <typename T, typename = void>
struct Get {
    static std::string value(const T&) {
        return std::string("[Ups can't convert this to value: ") + typeid(T).name() + "]";
    }
};

template <typename T>
struct Get<T, typename Void_t<decltype(std::to_string(std::declval<T>()))>::type> {
    static std::string value(const T& v) {
        return "[" + std::to_string(v) + "]";
    }
};

template <>
struct Get<std::string, void> {
    static std::string value(const std::string& v) {
        return "[" + v + "]";
    }
};

template <typename T>
struct Get<T, typename Void_t<decltype(begin(std::declval<T>())), decltype(end(std::declval<T>()))>::type> {
    template <typename Container>
    static std::string join(const Container& c, const char* glue = ", ") {
        std::stringstream oss;
        const char* s = "";
        for (const auto& v : c) {
            oss << s << v;
            s = glue;
        }
        return oss.str();
    }

    static std::string value(const T& v) {
        return "[" + join(v) + "]";
    }
};

template <>
struct Get<ov::op::util::FrameworkNodeAttrs, void> {
    static std::string value(const ov::op::util::FrameworkNodeAttrs& attrs) {
        std::stringstream oss;
        const auto& a = attrs;
        oss << "version=" << attrs.get_opset_name() << ", ";
        oss << "type=" << attrs.get_type_name() << ", ";
        oss << "attrs[";
        for (const auto& item : a) {
            oss << item.first << "=" << item.second << " ";
        }
        oss << "]";
        return "[" + oss.str() + "]";
    }
};

template <>
struct Get<ov::Dimension, void> {
    static std::string value(const ov::Dimension& dim) {
        std::stringstream dim_str;
        dim_str << dim;
        return dim_str.str();
    }
};

template <>
struct Get<ov::PartialShape, void> {
    static std::string value(const ov::PartialShape& shape) {
        std::stringstream shape_str;
        shape_str << shape;
        return shape_str.str();
    }
};

template <>
struct Get<std::shared_ptr<ov::op::util::Variable>, void> {
    static std::string value(const std::shared_ptr<ov::op::util::Variable>& variable) {
        std::stringstream oss;
        const auto variable_info = variable->get_info();
        oss << "[";
        oss << "data_shape=" << variable_info.data_shape << ", ";
        oss << "data_type=" << variable_info.data_type << ", ";
        oss << "variable_id=" << variable_info.variable_id;
        oss << "]";
        return oss.str();
    }
};

}  // namespace str

class ReadAndCompareAttributes : public ov::AttributeVisitor {
public:
    ReadAndCompareAttributes(const ReadAndStoreAttributes& ref, Comparator::CmpValues check_flags)
        : m_attr_ref(ref),
          m_cmp_result{ref.read_result()},
          m_check_flags(check_flags) {}

    void on_adapter(const std::string& name, ov::ValueAccessor<void>& adapter) override {
        verify_others(name, adapter);
    }

#define ON_ADAPTER(TYPE) \
    void on_adapter(const std::string& name, ov::ValueAccessor<TYPE>& adapter) override { verify(name, adapter.get()); }

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

    void on_adapter(const std::string& name, ov::ValueAccessor<std::shared_ptr<ov::Model>>& adapter) override {
        verify_function(name, adapter);
    }

    bool all_attr_was_compared() const {
        return m_visited_attributes.size() == m_attr_ref.attributes_number();
    }

    size_t compared_attr_number() const {
        return m_visited_attributes.size();
    }

    const Result& cmp_result() const {
        return m_cmp_result;
    }

private:
    bool should_return() const {
        return m_fast_exit && m_cmp_result.has_error();
    }

    template <typename AttrValue>
    void verify(const std::string& name, const AttrValue& attr_value);

    void verify_mem_buf(const std::string& name, const std::shared_ptr<ov::AlignedBuffer>& buffer);
    void verify_string_aligned_buffer(const std::string& name, const std::shared_ptr<ov::StringAlignedBuffer>& buffer);

    using ModelAccessor = ov::ValueAccessor<std::shared_ptr<ov::Model>>;

    void verify_function(const std::string& name, ModelAccessor& adapter);

    void verify_others(const std::string& name, ov::ValueAccessor<void>& adapter);
    //-- DATA --
    const ReadAndStoreAttributes& m_attr_ref;
    Result m_cmp_result;
    Comparator::CmpValues m_check_flags;
    std::set<AttrName> m_visited_attributes;
    static constexpr bool m_fast_exit{true};
};
class CompareNodesAttributes {
public:
    using ReadAndStoreAttributes = detail::ReadAndStoreAttributes;
    using ReadAndCompareAttributes = detail::ReadAndCompareAttributes;

    CompareNodesAttributes(Comparator::CmpValues m_compare_flags) : m_compare_attr(m_store_attr, m_compare_flags) {}

    ReadAndStoreAttributes& get_ref_reader() {
        return m_store_attr;
    }

    ReadAndCompareAttributes& get_cmp_reader() {
        return m_compare_attr;
    }

    bool equal() const {
        return m_compare_attr.all_attr_was_compared() && !m_compare_attr.cmp_result().has_error();
    }

    friend std::string to_string(const CompareNodesAttributes& c) {
        const auto& result = c.m_compare_attr.cmp_result();
        if (result.has_error()) {
            return result.message();
        }
        if (!c.m_compare_attr.all_attr_was_compared()) {
            return "not all of attr was compared: " + to_str(c.m_compare_attr.compared_attr_number()) + " vs " +
                   to_str(c.m_store_attr.attributes_number());
        }
        return "looks good [compared " + to_str(c.m_compare_attr.compared_attr_number()) + " attributes]";
    }

private:
    ReadAndStoreAttributes m_store_attr;
    ReadAndCompareAttributes m_compare_attr;
};

}  // namespace detail

Comparator::Result compare(ov::Node* node1, ov::Node* node2, Comparator::CmpValues comparition_flags);

}  // namespace attributes

struct AccuracyCheckResult {
    bool status;
    std::string message;
};

AccuracyCheckResult accuracy_check(const std::shared_ptr<ov::Model>& ref_function,
                                   const std::shared_ptr<ov::Model>& cur_function,
                                   float abs_threshold,
                                   float rel_threshold);
