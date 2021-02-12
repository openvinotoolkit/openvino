// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <queue>

#include <ngraph/dimension.hpp>
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pass/pass.hpp>
#include <ngraph/opsets/opset6.hpp>
#include "test_common.hpp"

#define DYN ngraph::Dimension::dynamic()

using TransformationTests = CommonTestUtils::TestsCommon;

class FunctionsComparator {
public:
    enum CmpValues {
        NONE = 0,
        CONST_VALUES = 1 << 0,
        NAMES = 1 << 1,
        RUNTIME_KEYS = 1 << 2,
        PRECISIONS = 1 << 3,
        ATTRIBUTES = 1 << 4,
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

    static constexpr FunctionsComparator no_default() noexcept {
        return FunctionsComparator{NONE};
    }
    static constexpr FunctionsComparator with_default() noexcept {
        return FunctionsComparator{PRECISIONS};
    }
    FunctionsComparator& enable(CmpValues f) noexcept {
        m_comparition_flags = static_cast<CmpValues>(m_comparition_flags | f);
        return *this;
    }
    constexpr bool should_compare(CmpValues f) const noexcept {
        return m_comparition_flags & f;
    }
    Result compare(
        const std::shared_ptr<ngraph::Function>& f1,
        const std::shared_ptr<ngraph::Function>& f2) const;

    Result operator()(
        const std::shared_ptr<ngraph::Function>& f1,
        const std::shared_ptr<ngraph::Function>& f2) const {
        return compare(f1, f2);
    }

private:
    constexpr explicit FunctionsComparator(CmpValues f) noexcept : m_comparition_flags(f) {}
    CmpValues m_comparition_flags;
};

///
/// \deprecated
/// \brief compare_functions is obsolete function use FunctionsComparator instead.
///
inline std::pair<bool, std::string> compare_functions(
    const std::shared_ptr<ngraph::Function>& f1,
    const std::shared_ptr<ngraph::Function>& f2,
    const bool compareConstValues = false,
    const bool compareNames = false,
    const bool compareRuntimeKeys = false,
    const bool comparePrecisions = true,
    const bool compareAttributes = false) {
    auto fc = FunctionsComparator::no_default();

    using Cmp = FunctionsComparator::CmpValues;
    if (compareConstValues) fc.enable(Cmp::CONST_VALUES);
    if (compareNames) fc.enable(Cmp::NAMES);
    if (compareRuntimeKeys) fc.enable(Cmp::RUNTIME_KEYS);
    if (comparePrecisions) fc.enable(Cmp::PRECISIONS);
    if (compareAttributes) fc.enable(Cmp::ATTRIBUTES);

    const auto r = fc(f1, f2);
    return {r.valid, r.message};
}

void check_rt_info(const std::shared_ptr<ngraph::Function>& f);

namespace ngraph {
namespace pass {
class InjectionPass;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::InjectionPass : public ngraph::pass::FunctionPass {
public:
    using injection_callback = std::function<void(std::shared_ptr<ngraph::Function>)>;

    explicit InjectionPass(injection_callback callback)
        : FunctionPass(), m_callback(std::move(callback)) {}

    bool run_on_function(std::shared_ptr<ngraph::Function> f) override {
        m_callback(f);
        return false;
    }

private:
    injection_callback m_callback;
};

template <typename T>
size_t count_ops_of_type(std::shared_ptr<ngraph::Function> f) {
    size_t count = 0;
    for (auto op : f->get_ops()) {
        if (ngraph::is_type<T>(op)) {
            count++;
        }
    }

    return count;
}

class TestOpMultiOut : public ngraph::op::Op {
public:
    NGRAPH_RTTI_DECLARATION;
    TestOpMultiOut() = default;

    TestOpMultiOut(const ngraph::Output<Node>& output_1, const ngraph::Output<Node>& output_2)
        : Op({output_1, output_2}) {
        validate_and_infer_types();
    }
    void validate_and_infer_types() override {
        set_output_size(2);
        set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
        set_output_type(1, get_input_element_type(1), get_input_partial_shape(1));
    }

    std::shared_ptr<Node> clone_with_new_inputs(
        const ngraph::OutputVector& new_args) const override {
        return std::make_shared<TestOpMultiOut>(new_args.at(0), new_args.at(1));
    }
};
// TODO: split definitions and implementations bellow
namespace attr_comparison {

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

using SubGraphOpInputDescription =
std::vector<std::shared_ptr<ngraph::op::util::SubGraphOp::InputDescription>>;

using SubGraphOpOutputDescription =
std::vector<std::shared_ptr<ngraph::op::util::SubGraphOp::OutputDescription>>;

using SpecialBodyPorts = ngraph::opset6::Loop::SpecialBodyPorts;

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
                private AttributeStorage<SubGraphOpInputDescription>,
                private AttributeStorage<SubGraphOpOutputDescription>,
                private AttributeStorage<SpecialBodyPorts> {
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
               storage<std::vector<std::string>>().get_attributes_number() +
               storage<SubGraphOpInputDescription>().get_attributes_number() +
               storage<SubGraphOpOutputDescription>().get_attributes_number() +
               storage<SpecialBodyPorts>().get_attributes_number();
    }
};

}  // namespace storage

class ReadAndStoreAttributes : public ngraph::AttributeVisitor, protected storage::Storage {
public:
    void on_adapter(const std::string& name, ngraph::ValueAccessor<void>& adapter) override {
        if (auto inputs =
                ngraph::as_type<ngraph::AttributeAdapter<SubGraphOpInputDescription>>(&adapter)) {
            insert(name, inputs->get());
        } else if (
                auto outputs =
                        ngraph::as_type<ngraph::AttributeAdapter<SubGraphOpOutputDescription>>(&adapter)) {
            insert(name, outputs->get());
        } else if (
                auto ports = ngraph::as_type<ngraph::AttributeAdapter<SpecialBodyPorts>>(&adapter)) {
            insert(name, ports->get());
        } else {
            m_read_result += "store   attr [ ERR ]: " + name +
                             " [drop `void` comparison which is '" + adapter.get_type_info().name +
                             "']";
        }
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<void*>& adapter) override {
        const auto beg = static_cast<unsigned char*>(adapter.get_ptr());
        const auto end = beg + adapter.size();
        insert(name, storage::MemoryChunk{storage::MemoryChunk::Data(beg, end)});
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
            const std::string&, ngraph::ValueAccessor<std::shared_ptr<ngraph::Function>>&) override {
        // handled by `compare_functions` drop it here
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
struct Equal<float> {
    static bool equal_value(float lhs, float rhs) {
        return std::abs(lhs - rhs) < 1e-5;
    }
};

template <>
struct Equal<double> {
    static bool equal_value(double lhs, double rhs) {
        return std::abs(lhs - rhs) < 1e-5;
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

template <>
struct Equal<SubGraphOpInputDescription::value_type> {
    static bool equal_value(
            SubGraphOpInputDescription::const_reference lhs,
            SubGraphOpInputDescription::const_reference rhs) {
        const auto& lhs_type_info = lhs->get_type_info();
        const auto& rhs_type_info = rhs->get_type_info();
        if (lhs_type_info != rhs_type_info) {
            return false;
        }
        using SubGraphOp = ngraph::op::util::SubGraphOp;
        if (lhs_type_info == SubGraphOp::SliceInputDescription::type_info) {
            const auto& l_input = static_cast<const SubGraphOp::SliceInputDescription&>(*lhs);
            const auto& r_input = static_cast<const SubGraphOp::SliceInputDescription&>(*rhs);
            return l_input.m_start == r_input.m_start && l_input.m_stride == r_input.m_stride &&
                   l_input.m_part_size == r_input.m_part_size && l_input.m_end == r_input.m_end &&
                   l_input.m_axis == r_input.m_axis;
        } else if (lhs_type_info == SubGraphOp::MergedInputDescription::type_info) {
            return true;
        } else if (lhs_type_info == SubGraphOp::InvariantInputDescription::type_info) {
            return true;
        }
        return false;
    }
};

template <>
struct Equal<SubGraphOpInputDescription> {
    static bool equal_value(
            const SubGraphOpInputDescription& lhs, const SubGraphOpInputDescription& rhs) {
        if (lhs.size() != rhs.size()) {
            return false;
        }
        return std::is_permutation(
                begin(lhs), end(lhs), begin(rhs),
                Equal<SubGraphOpInputDescription::value_type>::equal_value);
    }
};

template <>
struct Equal<SubGraphOpOutputDescription::value_type> {
    static bool equal_value(
            SubGraphOpOutputDescription::const_reference lhs,
            SubGraphOpOutputDescription::const_reference rhs) {
        const auto& lhs_type_info = lhs->get_type_info();
        const auto& rhs_type_info = rhs->get_type_info();
        if (lhs_type_info != rhs_type_info) {
            return false;
        }
        using SubGraphOp = ngraph::op::util::SubGraphOp;
        if (lhs_type_info == SubGraphOp::ConcatOutputDescription::type_info) {
            const auto& l_output = static_cast<const SubGraphOp::ConcatOutputDescription&>(*lhs);
            const auto& r_output = static_cast<const SubGraphOp::ConcatOutputDescription&>(*rhs);
            return l_output.m_start == r_output.m_start && l_output.m_stride == r_output.m_stride &&
                   l_output.m_part_size == r_output.m_part_size &&
                   l_output.m_end == r_output.m_end && l_output.m_axis == r_output.m_axis;
        } else if (lhs_type_info == SubGraphOp::BodyOutputDescription::type_info) {
            const auto& l_output = static_cast<const SubGraphOp::BodyOutputDescription&>(*lhs);
            const auto& r_output = static_cast<const SubGraphOp::BodyOutputDescription&>(*rhs);
            return l_output.m_iteration == r_output.m_iteration;
        }
        return false;
    }
};

template <>
struct Equal<SubGraphOpOutputDescription> {
    static bool equal_value(
            const SubGraphOpOutputDescription& lhs, const SubGraphOpOutputDescription& rhs) {
        if (lhs.size() != rhs.size()) {
            return false;
        }
        return std::is_permutation(
                begin(lhs), end(lhs), begin(rhs),
                Equal<SubGraphOpOutputDescription::value_type>::equal_value);
    }
};

template <>
struct Equal<SpecialBodyPorts> {
    static bool equal_value(const SpecialBodyPorts& lhs, const SpecialBodyPorts& rhs) {
        return lhs.current_iteration_input_idx == rhs.current_iteration_input_idx;
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
struct Get<
        T,
        typename Void_t<decltype(begin(std::declval<T>())), decltype(end(std::declval<T>()))>::type> {
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

}  // namespace str

class ReadAndCompareAttributes : public ngraph::AttributeVisitor {
public:
    ReadAndCompareAttributes(const ReadAndStoreAttributes& ref)
            : m_attr_ref(ref), m_cmp_result{ref.read_result()} {}

    void on_adapter(const std::string& name, ngraph::ValueAccessor<void>& adapter) override {
        if (should_return()) {
            return;
        }
        m_visited_attributes.insert(name);
        if (auto inputs =
                ngraph::as_type<ngraph::AttributeAdapter<SubGraphOpInputDescription>>(&adapter)) {
            verify(name, inputs->get());
        } else if (
                auto outputs =
                        ngraph::as_type<ngraph::AttributeAdapter<SubGraphOpOutputDescription>>(&adapter)) {
            verify(name, outputs->get());
        } else if (
                auto ports = ngraph::as_type<ngraph::AttributeAdapter<SpecialBodyPorts>>(&adapter)) {
            verify(name, ports->get());
        } else {
            m_cmp_result += "compare attr [ ERR ]: " + name +
                            " [drop `void` comparison which is '" + adapter.get_type_info().name +
                            "']";
        }
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<void*>& adapter) override {
        if (should_return()) {
            return;
        }
        m_visited_attributes.insert(name);
        const auto ref_value = m_attr_ref.get<storage::MemoryChunk>(name);
        if (!ref_value) {
            m_cmp_result += "missing attribute name: '" + name + "'";
            return;
        }

        if (adapter.size() != ref_value->size() ||
            std::memcmp(ref_value->data(), adapter.get_ptr(), ref_value->size()) != 0) {
            m_cmp_result += "mismatch in value: '" + name + "' : look in to the mem buffer";
            return;
        }
    }

#define ON_ADAPTER(TYPE)                                                                      \
    void on_adapter(const std::string& name, ngraph::ValueAccessor<TYPE>& adapter) override { \
        verify(name, adapter.get());                                                          \
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
            const std::string&, ngraph::ValueAccessor<std::shared_ptr<ngraph::Function>>&) override {
        // handled by `compare_functions` drop it here
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
    void verify(const std::string& name, const AttrValue& attr_value) {
        if (should_return()) {
            return;
        }
        m_visited_attributes.insert(name);
        const auto ref_value = m_attr_ref.get<AttrValue>(name);
        if (!ref_value) {
            m_cmp_result += "missing attribute name: '" + name + "'";
            return;
        }

        if (!equal::Equal<AttrValue>::equal_value(*ref_value, attr_value)) {
            m_cmp_result += "mismatch in value: '" + name +
                            "' : " + str::Get<AttrValue>::value(*ref_value) + " vs " +
                            str::Get<AttrValue>::value(attr_value);
        }
    }

    const ReadAndStoreAttributes& m_attr_ref;
    Result m_cmp_result;
    std::set<AttrName> m_visited_attributes;
    bool m_fast_exit{true};
};

}  // namespace attr_comparison


class CompareNodesAttributes {
public:
    CompareNodesAttributes() : m_compare_attr(m_store_attr) {}

    attr_comparison::ReadAndStoreAttributes& get_ref_reder() {
        return m_store_attr;
    }

    attr_comparison::ReadAndCompareAttributes& get_cmp_reader() {
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
            return "not all of attr was compared: " +
                   std::to_string(c.m_compare_attr.compared_attr_number()) + " vs " +
                   std::to_string(c.m_store_attr.attributes_number());
        }
        return "looks good [compared " + std::to_string(c.m_compare_attr.compared_attr_number()) +
               " attributes]";
    }

private:
    attr_comparison::ReadAndStoreAttributes m_store_attr;
    attr_comparison::ReadAndCompareAttributes m_compare_attr;
};
