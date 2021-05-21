// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <limits>
#include <memory>
#include <queue>

#include <ngraph/dimension.hpp>
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pass/pass.hpp>
#include <ngraph/opsets/opset6.hpp>

#include "ie_common.h"
#include <ngraph_ops/framework_node.hpp>

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
class Comparator {
public:
    using CmpValues = FunctionsComparator::CmpValues;
    using Result = FunctionsComparator::Result;
    using ComparedNodes = std::pair<ngraph::Node*, ngraph::Node*>;

    explicit Comparator(CmpValues f) : m_comparition_flags(f) {}

    Result compare(
            const std::shared_ptr<ngraph::Function>& f1, const std::shared_ptr<ngraph::Function>& f2);

    Result compare(ngraph::Node* node1, ngraph::Node* node2) {
        std::stringstream errors;
        const auto result = compare(node1, node2, errors);
        if (!result.valid) {
            return result;
        }
        const auto msg = errors.str();
        return msg.empty() ? Result::ok() : Result::error(msg);
    }

    Comparator recreate() const {
        return Comparator(m_comparition_flags);
    }

    void compare_inputs(ngraph::Node* node1, ngraph::Node* node2, std::ostream& err_log);

    void compare_outputs(ngraph::Node* node1, ngraph::Node* node2, std::ostream& err_log);

private:
    bool should_compare(CmpValues f) const noexcept {
        return m_comparition_flags & f;
    }

    ///
    /// \param err_log - will be fill by minor errors if happen
    /// \return only fatality error if some minor one appears it will be add to err_log
    ///
    Result compare(ngraph::Node* node1, ngraph::Node* node2, std::ostream& err_log);

    void add_nodes_inputs_to_queue(ngraph::Node* node1, ngraph::Node* node2);

    //-- DATA --
    CmpValues m_comparition_flags;

    std::queue<ComparedNodes> q;
    std::unordered_set<ngraph::Node*> used;
};

inline namespace tools {
template<typename T>
std::string to_str(const T &v) {
    using std::to_string;
    return to_string(v);
}
template<typename Node>
std::string name(const Node &n) {
    return n->get_friendly_name();
}
}
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
                private AttributeStorage<std::shared_ptr<ngraph::Function>>,
                private AttributeStorage<SubGraphOpInputDescription>,
                private AttributeStorage<SubGraphOpOutputDescription>,
                private AttributeStorage<ngraph::op::FrameworkNodeAttrs> {
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
               storage<std::shared_ptr<ngraph::Function>>().get_attributes_number() +
               storage<SubGraphOpInputDescription>().get_attributes_number() +
               storage<SubGraphOpOutputDescription>().get_attributes_number() +
               storage<ngraph::op::FrameworkNodeAttrs>().get_attributes_number();
    }
};

}  // namespace storage

class ReadAndStoreAttributes : public ngraph::AttributeVisitor, protected storage::Storage {
public:
    void on_adapter(const std::string& name, ngraph::ValueAccessor<void>& adapter) override;

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
    ON_ADAPTER(std::shared_ptr<ngraph::Function>)

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
struct Equal<ngraph::bfloat16> {
    static bool equal_value(ngraph::bfloat16 lhs, ngraph::bfloat16 rhs) {
        if (lhs.to_bits() == rhs.to_bits()) {
            return true;
        }
        return std::abs(lhs - rhs) < 1e-3;
    }
};

template <>
struct Equal<ngraph::float16> {
    static bool equal_value(ngraph::float16 lhs, ngraph::float16 rhs) {
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
        return (std::isinf(lhs) && std::isinf(rhs)) ||
               (std::isnan(lhs) && std::isnan(rhs));
    }
};

template <>
struct Equal<double> {
    static bool equal_value(double lhs, double rhs) {
        if (std::isfinite(lhs) && std::isfinite(rhs)) {
            return std::abs(lhs - rhs) < 1e-5;
        }
        return (std::isinf(lhs) && std::isinf(rhs)) ||
               (std::isnan(lhs) && std::isnan(rhs));
    }
};

template <typename T>
struct Equal<std::vector<T>> {
    static bool equal_value(const std::vector<T>& lhs, const std::vector<T>& rhs) {
        return lhs.size() == rhs.size() &&
               std::equal(begin(lhs), end(lhs), begin(rhs), Equal<T>::equal_value);
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
struct Equal<uint8_t*> {
    static constexpr uint8_t BITS_IN_BYTE_COUNT = 8;

    static inline uint8_t extract_bit(uint8_t val, uint8_t bit) {
        return (val >> bit) & 0x01;
    }

    static bool equal_value(const uint8_t* lhs, const uint8_t* rhs,
                            size_t lhs_bit_size, size_t rhs_bit_size) {
        if (lhs_bit_size != rhs_bit_size) return false;

        for (size_t bit_idx = 0; bit_idx < lhs_bit_size; bit_idx++) {
            const auto byte_idx_result(bit_idx / BITS_IN_BYTE_COUNT);
            if (byte_idx_result > std::numeric_limits<uint8_t>::max())
                IE_THROW() << "(bit_idx / BITS_IN_BYTE_COUNT) bigger than uint8_t::max_value";

            const uint8_t byte_idx(static_cast<uint8_t>(byte_idx_result));
            const uint8_t bit_in_byte_idx = 7 - (bit_idx % BITS_IN_BYTE_COUNT);

            if (extract_bit(lhs[byte_idx], bit_in_byte_idx) !=
                extract_bit(rhs[byte_idx], bit_in_byte_idx)) {
                return false;
            }
        }

        return true;
    }
};

using Constant = ngraph::opset1::Constant;
template <>
struct Equal<std::shared_ptr<Constant>> {
    static bool equal_value(
            const std::shared_ptr<Constant>& lhs, const std::shared_ptr<Constant>& rhs) {
        const auto lhs_t = lhs->get_element_type();
        const auto rhs_t = rhs->get_element_type();
        if (lhs_t != rhs_t) {
            return false;
        }

        switch (lhs_t) {
            case ngraph::element::Type_t::u1: {
                const auto lhs_v = static_cast<const uint8_t*>(lhs->get_data_ptr());
                const auto rhs_v = static_cast<const uint8_t*>(rhs->get_data_ptr());
                const auto lhs_bit_size = shape_size(lhs->get_shape());
                const auto rhs_bit_size = shape_size(rhs->get_shape());
                return Equal<uint8_t*>::equal_value(lhs_v, rhs_v, lhs_bit_size,
                                                    rhs_bit_size);
            }
            case ngraph::element::Type_t::bf16: {
                auto lhs_v = lhs->cast_vector<ngraph::bfloat16>();
                auto rhs_v = rhs->cast_vector<ngraph::bfloat16>();
                return Equal<std::vector<ngraph::bfloat16>>::equal_value(lhs_v, rhs_v);
                break;
            }
            case ngraph::element::Type_t::f16: {
                const auto& lhs_v = lhs->cast_vector<ngraph::float16>();
                const auto& rhs_v = rhs->cast_vector<ngraph::float16>();
                return Equal<std::vector<ngraph::float16>>::equal_value(lhs_v, rhs_v);
                break;
            }
            case ngraph::element::Type_t::f32: {
                const auto& lhs_v = lhs->cast_vector<float>();
                const auto& rhs_v = rhs->cast_vector<float>();
                return Equal<std::vector<float>>::equal_value(lhs_v, rhs_v);
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

template <>
struct Get<ngraph::op::FrameworkNodeAttrs, void> {
    static std::string value(const ngraph::op::FrameworkNodeAttrs& attrs) {
        std::stringstream oss;
        const auto & a = attrs;
        oss << "version=" << attrs.get_opset_name() << ", ";
        oss << "type=" << attrs.get_type_name() << ", ";
        oss << "attrs[";
        for (const auto & item : a) {
            oss << item.first << "=" << item.second << " ";
        }
        oss << "]";
        return "[" + oss.str() + "]";
    }
};


}  // namespace str

class ReadAndCompareAttributes : public ngraph::AttributeVisitor {
public:
    ReadAndCompareAttributes(const ReadAndStoreAttributes& ref, Comparator::CmpValues check_flags)
            : m_attr_ref(ref), m_cmp_result{ref.read_result()}, m_check_flags(check_flags) {}

    void on_adapter(const std::string& name, ngraph::ValueAccessor<void>& adapter) override {
        verify_others(name, adapter);
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
            const std::string& name,
            ngraph::ValueAccessor<std::shared_ptr<ngraph::Function>>& adapter) override {
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

    void verify_mem_buf(
            const std::string& name, const std::shared_ptr<ngraph::runtime::AlignedBuffer>& buffer);

    using FunctionAccessor = ngraph::ValueAccessor<std::shared_ptr<ngraph::Function>>;

    void verify_function(const std::string& name, FunctionAccessor& adapter);

    void verify_others(const std::string& name, ngraph::ValueAccessor<void>& adapter);
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

    CompareNodesAttributes(Comparator::CmpValues m_compare_flags)
            : m_compare_attr(m_store_attr, m_compare_flags) {}

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
            return "not all of attr was compared: " +
                   to_str(c.m_compare_attr.compared_attr_number()) + " vs " +
                   to_str(c.m_store_attr.attributes_number());
        }
        return "looks good [compared " + to_str(c.m_compare_attr.compared_attr_number()) +
               " attributes]";
    }

private:
    ReadAndStoreAttributes m_store_attr;
    ReadAndCompareAttributes m_compare_attr;
};

}  // namespace detail

Comparator::Result compare(ngraph::Node* node1, ngraph::Node* node2, Comparator::CmpValues comparition_flags);

}  // namespace attributes