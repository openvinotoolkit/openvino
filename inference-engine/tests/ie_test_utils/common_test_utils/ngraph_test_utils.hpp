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
/// \brief compare_functions is obsolete function use FunctionComparator instead.
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
