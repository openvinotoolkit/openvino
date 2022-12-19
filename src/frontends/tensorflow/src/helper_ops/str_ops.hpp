// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <cassert>

#include "openvino/frontend/tensorflow/frontend.hpp"
#include "helper_ops/internal_operation.hpp"
#include "openvino/frontend/tensorflow/decoder.hpp"
#include "openvino/opsets/opset1.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {


using ConstantVector = std::vector<std::shared_ptr<ov::opset1::Constant>>;

using std::make_shared;

inline bool all_inputs_are_constants (Node* node) {
    for(size_t i = 0; i < node->get_input_size(); ++i) {
        if(!is_type<ov::opset1::Constant>(node->get_input_node_shared_ptr(i))) {
            return false;
        }
    }
    return true;
}


template <typename Node>
ConstantVector evaluate_internal_helper (Node* node, const ov::TensorVector& inputs) {
    std::vector<std::shared_ptr<ov::opset1::Constant>> results;
    //return results;
    if(inputs.size() == 1 && inputs[0].get_shape().size() == 1) {
        // Scalar str implementation represented as 1d/u8
        auto in = inputs[0];

        std::cerr << "in.get_shape() = " << in.get_shape() << "\n";
        //auto out = HostTensor();
        auto str_input = std::string(reinterpret_cast<const char*>(in.data()), in.get_shape()[0]);
        auto str_output = node->evaluate_single(str_input);
        results.push_back(make_shared<ov::opset1::Constant>(element::u8, Shape{str_output.length()}, str_output.data()));
    } else {
        auto begins = reinterpret_cast<const int*>(inputs[0].data());    // i32
        auto ends = reinterpret_cast<const int*>(inputs[1].data());    // i32
        auto chars = reinterpret_cast<const char*>(inputs[2].data());    // u8
        size_t size = inputs[0].get_shape()[0];

        std::vector<int> new_begins, new_ends;
        std::string new_chars;

        for(size_t i = 0; i < size; ++i) {
            auto str_input = std::string(chars + begins[i], chars + ends[i]);
            std::cerr << "Input string " << i << " is " << str_input << "\n";
            auto str_output = node->evaluate_single(str_input);
            std::cerr << "Output string " << i << " is " << str_output << "\n";
            new_begins.push_back(new_chars.length());
            new_chars += str_output;
            new_ends.push_back(new_chars.length());
        }

        results.push_back(make_shared<ov::opset1::Constant>(element::i32, Shape{size}, &new_begins[0]));
        results.push_back(make_shared<ov::opset1::Constant>(element::i32, Shape{size}, &new_ends[0]));
        results.push_back(make_shared<ov::opset1::Constant>(element::u8, Shape{new_chars.length()}, new_chars.data()));
    }
    return results;
}

template <typename Node>
inline void validate_and_infer_types_helper_1_to_1 (Node* node) {

    // Shape/type prop mode
    // FIXME: suppose it the same number of outputs as inputs of the same type
    for(size_t i = 0; i < node->get_input_size(); ++i) {
        node->set_output_type(i, node->get_input_element_type(i), node->get_input_partial_shape(i));
        StructuralTypeAttribute::copy(
            node->get_input_tensor(i).get_rt_info(),
            node->get_output_tensor(i).get_rt_info());
    }

    if(all_inputs_are_constants(node)) {
        // Evaluate mode
        std::cerr << "[ EVALUATION MODE ]";
        ov::TensorVector inputs;
        ConstantVector outputs;
        // FIXME: remove this part when CPU fixes evaluate with internally dynamic operations
        for(size_t i = 0; i < node->get_input_size(); ++i) {
            auto constant = std::dynamic_pointer_cast<ov::opset1::Constant>(node->get_input_node_shared_ptr(i));
            inputs.push_back(Tensor(constant->get_element_type(), constant->get_shape(), const_cast<void*>(constant->get_data_ptr())));
        }
        outputs = evaluate_internal_helper(node, inputs);

        for(size_t i = 0; i < node->get_output_size(); ++i) {
            node->set_output_type(i, outputs[i]->get_element_type(), outputs[i]->get_shape());
        }
    }
}


template <typename Node>
bool evaluate_helper (Node* node, ov::TensorVector& outputs, const ov::TensorVector& inputs) {
    auto results = evaluate_internal_helper(node, inputs);
    for(size_t i = 0; i < results.size(); ++i) {
        size_t length = results[i]->get_byte_size();
        std::cerr << "Expected length " << length << ", allocated: " << outputs[i].get_shape() << "\n";
        memcpy(outputs[i].data(), results[i]->get_data_ptr(), length);
    }
    return true;
}

// https://www.tensorflow.org/text/api_docs/python/text/case_fold_utf8
class CaseFoldUTF8 : public ov::op::Op {
public:
    OPENVINO_OP("CaseFoldUTF8");

    CaseFoldUTF8(const OutputVector& arguments) : ov::op::Op(arguments) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        return validate_and_infer_types_helper_1_to_1(this);

        std::cerr << "+++++++++++ CaseFoldUTF8 +++++++++++\n";
        std::cerr << input_value(0).get_node_shared_ptr() << "\n";
        if(std::dynamic_pointer_cast<ov::opset1::Constant>(input_value(0).get_node_shared_ptr())) {
            set_output_type(0, get_input_element_type(0), PartialShape{get_input_partial_shape(0)[0] + addon.size()});
            std::cerr << "Evaluate mode\n";
        } else {
            set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
            std::cerr << "Validation mode\n";
        }

        StructuralTypeAttribute::copy(get_input_tensor(0).get_rt_info(), get_output_tensor(0).get_rt_info());
        std::cerr << "**********************\n";
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        std::cerr << "[ CLONING ] CaseFoldUTF8\n";
        return std::make_shared<CaseFoldUTF8>(inputs);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
        return evaluate_helper(this, outputs, inputs);
    }

    std::string evaluate_single(const std::string& input) const {
        return "CaseFoldUTF8(" + input + ")";
    }

    bool has_evaluate() const {
        return true;
    }

    std::string addon = "CaseFoldUTF8()";
};


class NormalizeUTF8 : public ov::op::Op {
public:
    OPENVINO_OP("NormalizeUTF8");

    NormalizeUTF8(
        const OutputVector& arguments,
        const std::string& normalization_form)
    : ov::op::Op(arguments),
        m_normalization_form(normalization_form)
    {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        return validate_and_infer_types_helper_1_to_1(this);
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        return std::make_shared<NormalizeUTF8>(inputs, m_normalization_form);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("normalization_form", m_normalization_form);
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
        return evaluate_helper(this, outputs, inputs);
    }

    std::string evaluate_single(const std::string& input) const {
        return "NormalizeUTF8(" + input + ", normalization_form = " + m_normalization_form + ")";
    }

    bool has_evaluate() const {
        return true;
    }

    std::string addon = "NormalizeUTF8(, normalization_form = )";
    std::string m_normalization_form;
};


class StaticRegexReplace : public ov::op::Op {
public:
    OPENVINO_OP("StaticRegexReplace");

    StaticRegexReplace(
        const OutputVector& arguments,
        const std::string& pattern,
        const std::string& rewrite)
    : ov::op::Op(arguments), m_pattern(pattern), m_rewrite(rewrite) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        return validate_and_infer_types_helper_1_to_1(this);
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        //std::cerr << "[ CLONING ] StaticRegexReplace\n";
        return std::make_shared<StaticRegexReplace>(inputs, m_pattern, m_rewrite);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("pattern", m_pattern);
        visitor.on_attribute("rewrite", m_rewrite);
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
        return evaluate_helper(this, outputs, inputs);
    }

    std::string evaluate_single(const std::string& input) const {
        return "StaticRegexReplace(" + input + ", pattern = " + m_pattern + ", rewrite = " + m_rewrite + ")";
    }

    bool has_evaluate() const {
        return true;
    }

private:

    std::string addon = "StaticRegexReplace(, pattern = , rewrite = )";
    std::string m_pattern;
    std::string m_rewrite;
};



}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
