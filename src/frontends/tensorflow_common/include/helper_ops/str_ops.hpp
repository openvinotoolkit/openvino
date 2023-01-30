// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <cassert>

#include "helper_ops/internal_operation.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/core/type/non_tensor_type.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {


class StructuralTypeWA {
public:
    StructuralTypeWA (Any structural_type, Any tensor) :
        m_structural_type(structural_type), m_tensor(tensor) {}

    Any m_structural_type;
    Any m_tensor;
};

namespace StructuralTypeProxy {


namespace StructuralType = element::StructuralType;


class Base {
public:

    Base (Any type) : m_type(type) {}

    virtual Any type () const {
        return m_type;
    }

    Any m_type;
};


template <typename PLowerTensorType>
class Tensor : public Base {
public:

    Tensor (const PLowerTensorType& tensor) : Base(StructuralType::Tensor(tensor->get_element_type())), m_tensor(tensor) {}
    PLowerTensorType m_tensor;
};


// Regular tensor with strings
template <typename PLowerTensorType>
class TensorStr : public Base {
public:

    TensorStr (const PLowerTensorType& begins, const PLowerTensorType& ends, const PLowerTensorType& values) :
        Base(StructuralType::Tensor(StructuralType::Str())), m_begins(begins), m_ends(ends), m_values(values)
    {}

    TensorStr (std::vector<PLowerTensorType>& inputs) :
        TensorStr(inputs[0], inputs[1], inputs[2])
    {}

    TensorStr (const PLowerTensorType& values) :
        Base(StructuralType::Tensor(StructuralType::Str())), m_values(values)
    {}

    PartialShape get_partial_shape () {
        return m_begins->get_partial_shape();
    }

    Shape get_shape () {
        return m_begins->get_shape();
    }

    std::string element_by_offset (size_t i) {
        const char* values = reinterpret_cast<char*>(m_values->template data<uint8_t>());
        if (m_begins) {
            auto begin = m_begins->template data<int32_t>()[i];
            auto end = m_ends->template data<int32_t>()[i];
            return std::string(values + begin, values + end);
        } else {
            return std::string(values, values + shape_size(m_values->get_shape()));
        }
    }

    PLowerTensorType m_begins = nullptr;
    PLowerTensorType m_ends = nullptr;
    PLowerTensorType m_values;
};


struct BindInput {
    BindInput (const std::vector<size_t>& _inputs, Any _structural_type) :
        inputs(_inputs), structural_type(_structural_type)
    {}

    BindInput (std::size_t start, std::size_t end, Any _structural_type) : structural_type(_structural_type) {
        for(size_t i = start; i < end; ++i)
            inputs.push_back(i);
    }

    std::vector<size_t> inputs;
    Any structural_type;
};


using BindInputs = std::vector<BindInput>;


inline std::vector<std::shared_ptr<Base>> structural_input_types (const std::vector<Input<Node>>& node_inputs, const BindInputs bindings) {
    std::vector<std::shared_ptr<Base>> result;
    // If there is structural types present in input tensors, they sould be segmented by means of marks in rt_info in corresponding tensors, each segment represent a single
    for(size_t i = 0; i < bindings.size(); ++i) {
        const auto& st = bindings[i].structural_type;
        const auto& indices = bindings[i].inputs;
        if(st.is<StructuralType::Tensor>()) {
            auto tensor = st.as<StructuralType::Tensor>();
            const Any& element_type = tensor.element_type;
            if(element_type.is<element::Type>()) {
                assert(indices.size() == 1);
                result.push_back(std::make_shared<Tensor<const descriptor::Tensor*>>(&node_inputs[indices[0]].get_tensor()));
            } else if(element_type.is<StructuralType::Str>()) {
                if(indices.size() == 1) {
                    result.push_back(std::make_shared<TensorStr<const descriptor::Tensor*>>(&node_inputs[indices[0]].get_tensor()));
                } else {
                    result.push_back(std::make_shared<TensorStr<const descriptor::Tensor*>>(&node_inputs[indices[0]].get_tensor(), &node_inputs[indices[1]].get_tensor(), &node_inputs[indices[2]].get_tensor()));
                }
            }
        } else {
            throw std::string("Type binding has unsupported structural data type");
        }
    }
}


class FRONTEND_API StructuralTypeMapAttribute : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("structural_type_mapX", "0");

    StructuralTypeMapAttribute() = default;

    StructuralTypeMapAttribute(const BindInputs& value) : value(value) {}

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        // TODO: Implement deserialization; now only serialization works
        auto str_value = to_string();
        visitor.on_attribute("value", str_value);
        return true;
    }

    std::string to_string() const override {
        std::ostringstream buf;

        for(size_t i = 0; i < value.size(); ++i) {
            if(i > 0) {
                buf << ", ";
            }
            StructuralType::print(buf, value[i].structural_type);
            const auto& inputs = value[i].inputs;
            buf << '(';
            for(size_t j = 0; j < inputs.size(); ++j) {
                if(j > 0) {
                    buf << ", ";
                }
                buf << inputs[j];
            }
            buf << ')';
        }

        return buf.str();
    }

    void set_input (Node::RTMap& rt_info) {
        rt_info["structural_type_input_map"] = *this;
    }

    void set_output (Node::RTMap& rt_info) {
        rt_info["structural_type_output_map"] = *this;
    }

    static BindInputs get_input (const Node::RTMap& rt_info) {
        auto p = rt_info.find("structural_type_input_map");
        if(p != rt_info.end()) {
            return p->second.as<StructuralTypeMapAttribute>().value;
        } else {
            return BindInputs();
        }
    }

    static BindInputs get_output (const Node::RTMap& rt_info) {
        auto p = rt_info.find("structural_type_output_map");
        if(p != rt_info.end()) {
            return p->second.as<StructuralTypeMapAttribute>().value;
        } else {
            return BindInputs();
        }
    }

    BindInputs value;
};


}

class FRONTEND_API StructuralTypeAttribute : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("structural_type", "0");

    StructuralTypeAttribute() = default;

    StructuralTypeAttribute(const ov::Any& value) : value(value) {}

    //Any merge(const ngraph::NodeVector& nodes) const override;

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        // TODO: Implement deserialization; now only serialization works
        auto str_value = to_string();
        visitor.on_attribute("value", str_value);
        return true;
    }

    std::string to_string() const override {
        std::ostringstream str;
        ov::element::StructuralType::print(str, value);
        return str.str();
    }

    ov::Any value;

    static void copy (const Node::RTMap& src, Node::RTMap& dst);
    static bool has_type (const Node::RTMap& src, const ov::Any& type);
    static void move_to_original (Node::RTMap& rt_info);
    static ov::Any get (const Node::RTMap& src);
};

using ConstantVector = std::vector<std::shared_ptr<ov::opset1::Constant>>;

using StructuralTypeProxy::BindInput;

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

    StructuralTypeProxy::StructuralTypeMapAttribute(StructuralTypeProxy::StructuralTypeMapAttribute::get_input(node->get_rt_info())).set_output(node->get_rt_info());

    if(all_inputs_are_constants(node)) {
        // Evaluate mode
        std::cerr << "[ EVALUATION MODE ] " << node->get_type_info().name << "\n";
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
class StructuralTypedOp : public ov::op::Op {
public:
    OPENVINO_OP("StructuralTypedOp", "0", Op);

    StructuralTypedOp () : ov::op::Op() {}

    StructuralTypedOp (const OutputVector& arguments, const StructuralTypeProxy::BindInputs& bind_inputs = {}) : ov::op::Op(arguments)
    {
        if(!bind_inputs.empty())
            StructuralTypeProxy::StructuralTypeMapAttribute(bind_inputs).set_input(get_rt_info());
    }
};

// https://www.tensorflow.org/text/api_docs/python/text/case_fold_utf8
class CaseFoldUTF8 : public StructuralTypedOp {
public:
    OPENVINO_OP("CaseFoldUTF8", "0", StructuralTypedOp);

    CaseFoldUTF8(const OutputVector& arguments, const StructuralTypeProxy::BindInputs& bind_inputs = {}) : StructuralTypedOp(arguments, bind_inputs) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        return validate_and_infer_types_helper_1_to_1(this);
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        std::cerr << "[ CLONING ] CaseFoldUTF8\n";
        return std::make_shared<CaseFoldUTF8>(inputs, StructuralTypeProxy::StructuralTypeMapAttribute::get_input(get_rt_info()));
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
};


class NormalizeUTF8 : public StructuralTypedOp {
public:
    OPENVINO_OP("NormalizeUTF8", "0", StructuralTypedOp);

    NormalizeUTF8(
        const OutputVector& arguments,
        const std::string& normalization_form,
        const StructuralTypeProxy::BindInputs& bind_inputs = {})
    : StructuralTypedOp(arguments, bind_inputs),
        m_normalization_form(normalization_form)
    {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        return validate_and_infer_types_helper_1_to_1(this);
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        return std::make_shared<NormalizeUTF8>(inputs, m_normalization_form, StructuralTypeProxy::StructuralTypeMapAttribute::get_input(get_rt_info()));
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

private:

    std::string m_normalization_form;
};


class StaticRegexReplace : public StructuralTypedOp {
public:
    OPENVINO_OP("StaticRegexReplace", "0", StructuralTypedOp);

    StaticRegexReplace(
        const OutputVector& arguments,
        const std::string& pattern,
        const std::string& rewrite,
        const StructuralTypeProxy::BindInputs& bind_inputs = {})
    : StructuralTypedOp(arguments, bind_inputs), m_pattern(pattern), m_rewrite(rewrite) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        return validate_and_infer_types_helper_1_to_1(this);
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        //std::cerr << "[ CLONING ] StaticRegexReplace\n";
        return std::make_shared<StaticRegexReplace>(inputs, m_pattern, m_rewrite, StructuralTypeProxy::StructuralTypeMapAttribute::get_input(get_rt_info()));
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

    std::string m_pattern;
    std::string m_rewrite;
};


class RegexSplitWithOffsets : public StructuralTypedOp {
public:
    OPENVINO_OP("RegexSplitWithOffsets", "0", StructuralTypedOp);

    RegexSplitWithOffsets (const OutputVector& inputs, const StructuralTypeProxy::BindInputs& bind_inputs = {}) : StructuralTypedOp(inputs, bind_inputs) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {

        //if(get_input_size() == 3) {
        auto bind_inputs = StructuralTypeProxy::StructuralTypeMapAttribute::get_input(get_rt_info());
        if(bind_inputs.empty()) {
            if(get_input_size() != 3) {
                std::cerr << "Expect 3 inputs in structural_type processing flow\n";
                throw "Error";
            }
            // Real type infer
            // TODO: Not always real types, it can be a scalar string on the input, it is represented as a single u8 tensor
            set_output_type(0, element::dynamic, get_input_partial_shape(0));
            get_output_tensor(0).get_rt_info()["structural_type"] =
                StructuralTypeAttribute(element::StructuralType::Ragged(element::StructuralType::Str()));
        //} else if (get_input_size() == 5) {
        } else if(bind_inputs.size() == 3) {
            // Code output ragged[str] tensor
            set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));  // ragged slices begins
            set_output_type(1, get_input_element_type(0), get_input_partial_shape(0));  // ragged slices ends
            set_output_type(2, get_input_element_type(0), PartialShape({Dimension()}));  // str begins
            set_output_type(3, get_input_element_type(0), PartialShape({Dimension()}));  // str ends
            set_output_type(4, get_input_element_type(2), PartialShape({Dimension()}));  // str chars

            StructuralTypeProxy::BindInputs bind_outputs{{0, 5, element::StructuralType::Ragged(element::StructuralType::Str())}};
            StructuralTypeProxy::StructuralTypeMapAttribute(bind_outputs).set_output(get_rt_info());

            if(all_inputs_are_constants(this)) {
                // Evaluate mode
                std::cerr << "[ EVALUATION MODE ] RegexSplitWithOffsets\n";
                ov::TensorVector inputs;
                ConstantVector outputs;
                // FIXME: remove this part when CPU fixes evaluate with internally dynamic operations
                for(size_t i = 0; i < get_input_size(); ++i) {
                    auto constant = std::dynamic_pointer_cast<ov::opset1::Constant>(get_input_node_shared_ptr(i));
                    inputs.push_back(Tensor(constant->get_element_type(), constant->get_shape(), const_cast<void*>(constant->get_data_ptr())));
                }
                outputs = evaluate_internal_helper(inputs);

                for(size_t i = 0; i < get_output_size(); ++i) {
                    set_output_type(i, outputs[i]->get_element_type(), outputs[i]->get_shape());
                }
            }
        }
    }

    ConstantVector evaluate_internal_helper(const ov::TensorVector& inputs) const {
        auto input = StructuralTypeProxy::TensorStr<const ov::TensorVector::value_type*>(&inputs[0], &inputs[1], &inputs[2]);

        // We are supporting 1D inputs only, therefore we are making 2D ragged tensor
        size_t regular_size = input.get_shape()[0];

        // Parts of ragged representation
        std::vector<int> new_ragged_begins(regular_size), new_ragged_ends(regular_size);
        std::vector<int> new_str_begins, new_str_ends;
        std::string new_chars;

        for(size_t i = 0; i < regular_size; ++i) {
            // Stub: synthetic processing
            std::string value = input.element_by_offset(i);
            size_t part_size = 1;
            size_t skip_size = 1;
            size_t string_begin = 0;
            assert(new_str_begins.size() == new_str_ends.size());
            new_ragged_begins.push_back(new_str_begins.size());
            while(string_begin < value.length()) {
                std::string part = value.substr(string_begin, part_size);
                new_str_begins.push_back(new_chars.length());
                new_chars += part;
                new_str_ends.push_back(new_chars.length());
                string_begin += part_size + skip_size;
                skip_size = 1 - skip_size;
                ++part_size;
                //std::cerr << "part: " << part << std::endl;
            }
            assert(new_str_begins.size() == new_str_ends.size());
            new_ragged_ends.push_back(new_str_begins.size());
            assert(new_ragged_begins.size() == new_ragged_ends.size());
        }

        ConstantVector results;
        results.push_back(make_shared<ov::opset1::Constant>(element::i32, Shape{regular_size}, &new_ragged_begins[0]));
        results.push_back(make_shared<ov::opset1::Constant>(element::i32, Shape{regular_size}, &new_ragged_ends[0]));
        results.push_back(make_shared<ov::opset1::Constant>(element::i32, Shape{new_str_begins.size()}, &new_str_begins[0]));
        results.push_back(make_shared<ov::opset1::Constant>(element::i32, Shape{new_str_ends.size()}, &new_str_ends[0]));
        results.push_back(make_shared<ov::opset1::Constant>(element::u8, Shape{new_chars.length()}, new_chars.data()));

        return results;
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        //std::cerr << "[ CLONING ] StaticRegexReplace\n";
        return std::make_shared<RegexSplitWithOffsets>(inputs, StructuralTypeProxy::StructuralTypeMapAttribute::get_input(get_rt_info()));
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
        auto results = evaluate_internal_helper(inputs);
        for(size_t i = 0; i < results.size(); ++i) {
            size_t length = results[i]->get_byte_size();
            std::cerr << "Expected length " << length << ", allocated: " << outputs[i].get_shape() << "\n";
            memcpy(outputs[i].data(), results[i]->get_data_ptr(), length);
        }
        return true;
    }

    bool has_evaluate() const {
        return true;
    }
};

class WordpieceTokenizeWithOffsets : public StructuralTypedOp {
public:
    OPENVINO_OP("WordpieceTokenizeWithOffsets", "0", StructuralTypedOp);

    WordpieceTokenizeWithOffsets (const OutputVector& inputs, const StructuralTypeProxy::BindInputs& bind_inputs = {}) : StructuralTypedOp(inputs, bind_inputs) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {

        std::cerr << "WordpieceTokenizeWithOffsets: -1\n";

        auto bind_inputs = StructuralTypeProxy::StructuralTypeMapAttribute::get_input(get_rt_info());
        if(bind_inputs.empty()) {
            std::cerr << "WordpieceTokenizeWithOffsets: 0\n";
            if(get_input_size() != 2) {
                std::cerr << "Expect 2 inputs in structural_type processing flow\n";
                throw "Error";
            }
            // Real type infer
            set_output_type(0, element::dynamic, get_input_partial_shape(0));
            auto st = StructuralTypeAttribute::get(get_input_tensor(0).get_rt_info());
            get_output_tensor(0).get_rt_info()["structural_type"] =
                StructuralTypeAttribute(element::StructuralType::Ragged(st));
        } else {
            std::cerr << "WordpieceTokenizeWithOffsets: 1\n";
            // Code output ragged[str] tensor, the same as bind_input[0]
            const auto& indices = bind_inputs[0].inputs;
            std::vector<size_t> new_indices;

            for(size_t i = 0; i < indices.size() - 1; ++i) {
                auto index = indices[i];
                set_output_type(i, get_input_element_type(index), get_input_partial_shape(index));
                new_indices.push_back(i);
            }

            for(size_t i = indices.size() - 2 - 1; i < indices.size(); ++i) {
                auto index = indices[i];
                set_output_type(i + 2, get_input_element_type(index), get_input_partial_shape(index));
                new_indices.push_back(i + 2);
            }

            StructuralTypeProxy::BindInputs bind_outputs{BindInput(new_indices, element::StructuralType::Ragged(bind_inputs[0].structural_type))};   // TODO: Add offsets
            StructuralTypeProxy::StructuralTypeMapAttribute(bind_outputs).set_output(get_rt_info());

            if(all_inputs_are_constants(this)) {
                // Evaluate mode
                std::cerr << "[ EVALUATION MODE ] WordpieceTokenizeWithOffsets\n";
                ov::TensorVector inputs;
                ConstantVector outputs;
                // FIXME: remove this part when CPU fixes evaluate with internally dynamic operations
                for(size_t i = 0; i < get_input_size(); ++i) {
                    auto constant = std::dynamic_pointer_cast<ov::opset1::Constant>(get_input_node_shared_ptr(i));
                    inputs.push_back(Tensor(constant->get_element_type(), constant->get_shape(), const_cast<void*>(constant->get_data_ptr())));
                }
                outputs = evaluate_internal_helper(inputs);

                for(size_t i = 0; i < get_output_size(); ++i) {
                    set_output_type(i, outputs[i]->get_element_type(), outputs[i]->get_shape());
                }
            }
        }
    }

    ConstantVector evaluate_internal_helper(const ov::TensorVector& inputs) const {
        ConstantVector results;
        // We always adding one new ragged dimension to input tensor
        // Each ragged dimension is represented by a pair of index tensors
        // We suppose Ragged Tensor or regular tensor of string in the input,
        // so it is represented as the 2*n + 2 + 1 tensors, where n is the number of input ragged dimensions
        // 2 is indices for str begins and end, and 1 is for symbols

        for(int i = 3; i < inputs.size() - 2 - 1; ++i) {
            results.push_back(make_shared<ov::opset1::Constant>(
                inputs[i].get_element_type(),
                inputs[i].get_shape(),
                inputs[i].data(inputs[i].get_element_type())));
        }

        // As this is a stub, so introduced ragged dimension contains just 1 element in size

        auto size = inputs[inputs.size() - 2].get_shape()[0];
        std::vector<int> passthrough_indices(size + 1);
        for(size_t i = 0; i < size + 1; ++i) {
            passthrough_indices[i] = static_cast<int>(i);
        }

        results.push_back(make_shared<ov::opset1::Constant>(element::i32, Shape{size}, &passthrough_indices[0]));
        results.push_back(make_shared<ov::opset1::Constant>(element::i32, Shape{size}, &passthrough_indices[1]));

        for(int i = inputs.size() - 2 - 1; i < inputs.size(); ++i) {
            results.push_back(make_shared<ov::opset1::Constant>(
                inputs[i].get_element_type(),
                inputs[i].get_shape(),
                inputs[i].data(inputs[i].get_element_type())));
        }

        return results;
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        return std::make_shared<WordpieceTokenizeWithOffsets>(inputs, StructuralTypeProxy::StructuralTypeMapAttribute::get_input(get_rt_info()));
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
        auto results = evaluate_internal_helper(inputs);
        for(size_t i = 0; i < results.size(); ++i) {
            size_t length = results[i]->get_byte_size();
            std::cerr << "Expected length " << length << ", allocated: " << outputs[i].get_shape() << "\n";
            memcpy(outputs[i].data(), results[i]->get_data_ptr(), length);
        }
        return true;
    }

    bool has_evaluate() const {
        return true;
    }
};



inline Any replace_stuctural_type_recursively (Any st, Any target, Any replacement) {
    using namespace element::StructuralType;
    if (st == target) {
        return replacement;
    } else if (st.is<element::StructuralType::Tensor>()) {
        return element::StructuralType::Tensor(replace_stuctural_type_recursively(st.as<element::StructuralType::Tensor>().element_type, target, replacement));
    } else if (st.is<element::StructuralType::List>()) {
        return element::StructuralType::List(replace_stuctural_type_recursively(st.as<element::StructuralType::List>().element_type, target, replacement));
    } else if (st.is<element::StructuralType::Ragged>()) {
        return element::StructuralType::Ragged(replace_stuctural_type_recursively(st.as<element::StructuralType::Ragged>().element_type, target, replacement));
    }
    return st;
}


class LookupTableFindV2 : public StructuralTypedOp {
public:
    OPENVINO_OP("LookupTableFindV2", "0", StructuralTypedOp);

    LookupTableFindV2 (const OutputVector& inputs, const StructuralTypeProxy::BindInputs& bind_inputs = {}) : StructuralTypedOp(inputs, bind_inputs) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {

        std::cerr << "LookupTableFindV2: -1\n";

        auto bind_inputs = StructuralTypeProxy::StructuralTypeMapAttribute::get_input(get_rt_info());
        if(bind_inputs.empty()) {
            std::cerr << "LookupTableFindV2: 0\n";
            if(get_input_size() < 2) {
                std::cerr << "Expect at least 2 inputs in structural_type processing flow\n";
                throw "Error";
            }
            // Real type infer
            set_output_type(0, element::dynamic, get_input_partial_shape(0));
            auto st = StructuralTypeAttribute::get(get_input_tensor(1).get_rt_info());
            auto new_st = replace_stuctural_type_recursively(st, element::StructuralType::Str(), element::i32);    // FIXME: i32 is a stub, should be read from input 0
            get_output_tensor(0).get_rt_info()["structural_type"] =
                StructuralTypeAttribute(new_st);
        } else {
            std::cerr << "LookupTableFindV2: 1\n";
            // Produce ragged*[int] as an input -- the same as input except 2 str index dimensions near the end
            const auto& indices = bind_inputs[1].inputs;
            std::vector<size_t> new_indices;
            std::cerr << "LookupTableFindV2: 2\n";

            for(size_t i = 0; i < indices.size() - 3; ++i) {    // all ragged components except the last 2 + 1 tensors for str indices and symbols
                auto index = indices[i];    // suppose order is predefined and everytime ordered (TODO)
                set_output_type(i, get_input_element_type(index), get_input_partial_shape(index));
                new_indices.push_back(i);
            }
            std::cerr << "LookupTableFindV2: 3\n";

            set_output_type(indices.size() - 3, element::i32, get_input_partial_shape(indices.size() - 2)); // indices.size() - 2 is dimension for str indices, it defines final shape for i32 output
            new_indices.push_back(indices.size() - 3);
            std::cerr << "LookupTableFindV2: 4\n";

            StructuralTypeProxy::BindInputs bind_outputs{BindInput(new_indices, replace_stuctural_type_recursively(bind_inputs[1].structural_type, element::StructuralType::Str(), element::i32))};
            StructuralTypeProxy::StructuralTypeMapAttribute(bind_outputs).set_output(get_rt_info());

            std::cerr << "LookupTableFindV2: 5\n";
            if(all_inputs_are_constants(this)) {
                // Evaluate mode
                std::cerr << "[ EVALUATION MODE ] LookupTableFindV2\n";
                ov::TensorVector inputs;
                ConstantVector outputs;
                // FIXME: remove this part when CPU fixes evaluate with internally dynamic operations
                for(size_t i = 0; i < get_input_size(); ++i) {
                    auto constant = std::dynamic_pointer_cast<ov::opset1::Constant>(get_input_node_shared_ptr(i));
                    inputs.push_back(Tensor(constant->get_element_type(), constant->get_shape(), const_cast<void*>(constant->get_data_ptr())));
                }
                outputs = evaluate_internal_helper(inputs);

                for(size_t i = 0; i < get_output_size(); ++i) {
                    set_output_type(i, outputs[i]->get_element_type(), outputs[i]->get_shape());
                }
            }
            std::cerr << "LookupTableFindV2: 6\n";
        }
    }

    ConstantVector evaluate_internal_helper(const ov::TensorVector& inputs) const {
        ConstantVector results;
        // We always adding one new ragged dimension to input tensor
        // Each ragged dimension is represented by a pair of index tensors
        // We suppose Ragged Tensor or regular tensor of string in the input,
        // so it is represented as the 2*n + 2 + 1 tensors, where n is the number of input ragged dimensions
        // 2 is indices for str begins and end, and 1 is for symbols

        for(int i = 3; i < inputs.size() - 2 - 1 - 1; ++i) {
            results.push_back(make_shared<ov::opset1::Constant>(
                inputs[i].get_element_type(),
                inputs[i].get_shape(),
                inputs[i].data(inputs[i].get_element_type())));
        }

        // Just ignore two index tensors for string, but other (outer ragged dimensions) are kept the same
        // FIXME: generating range of numbers for now regardless of the original content

        auto size = inputs[inputs.size() - 2].get_shape()[0];
        std::vector<int> range(size);
        for(size_t i = 0; i < size; ++i) {
            range[i] = static_cast<int>(i);
        }

        results.push_back(make_shared<ov::opset1::Constant>(element::i32, Shape{size}, &range[0]));

        return results;
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        return std::make_shared<LookupTableFindV2>(inputs, StructuralTypeProxy::StructuralTypeMapAttribute::get_input(get_rt_info()));
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
        auto results = evaluate_internal_helper(inputs);
        std::cerr << "[ INFO ] Required outputs: " << outputs.size() << "\n";
        std::cerr << "[ INFO ] Proposed outputs: " << results.size() << "\n";
        for(size_t i = 0; i < results.size(); ++i) {
            size_t length = results[i]->get_byte_size();
            std::cerr << "Expected length " << length << ", allocated: " << outputs[i].get_shape() << "\n";
            memcpy(outputs[i].data(), results[i]->get_data_ptr(), length);
        }
        return true;
    }

    bool has_evaluate() const {
        return true;
    }
};


class StructPack : public ov::op::Op {
public:
    OPENVINO_OP("INTERNAL::StructPack");

    StructPack(const OutputVector& arguments, Any res_type, const PartialShape& res_shape)
        : ov::op::Op(arguments), m_res_type(res_type), m_res_shape(res_shape) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        set_output_type(0, element::dynamic, m_res_shape);
        get_output_tensor(0).get_rt_info()["structural_type"] = StructuralTypeAttribute(m_res_type);
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        return make_shared<StructPack>(inputs, m_res_type, m_res_shape);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        // FIXME: Serialization only, there is no deserialization
        std::string m_res_type_str = m_res_type->to_string();
        visitor.on_attribute("res_type", m_res_type_str);
        visitor.on_attribute("res_shape", m_res_shape);
        return true;
    }

    bool has_evaluate() const {
        return false;
    }

    Any m_res_type;
    PartialShape m_res_shape;
};


}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
