// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include "openvino/frontend/tensorflow/frontend.hpp"
#include "helper_ops/internal_operation.hpp"
#include "openvino/frontend/tensorflow/decoder.hpp"
#include "openvino/opsets/opset1.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {


// https://www.tensorflow.org/text/api_docs/python/text/case_fold_utf8
class CaseFoldUTF8 : public ov::op::Op {
public:
    OPENVINO_OP("CaseFoldUTF8");

    CaseFoldUTF8(const OutputVector& arguments) : ov::op::Op(arguments) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        std::cerr << "+++++++++++ CaseFoldUTF8 +++++++++++\n";
        std::cerr << input_value(0).get_node_shared_ptr() << "\n";
        if(std::dynamic_pointer_cast<ov::opset1::Constant>(input_value(0).get_node_shared_ptr())) {
            set_output_type(0, get_input_element_type(0), PartialShape{get_input_partial_shape(0)[0] + prefix.size()});
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
        // TODO: Implement real code instead of this placeholder

        if(inputs.size() == 1 && inputs[0].get_shape().size() == 1) {
            std::cerr << "[ EVALUATE ] CaseFoldUTF8\n";
            // Scalar str implementation represented as 1d/u8
            auto in = inputs[0];
            auto out = outputs[0];

            std::cerr << "in.get_shape() = " << in.get_shape() << "\n";
            std::cerr << "out.get_shape() = " << out.get_shape() << "\n";
            //auto out = HostTensor();
            out.set_shape(Shape{in.get_shape()[0] + prefix.size()});
            std::cerr << "Changed tensor shape to " << out.get_shape() << "\n";
            memcpy(out.data(), prefix.data(), prefix.size());
            memcpy((char*)out.data() + prefix.size(), in.data(), in.get_byte_size());
            return true;
        } else {
            // No implementation for the extended form of operation
            return false;
        }
    }

    bool has_evaluate() const {
        return true;
    }

    std::string prefix = "<Prefix from CaseFoldUTF8>";
};


class NormalizeUTF8 : public ov::op::Op {
public:
    OPENVINO_OP("NormalizeUTF8");

    NormalizeUTF8(const OutputVector& arguments) : ov::op::Op(arguments) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        std::cerr << "+++++++++++ NormalizeUTF8 +++++++++++\n";
        std::cerr << input_value(0).get_node_shared_ptr() << "\n";
        
        if(std::dynamic_pointer_cast<ov::opset1::Constant>(input_value(0).get_node_shared_ptr())) {
            set_output_type(0, get_input_element_type(0), PartialShape{get_input_partial_shape(0)[0] + suffix.size()});
            std::cerr << "Evaluate mode\n";
        } else {
            set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
        }
        
        StructuralTypeAttribute::copy(get_input_tensor(0).get_rt_info(), get_output_tensor(0).get_rt_info());
        std::cerr << "**********************\n";
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        std::cerr << "[ CLONING ] NormalizeUTF8\n";
        return std::make_shared<NormalizeUTF8>(inputs);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
        // TODO: Implement real code instead of this placeholder

        if(StructuralTypeAttribute::has_type(get_output_tensor(0).get_rt_info(), element::StructuralType::Str())) {
            std::cerr << "[ EVALUATE ] Detected hidden str type\n";
        } else {
            std::cerr << "[ EVALUATE ] [ ERROR ] Missing hiddent data type\n";
        }

        if(inputs.size() == 1 && inputs[0].get_shape().size() == 1) {

            // Scalar str implementation represented as 1d/u8

            auto in = inputs[0];
            auto out = outputs[0];

            std::cerr << "[ EVALUATE ] NormalizeUTF8\n";
            std::cerr << "out.get_shape() = " << out.get_shape() << "\n";
            std::cerr << "in.get_shape() = " << in.get_shape() << "\n";

            out.set_shape(Shape{in.get_shape()[0] + suffix.size()});
            std::cerr << "Changed tensor shape to " << out.get_shape() << "\n";
            memcpy(out.data(), in.data(), in.get_byte_size());
            memcpy((char*)out.data() + in.get_byte_size(), suffix.data(), suffix.size());
            return true;
        } else {
            // No implementation for the extended form
            return false;
        }
    }

    bool has_evaluate() const {
        return true;
    }

    std::string suffix = "<Suffix from NormalizeUTF8>";
};


}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
