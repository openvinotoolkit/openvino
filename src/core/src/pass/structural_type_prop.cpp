// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>
#include <numeric>
#include <fstream>
#include <string>

#include <openvino/op/str_ops.hpp>

#include <openvino/pass/structural_type_prop.hpp>

#include <ngraph/opsets/opset8.hpp>
#include <openvino/core/rt_info.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <openvino/pass/pattern/op/or.hpp>
#include <openvino/pass/manager.hpp>
#include <openvino/core/type/non_tensor_type.hpp>
#include <openvino/opsets/opset9.hpp>
#include <openvino/op/util/framework_node.hpp>

using std::make_shared;
using std::shared_ptr;

namespace {
    // TODO: Remove this duplicate: CPU transforms has it, copied and pasted here

    bool is_data_movement_operation(const std::shared_ptr<ov::Node>& node) {
        return ov::is_type<ov::op::v0::Squeeze>(node) ||
               ov::is_type<ov::op::v1::StridedSlice>(node) ||
               ov::is_type<ov::op::v0::Unsqueeze>(node) ||
               ov::is_type<ov::op::v1::Reshape>(node) ||
               ov::is_type<ov::op::v1::Transpose>(node) ||
               ov::is_type<ov::op::v0::ShuffleChannels>(node) ||
               ov::is_type<ov::op::v7::Roll>(node) ||
               ov::is_type<ov::op::v0::ReverseSequence>(node) ||
               ov::is_type<ov::op::v0::DepthToSpace>(node) ||
               ov::is_type<ov::op::v1::BatchToSpace>(node) ||
               ov::is_type<ov::op::v1::Broadcast>(node) ||
               ov::is_type<ov::op::v3::Broadcast>(node) ||
               ov::is_type<ov::op::v1::Gather>(node) ||
               ov::is_type<ov::op::v7::Gather>(node) ||
               ov::is_type<ov::op::v8::Gather>(node) ||
               ov::is_type<ov::op::v0::Parameter>(node);
    }

    bool is_str_operation(const std::shared_ptr<ov::Node>& node) {
        return ov::is_type<ov::frontend::tensorflow::CaseFoldUTF8>(node) ||
               ov::is_type<ov::frontend::tensorflow::NormalizeUTF8>(node) ||
               ov::is_type<ov::frontend::tensorflow::StaticRegexReplace>(node) ||
               ov::is_type<ov::frontend::tensorflow::WordpieceTokenizeWithOffsets>(node) ||
               ov::is_type<ov::frontend::tensorflow::LookupTableFindV2>(node) ||
               ov::is_type<ov::frontend::tensorflow::RegexSplitWithOffsets>(node);
    }

    bool is_scalar_like(const std::shared_ptr<ov::Node>& node) {
        auto constantNode = std::dynamic_pointer_cast<ov::opset9::Constant>(node);
        return constantNode != nullptr && shape_size(constantNode->get_shape()) == 1;
    }

    std::shared_ptr<ov::op::util::FrameworkNode> as_tf_op_type(
        const std::shared_ptr<ov::Node>& node,
        const std::string& tf_op_type)
    {
        auto fw_node = std::dynamic_pointer_cast<ov::op::util::FrameworkNode>(node);
        if(!fw_node) {
            return nullptr;
        }
        const auto& info = node->get_rt_info();
        auto entry = info.find("tf_orig_type");
        if(entry != info.end() &&
                entry->second.is<std::string>() &&
                entry->second.as<std::string>() == tf_op_type)
        {
            return fw_node;
        }
        return nullptr;
    }
} // namespace

namespace ov {
namespace frontend {
namespace tensorflow {


using namespace ov;
using namespace ov::frontend::tensorflow;

void StructuralTypeAttribute::copy (const Node::RTMap& src, Node::RTMap& dst) {
    Any st = get(src);
    if(!st.empty()) {
        dst["structural_type"] = StructuralTypeAttribute(st);
    }
}


ov::Any StructuralTypeAttribute::get (const Node::RTMap& src) {
    auto pstructural_type = src.find("structural_type");
    if(pstructural_type != src.end()) {
        return pstructural_type->second.as<StructuralTypeAttribute>().value;
    } else {
        return Any();
    }
}


bool StructuralTypeAttribute::has_type (const Node::RTMap& rt_info, const ov::Any& type) {
    Any st = get(rt_info);
    return !st.empty() && type == st;
}


void StructuralTypeAttribute::move_to_original (Node::RTMap& rt_info) {
    auto pstructural_type = rt_info.find("structural_type");
    if(pstructural_type != rt_info.end()) {
        rt_info["orig_structural_type"] = pstructural_type->second;
        rt_info.erase(pstructural_type);
    }
}


namespace pass {

StructuralTypeProp::StructuralTypeProp() {
    //auto data_movement = ov::pattern::wrap_type<ov::op::Op>(
    //    ov::pass::pattern::op::as_value_predicate(is_data_movement_operation));
    auto data_movement = ov::pass::pattern::wrap_type<ov::op::Op>(
        ov::pass::pattern::op::as_value_predicate([] (std::shared_ptr<Node>) -> bool { return true; }));
    std::cerr << "[ INFO TF FE ] Registering StructuralTypeProp\n";

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto node = m.get_match_root();

        std::cerr << "[ INFO TF FE ] Matching prop: " << node->get_type_name() << "\n";

        // Depending on operation, propagate structural type field
        // TODO: This code should be moved to the operations themselves, but now we are trying
        // to avoid any impact on OV structures and implement it externally.
        // Code amount required to implement it in core will be similar to what we are doing
        // here except we won't have similar mega-switches based on op types.

        if (auto parameter = std::dynamic_pointer_cast<ov::op::v0::Parameter>(node)) {
            // Parameter should have a special RT info attribute `structural_type` that should be copied
            // to the output tensor rt_info

            std::cerr << "[ INFO TF FE ] Detected Parameter\n";

            StructuralTypeAttribute::copy(parameter->get_rt_info(), parameter->get_output_tensor(0).get_rt_info());
        } else if (auto reshape = std::dynamic_pointer_cast<ov::op::v1::Reshape>(node)) {
            std::cerr << "[ INFO TF FE ] Detected Reshape\n";
            StructuralTypeAttribute::copy(reshape->get_input_tensor(0).get_rt_info(), reshape->get_output_tensor(0).get_rt_info());
        } else {
            // Make usual propagation by op design
            node->validate_and_infer_types();
        }

        return false;
    };

    auto m = make_shared<ov::pass::pattern::Matcher>(data_movement, "ov::frontend::tensorflow::pass::StructuralTypeProp");
    register_matcher(m, callback);
}


ReplaceStrByU81D::ReplaceStrByU81D() {
    auto str_tensor = ov::pass::pattern::wrap_type<ov::op::Op>(
        ov::pass::pattern::op::ValuePredicate([](ov::Output<ov::Node> x) {
            std::cerr << "get_rt_info: " << x.get_tensor().get_rt_info().size() << "\n";
            //return false;
            std::cerr.flush();
            auto rank = x.get_tensor().get_partial_shape().rank();
            std::cerr << "[ REPLACE STR BY U81D ] RANK IS " << (rank.is_static() && rank.get_length() == 0) << "\n";
            return rank.is_static() && rank.get_length() == 0 &&
                StructuralTypeAttribute::has_type(x.get_tensor().get_rt_info(), element::StructuralType::Str());
            // FIXME: Check that this is a scalar, otherwise this transformation doesn't work
            // FIXME: For now we re-interpret all tensors that have Str type as a scalar tensors
        }));

    std::cerr << "[ INFO TF FE ] Registering ReplaceStrByU81D\n";

    auto callback = [](ov::pass::pattern::Matcher& m) {
        //return false;
        auto port = m.get_match_value();  // TODO: iterate over all outputs and check each of it to match the criteria
        auto node = m.get_match_root();

        std::cerr << "[ INFO TF FE ] Detected tensor with Str type: " << node->get_type_name() << "\n";

        if (auto parameter = std::dynamic_pointer_cast<ov::op::v0::Parameter>(node)) {
            std::cerr << "Parameter override to u8/1d\n";
            parameter->set_element_type(element::u8);
            parameter->set_partial_shape(PartialShape{Dimension()});
        }

        // Just setting type and shape without shape propagation -- will require re-validation of the function
        // in the end to catch all inconsistencies due to possible bugs.

        port.get_tensor().set_tensor_type(element::u8, PartialShape{Dimension()});
        //std::cerr << "move to original\n";
        //StructuralTypeAttribute::move_to_original(port.get_tensor().get_rt_info());
        return false;
    };

    auto m = make_shared<ov::pass::pattern::Matcher>(str_tensor, "ov::frontend::tensorflow::pass::ReplaceStrByU81D");
    register_matcher(m, callback);
}


bool DecomposeStrParameters::run_on_model(const std::shared_ptr<Model>& model) {
    // Search for Parameter with List[Tensor] types

    ParameterVector parameters = model->get_parameters();
    ParameterVector new_parameters;  // collect decomposed parameters

    StructuralTypeProxy::BindInputs bind_inputs;
    // The following is obsolete:
    // In the main loop below, indices are collected relatively to the current number of parameters.
    // The final number will change depending on how many parameters are decomposed, because
    // each decomposed parameter is deleted from the set of parameters.
    // So in the end after the main loop there will be a correction loop over all indices
    // to adjust indices values to make them correct relative to the really left number of original parameters.

    for (size_t i = 0; i < parameters.size(); ++i) {
        auto parameter = parameters[i];
        std::cerr << "[ PARAMETER ] " << i << ": " << parameter << "\n";

        /////////////////// VOCAB READING HACK //////////////////

        std::vector<Input<Node>> relevant;

        // Check consumers, find for WordpieceTokenize
        auto consumers = parameter->get_output_target_inputs(0);
        for(std::set<Input<Node>>::iterator i = consumers.begin(); i != consumers.end(); ++i) {
            std::cerr << "    " << i->get_node() << "\n";
            if(dynamic_cast<const WordpieceTokenizeWithOffsets*>(i->get_node()) || dynamic_cast<const LookupTableFindV2*>(i->get_node())) {
                relevant.push_back(*i);
            }
        }

        if(relevant.size() == consumers.size()) {
            std::cerr << "    All consumers allow Vocab replacement HACK\n";
            // FIXME: should get path from the model, now it is hard-coded
            std::string vocab_file_path = "/home/slyalin/jupyter/openvino-main/tmpa38yfctx_saved_model/assets/vocab.txt";
            std::ifstream vocab_file(vocab_file_path);
            std::string line;
            std::vector<int32_t> indices;
            std::string symbols;
            while(std::getline(vocab_file, line)) {
                indices.push_back(symbols.length());
                //std::cerr << line << ' ';
                symbols += line;
            }
            indices.push_back(symbols.length());

            OutputVector results;
            results.push_back(make_shared<ov::opset1::Constant>(element::i32, Shape{indices.size() - 1}, &indices[0]));
            results.push_back(make_shared<ov::opset1::Constant>(element::i32, Shape{indices.size() - 1}, &indices[1]));
            results.push_back(make_shared<ov::opset1::Constant>(element::u8, Shape{symbols.length()}, symbols.data()));

            auto struct_pack = make_shared<StructPack>(
                results,
                element::StructuralType::Str(),
                PartialShape{indices.size() - 1}
            );

            replace_node(parameter, struct_pack);
            model->remove_parameter({parameter});
            continue;
        }

        /////////////////////////////////////////////////////////

        // Check 1D and Str structural type
        auto rank = parameter->get_partial_shape().rank();
        auto rt_info = parameter->get_rt_info();
        rt_info["structural_type"] = StructuralTypeAttribute(element::StructuralType::Str());
        //parameter->set_element_type(element::dynamic);
        //parameter->set_partial_shape(PartialShape{Dimension()});
        parameter->validate_and_infer_types();
        if(
            parameter->get_element_type() == element::dynamic &&
            StructuralTypeAttribute::has_type(rt_info, element::StructuralType::Str()))
            // TODO: Also should capture Tensor(Str), not only Str
        {
            if(rank.is_static() && rank.get_length() == 1) {
                std::cerr << "Applying decomposition for parameter: " << parameter->get_name() << "\n";
                OutputVector inputs_for_struct_pack;

                bind_inputs.push_back(StructuralTypeProxy::BindInput(
                    {new_parameters.size(), new_parameters.size() + 1, new_parameters.size() + 2},
                    element::StructuralType::Tensor(element::StructuralType::Str())));

                // for individual strings start and end indices
                for (size_t i = 0; i < 2; ++i) {
                    auto new_parameter =
                        make_shared<opset9::Parameter>(element::i32, parameter->get_partial_shape());
                    new_parameters.push_back(new_parameter);
                    inputs_for_struct_pack.push_back(new_parameter);
                    // TODO: add links via RT info between original parameter and new ones
                }

                // for tensor elements
                auto new_parameter =
                    // using u8 here because we know that we are dealing with strings
                    make_shared<opset9::Parameter>(/*element::dynamic*/element::u8, PartialShape{Dimension()});
                new_parameters.push_back(new_parameter);
                inputs_for_struct_pack.push_back(new_parameter);

                auto struct_pack = make_shared<StructPack>(
                    inputs_for_struct_pack,
                    element::StructuralType::Str(), // parameter->get_rt_info()["structural_type"].as<StructuralTypeAttribute>().value
                    parameter->get_partial_shape()
                );

                replace_node(parameter, struct_pack);
                model->remove_parameter({parameter});
                continue;
            } else if(rank.is_static() && rank.get_length() == 0) {
                std::cerr << "Parameter override to u8/1d\n";
                bind_inputs.push_back(StructuralTypeProxy::BindInput(
                    {new_parameters.size()},
                    element::StructuralType::Tensor(element::StructuralType::Str())));
                auto new_parameter = make_shared<opset9::Parameter>(element::u8, PartialShape{Dimension()});
                auto struct_pack = make_shared<StructPack>(
                    new_parameter->outputs(),
                    element::StructuralType::Str(), // parameter->get_rt_info()["structural_type"].as<StructuralTypeAttribute>().value
                    parameter->get_partial_shape()
                );
                new_parameters.push_back(new_parameter);
                replace_node(parameter, struct_pack);
                model->remove_parameter({parameter});
                continue;
            }
        }/* else if (parameter->get_element_type() == element::dynamic && rt_info.find("structural_type") == rt_info.end()) {
            std::cerr << "[ INFO ] Dynamically typed parameter without structural type, assume this is the resource with vocab.\n";
            // FIXME: this is a hack to load vocab.txt without real presence of necessary nodes in graph that helps to identify the vocab
            // TODO: Remove this part when necessary part of a model starts to appear in the graph and we can handle it in a normal way
        }*/

        // Handly untouched parameters in the same way as new parameters to make index adjustment easier
        // and maintain the same order of parameter groups (straight and decomposed)
        Any element_type = StructuralTypeAttribute::get(rt_info);
        if(element_type.empty()) {
            element_type = parameter->get_element_type();
        }
        bind_inputs.push_back(StructuralTypeProxy::BindInput({new_parameters.size()}, element::StructuralType::Tensor(element_type)));
        new_parameters.push_back(parameter);
        model->remove_parameter({parameter});
    }

    model->add_parameters(new_parameters);
    StructuralTypeProxy::StructuralTypeMapAttribute(bind_inputs).set_input(model->get_rt_info());
    return true;
}


OutputVector get_inputs (std::shared_ptr<Node> node) {
    OutputVector result;
    for(size_t i = 0; i < node->get_input_size(); ++i) {
        result.push_back(node->get_input_source_output(i));
    }
    return result;
}

using ov::pass::pattern::wrap_type;
using ov::pass::pattern::op::as_value_predicate;
using ov::pass::pattern::any_input;
using StructuralTypeProxy::BindInputs;

/*
using ov::opset9::Constant;

template <typename T>
shared_ptr<Constant> const_value (const T& value, size_t rank = 0, element::Type et = element::i32) {
    return make_shared<Constant>(et, Shape(rank, 1), value);
}
*/

ThroughStrOpsProp::ThroughStrOpsProp() {
    //auto input = wrap_type<StructPack>();
    auto node = wrap_type<ov::op::Op>(as_value_predicate(is_str_operation));

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto node = m.get_match_root();
        std::cerr << "[ INFO TF FE ] Matching str op: " << node->get_type_name() << "\n";

        // Replace each input that consumes StructPack with decomposed tensors
        // Insert StructPack for each output that has type str

        // Inputs
        const auto& inputs = node->inputs();
        OutputVector new_inputs;
        StructuralTypeProxy::BindInputs bind_inputs;
        bool at_least_one = false;
        for(size_t i = 0; i < inputs.size(); ++i) {
            auto input = inputs[i];
            if(ov::is_type<StructPack>(input.get_source_output().get_node_shared_ptr())) {
                Any st = StructuralTypeAttribute::get(input.get_tensor().get_rt_info());
                if(st.empty()) {
                    std::cerr << "[ ERROR ] StructPack produces value without stuctural_type attached\n";
                }
                std::cerr << "[ 1 ]\n";
                auto input_inputs = get_inputs(inputs[i].get_source_output().get_node_shared_ptr());
                bind_inputs.push_back({new_inputs.size(), new_inputs.size() + input_inputs.size(), st});
                new_inputs.insert(new_inputs.end(), input_inputs.begin(), input_inputs.end());
                at_least_one = true;
            } else {
                std::cerr << "[ 2 ]\n";
                bind_inputs.push_back({{new_inputs.size()}, element::StructuralType::Tensor(input.get_element_type())});
                new_inputs.push_back(input.get_source_output());
            }
        }

        if(!at_least_one) {
                std::cerr << "[ 3 ]\n";
            return false;
        }

        std::cerr << "[ 4 ]\n";

        // Set the property in old node to let it flow to the new node in clone_with_new_inputs
        // Need to do that because the property should be in the node from the moment of construction
        // to let validate_and_infer_types to take it into account.
        StructuralTypeProxy::StructuralTypeMapAttribute(bind_inputs).set_input(node->get_rt_info());

        auto new_node = node->clone_with_new_inputs(new_inputs);
        // new_node should have an extended set of outputs due to lowering semantics
        // inside the node; so the operation behind the node should expect lowered input set
        // and react accordingly by providing lowered outputs

        // Outputs
        const auto& outputs = node->outputs();
        OutputVector new_outputs;

        BindInputs bind_outputs = StructuralTypeProxy::StructuralTypeMapAttribute::get_output(new_node->get_rt_info());
        at_least_one = false;

        for(size_t i = 0; i < bind_outputs.size(); ++i) {
            // For each group of output create StuctPack
            const auto& indices = bind_outputs[i].inputs;
            Any st = bind_outputs[i].structural_type;
            PartialShape ps = node->output(i).get_partial_shape();
            //if(st != StructuralTypeAttribute::get(node->output(i).get_tensor().get_rt_info())) {
            //    std::cerr << "[ ERROR ] Strcutural types for old node and lowered nodes do not match\n";
            //    return false;
            //}

            if(st.is<element::StructuralType::Tensor>()) {
                if(indices.size() != 1) {
                    std::cerr << "[ ERROR ] Tensor has more that 1 packets tensors.\n";
                    return false;
                }

                new_outputs.push_back(outputs[indices[0]]);
            } else {
                OutputVector inputs;
                for(size_t j = 0; j < indices.size(); ++j)
                    inputs.push_back(new_node->output(indices[j]));

                new_outputs.push_back(make_shared<StructPack>(inputs, st, ps));
                at_least_one = true;
            }
        }

        if(bind_outputs.empty()) {
            replace_node(node, new_node);
        } else {
            replace_node(node, new_outputs);
        }
        return true;
    };

    auto m = make_shared<ov::pass::pattern::Matcher>(node, "ov::frontend::tensorflow::pass::ThroughStrOpsProp");
    register_matcher(m, callback);
}


ThroughReshapeProp::ThroughReshapeProp() {
    // Should better match node that has at least one StructPack at least at one inputs
    auto input = wrap_type<StructPack>();
    auto node = make_shared<ov::pass::pattern::op::Or>(OutputVector{
        ov::pass::pattern::wrap_type<ov::opset9::Reshape>(OutputVector{input, any_input()}),
        ov::pass::pattern::wrap_type<ov::opset9::Unsqueeze>(OutputVector{input, any_input()})});

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto node = m.get_match_root();
        std::cerr << "[ INFO TF FE ] Matching Reshape op: " << node->get_type_name() << "\n";

        // Replace each input that consumes StructPack with decomposed tensors
        // Insert StructPack for each output that has type str

        // Inputs
        auto input = node->input(0);
        auto target_shape = node->get_output_partial_shape(0);  // validation has already done a good part of the job here, just reuse
        if(StructuralTypeAttribute::has_type(input.get_tensor().get_rt_info(), element::StructuralType::Str())) {

            auto input_inputs = get_inputs(input.get_source_output().get_node_shared_ptr());

            auto rank = input.get_partial_shape().rank();
            OPENVINO_ASSERT(rank.is_static(), "Rank is dynamic, not supported");
            OPENVINO_ASSERT(target_shape.rank().is_static(), "Expected static rank after Reshape op");
            auto target_rank = target_shape.rank().get_length();
            if(rank.get_length() == 0) {
                std::cerr << "[ 2 ]\n";
                // Scalar case, represented as a single input tensor of rank 1
                OPENVINO_ASSERT(input_inputs.size() == 1, "Expected one input to StructPack when output type is scalar Str");
                OPENVINO_ASSERT(std::dynamic_pointer_cast<ov::opset9::Reshape>(node), "Unsqueezing from a scalar with string element is now unsupported");

                if(target_rank == 0) {
                    // Nothing to do as we are converting scalar to scalar
                    return false;
                }

                // Reshape scalar to non scalar

                auto begins = const_value(0, target_rank);
                auto ends = make_shared<opset9::Reshape>(make_shared<opset9::ShapeOf>(input_inputs[0]), const_value(1, target_rank), false);     // TODO: Unsqeeze looks better?
                auto new_node = make_shared<StructPack>(
                    OutputVector{begins, ends, input_inputs[0]},
                    element::StructuralType::Str(),
                    target_shape);
                replace_node(node, new_node);
                return true;
            } else {
                // Not a scalar case, represented as three input tensors: (begins, ends, elements)
                OPENVINO_ASSERT(input_inputs.size() == 3, "Expected three inputs to StructPack when output type is not a scalar Str");

                if(target_rank == 0) {
                    OPENVINO_ASSERT(false, "Not a scalar to scalar reshape for Str tensors is not supported");
                    return false;
                } else {
                    // Just Reshape indices shape in the same way as this Reshape works
                    OutputVector new_inputs;
                    auto begins = node->clone_with_new_inputs({input_inputs[0], node->input(1).get_source_output()});
                    auto ends = node->clone_with_new_inputs({input_inputs[1], node->input(1).get_source_output()});

                    auto new_node = make_shared<StructPack>(
                        OutputVector{begins, ends, input_inputs[2]},
                        element::StructuralType::Str(),
                        target_shape);
                    replace_node(node, new_node);
                    return true;
                }
            }

        }
        return false;
    };

    auto m = make_shared<ov::pass::pattern::Matcher>(node, "ov::frontend::tensorflow::pass::ThroughStrOpsProp");
    register_matcher(m, callback);
}


bool is_empty_string (Output<Node> output) {
    // TODO: stub, provide real implementation supposing that output is Constant
    return false;
}


ThroughNotEqualProp::ThroughNotEqualProp() {
    // Should better match node that has at least one StructPack at least at one inputs
    auto input1 = wrap_type<StructPack>();
    auto input2 = wrap_type<StructPack>();
    auto node = ov::pass::pattern::wrap_type<ov::opset9::NotEqual>(OutputVector{input1, input2});

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto node = m.get_match_root();
        std::cerr << "[ INFO TF FE ] Matching NotEqual op: " << node->get_type_name() << "\n";

        auto inputs = node->inputs();
        auto target_shape = node->get_output_partial_shape(0);  // validation has already done a good part of the job here, just reuse
        if(
            StructuralTypeAttribute::has_type(inputs[0].get_tensor().get_rt_info(), element::StructuralType::Str()) &&
            StructuralTypeAttribute::has_type(inputs[1].get_tensor().get_rt_info(), element::StructuralType::Str())) {

            // Now support only a signle case: when one of the strings is empty, try to find such a string as an argument first

            auto not_empty_input = is_empty_string(inputs[0].get_source_output().get_node_shared_ptr()) ? inputs[1] : inputs[0];
            auto input_inputs = get_inputs(not_empty_input.get_source_output().get_node_shared_ptr());

            auto new_node = node->clone_with_new_inputs({input_inputs[0], input_inputs[1]});
            replace_node(node, new_node);
            return true;
        }
        return false;
    };

    auto m = make_shared<ov::pass::pattern::Matcher>(node, "ov::frontend::tensorflow::pass::ThroughNotEqualProp");
    register_matcher(m, callback);
}


bool DecomposeStructResults::run_on_model(const std::shared_ptr<Model>& model) {
    // Search for Parameter with List[Tensor] types

    StructuralTypeProxy::BindInputs bind_outputs;

    ResultVector results =
        model->get_results();  // make a copy, leter results in the model are going to be modified

    ResultVector new_results;

    for (size_t i = 0; i < results.size(); ++i) {
        auto result = results[i];
        auto node = result->get_input_node_ptr(0);
        if(is_type<StructPack>(node))
        {
            //BindInputs bind_output({}, );
            std::vector<size_t> indices;
            auto inputs = node->inputs();
            auto st = StructuralTypeAttribute::get(result->get_input_tensor(0).get_rt_info());
            for (auto input : inputs) {
                indices.push_back(new_results.size());
                new_results.push_back({make_shared<opset9::Result>(input.get_source_output())});
            }
            bind_outputs.push_back({indices, st});
            model->remove_result(result);
        } else {
            Any element_type = result->get_element_type();
            bind_outputs.push_back(StructuralTypeProxy::BindInput({new_results.size()}, element::StructuralType::Tensor(element_type)));
            new_results.push_back(result);
            model->remove_result(result);
        }
    }

    model->add_results(new_results);
    StructuralTypeProxy::StructuralTypeMapAttribute(bind_outputs).set_output(model->get_rt_info());
    return true;
}



bool ReplaceParameterByVocab::run_on_model(const std::shared_ptr<Model>& model) {
    // Search for Parameter with List[Tensor] types
    std::cerr << "ReplaceParameterByVocab\n";
    auto parameters = model->get_parameters();
    for (size_t i = 0; i < parameters.size(); ++i) {
        auto parameter = parameters[i];
        std::cerr << "    [ PARAMETER ] " << i << ": ";
        std::cerr << parameter << "\n";
    }

    return true;
}


ThroughTensorListSetItem::ThroughTensorListSetItem() {
    // Should better match node that has at least one StructPack at least at one inputs
    //auto list = wrap_type<StructPack>();
    auto node = wrap_type<ov::op::util::FrameworkNode>(OutputVector{any_input(), any_input(), any_input()});

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        using namespace opset10;

        if(auto node = as_tf_op_type(m.get_match_root(), "TensorListSetItem")) {
            std::cerr << "Found TensorListSetItem: " << node << "\n";
            auto sp = std::dynamic_pointer_cast<StructPack>(node->get_input_node_shared_ptr(0));
            if(!sp) {
                std::cerr << "[ ERROR ] Coudn't decode StructPack at the first input port of TensorListSetItem\n";
                std::cerr << node->get_input_node_shared_ptr(0) << "\n";
                return false;
            }

            auto shapes =   sp->get_input_source_output(0);
            auto begins =   sp->get_input_source_output(1);
            auto ends =     sp->get_input_source_output(2);
            auto elements = sp->get_input_source_output(3);

            auto zero_1d = const_value(0, 1, begins.get_element_type());
            auto zero = const_value(0);
            typedef std::vector<int64_t> V;

            auto index_scalar = node->get_input_source_output(1);
            auto index = make_shared<Unsqueeze>(index_scalar, zero);
            auto item = node->get_input_source_output(2);

            auto begin = make_shared<Gather>(begins, index, zero);
            auto end = make_shared<Gather>(ends, index, zero);
            auto len = make_shared<ShapeOf>(begins, begins.get_element_type());

            // Get two parts of elements: before and after a target item area
            auto before = make_shared<StridedSlice>(elements, zero_1d, begin, V{1}, V{0});
            auto after = make_shared<StridedSlice>(elements, end, zero_1d, V{0}, V{1});
            auto flat = make_shared<Reshape>(item, const_value(-1, 1), false);
            auto new_elements = make_shared<Concat>(OutputVector{before, flat, after}, 0);
            auto new_end = make_shared<Add>(begin, make_shared<ShapeOf>(flat, begins.get_element_type()));
            auto shift = make_shared<Subtract>(new_end, end);

            auto one_1d = const_value(1, 1, begins.get_element_type());

            // TODO: begins_shift/ends_shift don't look very efficient, try StridedSplice, Add and Concat instead

            auto begins_shift = make_shared<Concat>(OutputVector{
                make_shared<Tile>(zero_1d, make_shared<Add>(index, one_1d)),
                make_shared<Tile>(shift, make_shared<Subtract>(make_shared<Subtract>(len, index), one_1d))},
                0);

            auto ends_shift = make_shared<Concat>(OutputVector{
                make_shared<Tile>(zero_1d, index),
                make_shared<Tile>(shift, make_shared<Subtract>(len, index))},
                0);

            auto new_begins = make_shared<Add>(begins, begins_shift);
            auto new_ends = make_shared<Add>(ends, ends_shift);

            //auto shape = make_shared<SpyOp>(OutputVector{make_shared<ShapeOf>(item, shapes.get_element_type())});
            auto shape = make_shared<ShapeOf>(item, shapes.get_element_type());

            #if 0 // this part has an issue in CPU: ranks mismatch presumably due to scalar index
            auto new_shapes = make_shared<ScatterUpdate>(shapes, index_scalar, shape, zero);
            #else
            #if 1
            //auto new_shapes = make_shared<SpyOp>(OutputVector{make_shared<ScatterUpdate>(shapes, index, make_shared<Unsqueeze>(shape, zero), zero)});
            auto new_shapes = make_shared<ScatterUpdate>(shapes, index, make_shared<Unsqueeze>(shape, zero), zero);
            #else
            auto index_2d = make_shared<Unsqueeze>(index, zero);
            make_shared<StridedSlice>(shapes, const_value(0, 2), index_2d, V{1, 1}, V{0, 1})
            make_shared<StridedSlice>(shapes, make_index_2d) ...
            #endif
            #endif

            auto new_sp = sp->clone_with_new_inputs({new_shapes, new_begins, new_ends, new_elements});

            replace_node(node, new_sp);

            return true;
        } else if(auto node = as_tf_op_type(m.get_match_root(), "TensorListGetItem")) {
            std::cerr << "Found TensorListGetItem: " << node << "\n";
            auto sp = std::dynamic_pointer_cast<StructPack>(node->get_input_node_shared_ptr(0));
            if(!sp) {
                std::cerr << "[ ERROR ] Coudn't decode StructPack at the first input port of TensorListGetItem\n";
                std::cerr << node->get_input_node_shared_ptr(0) << "\n";
                return false;
            }

            auto shapes =   sp->get_input_source_output(0);
            auto begins =   sp->get_input_source_output(1);
            auto ends =     sp->get_input_source_output(2);
            auto elements = sp->get_input_source_output(3);

            auto index = node->get_input_source_output(1);
            auto zero = const_value(0);
            typedef std::vector<int64_t> V;

            // Get part of elements which correspond to a required item tensor
            auto flat = make_shared<StridedSlice>(
                elements,
                make_shared<Unsqueeze>(make_shared<Gather>(begins, index, zero), zero),
                make_shared<Unsqueeze>(make_shared<Gather>(ends,   index, zero), zero),
                V{0}, V{0});

            // auto flat = make_shared<StridedSlice>(
            //     elements,
            //     make_shared<SpyOp>(OutputVector{make_shared<Unsqueeze>(make_shared<Gather>(begins, index, zero), zero)}),
            //     make_shared<SpyOp>(OutputVector{make_shared<Unsqueeze>(make_shared<Gather>(ends,   index, zero), zero)}),
            //     V{0}, V{0});

            // Get shape that belongs to that area
            auto shape = make_shared<Gather>(shapes, index, zero);

            // Shape `flat` to obtained `shape`
            // TODO: Learn how to use node->input(2) which contains element shape (double Reshape to fix rank? not sure it adds value...)
            //auto item = make_shared<SpyOp>(OutputVector{make_shared<Reshape>(flat, shape, false)});
            auto item = make_shared<Reshape>(flat, shape, false);
            replace_node(node, item);
            return true;
        }
        return false;
    };

    auto m = make_shared<ov::pass::pattern::Matcher>(node, "ov::frontend::tensorflow::pass::ThroughTensorListSetItem");
    register_matcher(m, callback);
}

struct PatchedInputDesc {
    size_t result_index;    // result index that updates this input at the next iterations
    std::vector<size_t> desc_indices;   // indices in input_descs which should be updated with new results
};

ThroughWhileProp::ThroughWhileProp() {
    // Should better match node that has at least one StructPack at least at one inputs
    auto node = wrap_type<opset10::Loop>();

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        using namespace opset10;
        auto node = std::dynamic_pointer_cast<opset10::Loop>(m.get_match_root());
        std::cerr << "[ INFO TF FE ] Matching While op: " << node->get_type_name() << "\n";

        //std::vector<size_t> remove_parameters;  // indices of removed parameters, collect and the prune them all

        // Search for input with StructPack
        size_t initial_input_size = node->get_input_size();
        auto& input_descs = node->get_input_descriptions();
        auto body = node->get_function();
        std::vector<PatchedInputDesc> pids;

        for(size_t i = 0; i < initial_input_size; ++i) {
            auto input_node = node->get_input_source_output(i).get_node_shared_ptr();
            std::cerr << "Testing input " << input_node << '\n';
            if(auto sp = std::dynamic_pointer_cast<StructPack>(input_node)) {
                std::cerr << "[ WHILE SP PROP ] Found\n";

                // This input is going to be replaced by all inputs coming to `sp`
                //remove_parameters.push_back(i);
                auto new_inputs = get_inputs(sp);
                node->set_argument(i, new_inputs[0]);
                for(size_t j = 1; j < new_inputs.size(); ++j) {
                    node->set_argument(node->get_input_size(), new_inputs[j]);  // add new input to Loop
                }
                // Replace current input/body parameter pair by the first element from new_inputs
                for(size_t k = 0; k < input_descs.size(); ++k) {
                    if(input_descs[k]->m_input_index == i) {
                        PatchedInputDesc pid;
                        auto mid = std::dynamic_pointer_cast<opset10::Loop::MergedInputDescription>(input_descs[k]);
                        if(mid) {
                            pid.result_index = mid->m_body_value_index;
                        }
                        auto old_body_parameter_index = input_descs[k]->m_body_parameter_index;
                        auto old_body_parameter = body->get_parameters()[old_body_parameter_index];
                        std::cerr << "old_body_parameter: " << old_body_parameter << "\n";
                        OutputVector new_parameters;

                        std::cerr << "Loop patching parameter with new shape: " << new_inputs[0].get_partial_shape() << "\n";
                        auto new_parameter = make_shared<opset10::Parameter>(
                            new_inputs[0].get_element_type(),
                            new_inputs[0].get_partial_shape());

                        body->set_parameter(old_body_parameter_index, new_parameter);

                        new_parameters.push_back(new_parameter);

                        // TODO: Any extra steps for inputs with back edges?

                        for(size_t j = 1; j < new_inputs.size(); ++j) {
                            if(mid) {
                                pid.desc_indices.push_back(input_descs.size());
                                input_descs.push_back(make_shared<opset10::Loop::MergedInputDescription>(
                                    node->get_input_size() - new_inputs.size() + j, body->get_parameters().size(), /* body_value_index --> */0/* <-- */));
                                std::cerr << "[ TF FE INFO ] Added new MergedInputDescription with body_value_index left uninitialized\n";
                                // body_value_index is not yet known because there hasn't been extended outputs yet
                                // TODO: Any extra steps for inputs with back edges?
                            } else if(std::dynamic_pointer_cast<opset10::Loop::SliceInputDescription>(input_descs[k])) {
                                std::cerr << "[ ERROR ] Cannot interpret SliceInputDescription for structural type\n";
                                throw "Error";
                            } else if(std::dynamic_pointer_cast<opset10::Loop::InvariantInputDescription>(input_descs[k])) {
                                input_descs.push_back(make_shared<opset10::Loop::InvariantInputDescription>(
                                    node->get_input_size() - new_inputs.size() + j, body->get_parameters().size()));
                                std::cerr << "[ TF FE INFO ] Added new InvariantInputDescription\n";
                            } else {
                                std::cerr << "[ ERROR ] Unknown InputDescription type\n";
                                throw "Error";
                            }

                            std::cerr << "Loop patching parameter with new shape: " << new_inputs[j].get_partial_shape() << "\n";
                            auto new_parameter = make_shared<opset10::Parameter>(
                                new_inputs[j].get_element_type(),
                                j != new_inputs.size() - 1 ? new_inputs[j].get_partial_shape() : PartialShape{Dimension()});    // FIXME: Hack with dynamic dimension due to broken ability to propagate it correctly in the Loop
                            std::cerr << ">>>>>>>>> Created parameter: " << new_parameter << "\n";
                            body->add_parameters({new_parameter});     // add new parameter
                            new_parameters.push_back(new_parameter);
                        }

                        if(mid) {
                            pids.push_back(pid);
                        }

                        // Modify body to connect all new parameters via a new StructPack which is a clone of `sp`
                        auto body_sp = sp->clone_with_new_inputs(new_parameters);
                        std::cerr << "Is about to replace " << old_body_parameter << " by " << body_sp << "\n";
                        replace_node(old_body_parameter, body_sp);
                    }
                }
            }
        }

        std::cerr << "[ TF FE INFO ] Inputs/Parameters are patched\n";

        // Apply StructPack propagation transformations to body
        ov::pass::Manager manager;
        auto propagators = manager.register_pass<ov::pass::GraphRewrite>();
        propagators->add_matcher<ov::pass::GraphRewrite>(std::make_shared<pass::ThroughTensorListSetItem>());
        manager.set_per_pass_validation(false);
        manager.run_passes(body);

        size_t initial_body_output_size = body->get_results().size();
        auto& output_descs = node->get_output_descriptions();
        size_t output_descs_size = output_descs.size();

        for(size_t i = 0; i < initial_body_output_size; ++i) {
            auto result = body->get_results()[i];
            auto source = result->get_input_node_shared_ptr(0);
            if(auto sp = std::dynamic_pointer_cast<StructPack>(source)) {
                auto inputs = get_inputs(sp);
                auto new_result_0 = make_shared<Result>(inputs[0]);
                body->set_result(i, new_result_0);
                for(size_t j = 1; j < inputs.size(); ++j) {
                    auto new_result_n = make_shared<Result>(inputs[j]);
                    body->add_results({new_result_n});
                }

                for(size_t k = 0; k < pids.size(); ++k) {
                    if(pids[k].result_index == i) {
                        for(size_t m = 0; m < pids[k].desc_indices.size(); ++m) {
                            std::dynamic_pointer_cast<opset10::Loop::MergedInputDescription>(
                                input_descs[pids[k].desc_indices[m]])->m_body_value_index =
                                    body->get_results().size() - inputs.size() + 1 + m;
                            std::cerr << "Patching input desk: result_index = " << pids[k].result_index << ", m_body_value_index = " << std::dynamic_pointer_cast<opset10::Loop::MergedInputDescription>(input_descs[pids[k].desc_indices[m]])->m_body_value_index << "\n";
                        }
                    }
                }

                for(size_t k = 0; k < output_descs_size; ++k) {
                    if(output_descs[k]->m_body_value_index == i) {
                        if(auto desc = std::dynamic_pointer_cast<opset10::Loop::BodyOutputDescription>(output_descs[k])) {
                            if(desc->m_iteration != -1) {
                                std::cerr << "[ ERROR ] Unsupported output iteration index, should be -1\n";
                                throw "Error";
                            }

                            std::cerr << "Patching output ports\n";

                            OutputVector new_outputs;
                            new_outputs.push_back(node->output(desc->m_output_index));
                            std::cerr << "Output index: " << desc->m_output_index << "\n";

                            for(size_t j = 1; j < inputs.size(); ++j) {
                                output_descs.push_back(make_shared<opset10::Loop::BodyOutputDescription>(
                                    body->get_results().size() - inputs.size() + j, node->get_output_size(), desc->m_iteration
                                ));
                                std::cerr << "    Output port " << node->get_output_size() << "\n";
                                node->set_output_size(node->get_output_size() + 1);
                                new_outputs.push_back(node->output(node->get_output_size() - 1));
                            }

                            auto target_inputs = node->get_output_target_inputs(desc->m_output_index);
                            auto new_sp = sp->clone_with_new_inputs(new_outputs);
                            std::cerr << "new_sp = " << new_sp << "\n";
                            std::cerr << "with this loop: " << node << "\n";

                            //node->output(desc->m_output_index).replace(new_sp->output(0));  // FIXME

                            for(auto input: target_inputs) {
                                input.replace_source_output(new_sp);
                            }

                        } else {
                            std::cerr << "[ ERROR ] StructPack passing out not supported output map type\n";
                            throw "Error";
                        }
                    }
                }
            }
        }

        //std::cerr << "About to validate body\n";
        //body->validate_nodes_and_infer_types();

        std::cerr << "About to validate loop\n";
        node->validate_and_infer_types();

        std::cerr << "Start serializing the body\n";
        serialize(body, "body.xml");

        std::cerr << "[TF FE INFO] Done with Loop\n";
        //throw "Error";

        return true;
    };

    auto m = make_shared<ov::pass::pattern::Matcher>(node, "ov::frontend::tensorflow::pass::ThroughWhileProp");
    register_matcher(m, callback);
}


ThroughTensorListStack::ThroughTensorListStack() {
    // Should better match node that has at least one StructPack at least at one inputs
    //auto list = wrap_type<StructPack>();
    auto node = wrap_type<ov::op::util::FrameworkNode>(OutputVector{any_input(), any_input()});

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        using namespace opset10;

        if(auto node = as_tf_op_type(m.get_match_root(), "TensorListStack")) {
            std::cerr << "Found TensorListStack: " << node << "\n";
            auto sp = std::dynamic_pointer_cast<StructPack>(node->get_input_node_shared_ptr(0));
            if(!sp) {
                std::cerr << "[ ERROR ] Coudn't decode StructPack at the first input port of TensorListStack\n";
                std::cerr << node->get_input_node_shared_ptr(0) << "\n";
                return false;
            }

            std::cerr << "Start TensorListStack decomposition\n";

            auto shapes =   sp->get_input_source_output(0);
            auto begins =   sp->get_input_source_output(1);
            auto ends =     sp->get_input_source_output(2);
            auto elements = sp->get_input_source_output(3);

            auto zero = const_value(0);

            // ! Suppose the list has at least one element, which is used to deduce shape
            //   Otherwise we need to check an input to this node with shape, but it may have undefined dimensions with -1 value, in which case there is no reliable source of shape

            // ! Suppose all elements have the same shape at least in all dimension except the 0th

            auto num_items = make_shared<ShapeOf>(begins, shapes.get_element_type());
            auto shape = make_shared<Gather>(shapes, zero, zero);
            auto target_shape = make_shared<Concat>(OutputVector{num_items, shape}, 0);

            // FIXME: Due to empty tensor padding an extra StrideSlice is required to cut off the padding
            auto target_shape_size = make_shared<ReduceProd>(target_shape, const_value(0), true);
            typedef std::vector<int64_t> V;
            auto cut_elements = make_shared<StridedSlice>(elements, const_value(0, 1), target_shape_size, V{1}, V{0});

            auto reshaped = make_shared<Reshape>(cut_elements, target_shape, false);

            replace_node(node, reshaped);

            std::cerr << "End TensorListStack decomposition\n";

            return true;
        }

        return false;
    };

    auto m = make_shared<ov::pass::pattern::Matcher>(node, "ov::frontend::tensorflow::pass::ThroughTensorListStack");
    register_matcher(m, callback);
}


}
}
}
}
