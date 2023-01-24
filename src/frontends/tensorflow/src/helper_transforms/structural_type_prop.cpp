// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "structural_type_prop.hpp"

#include <memory>
#include <vector>
#include <numeric>

#include "../helper_ops/str_ops.hpp"

#include "openvino/frontend/tensorflow/frontend.hpp"
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include "openvino/core/type/non_tensor_type.hpp"
#include <openvino/opsets/opset9.hpp>

using std::make_shared;
using std::shared_ptr;

namespace {
    // TODO: Remove this duplicate: CPU transforms has it, copied and pasted here

    bool is_data_movement_operation(const std::shared_ptr<ngraph::Node>& node) {
        return ov::is_type<ngraph::op::v0::Squeeze>(node) ||
               ov::is_type<ov::op::v1::StridedSlice>(node) ||
               ov::is_type<ngraph::op::v0::Unsqueeze>(node) ||
               ov::is_type<ngraph::op::v1::Reshape>(node) ||
               ov::is_type<ngraph::op::v1::Transpose>(node) ||
               ov::is_type<ngraph::op::v0::ShuffleChannels>(node) ||
               ov::is_type<ngraph::op::v7::Roll>(node) ||
               ov::is_type<ngraph::op::v0::ReverseSequence>(node) ||
               ov::is_type<ngraph::op::v0::DepthToSpace>(node) ||
               ov::is_type<ngraph::op::v1::BatchToSpace>(node) ||
               ov::is_type<ngraph::op::v1::Broadcast>(node) ||
               ov::is_type<ngraph::op::v3::Broadcast>(node) ||
               ov::is_type<ngraph::op::v1::Gather>(node) ||
               ov::is_type<ngraph::op::v7::Gather>(node) ||
               ov::is_type<ngraph::op::v8::Gather>(node) ||
               ov::is_type<ngraph::op::v0::Parameter>(node);
    }

    bool is_str_operation(const std::shared_ptr<ngraph::Node>& node) {
        return ov::is_type<ov::frontend::tensorflow::CaseFoldUTF8>(node) ||
               ov::is_type<ov::frontend::tensorflow::NormalizeUTF8>(node) ||
               ov::is_type<ov::frontend::tensorflow::StaticRegexReplace>(node) ||
               ov::is_type<ov::frontend::tensorflow::WordpieceTokenizeWithOffsets>(node) ||
               ov::is_type<ov::frontend::tensorflow::RegexSplitWithOffsets>(node);
    }

    bool is_scalar_like(const std::shared_ptr<ngraph::Node>& node) {
        auto constantNode = std::dynamic_pointer_cast<ngraph::opset8::Constant>(node);
        return constantNode != nullptr && shape_size(constantNode->get_shape()) == 1;
    }
} // namespace

namespace ov {
namespace frontend {
namespace tensorflow {
namespace pass {

StructuralTypeProp::StructuralTypeProp() {
    //auto data_movement = ngraph::pattern::wrap_type<ov::op::Op>(
    //    ov::pass::pattern::op::as_value_predicate(is_data_movement_operation));
    auto data_movement = ngraph::pattern::wrap_type<ov::op::Op>(
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
    auto str_tensor = ngraph::pattern::wrap_type<ov::op::Op>(
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
        std::cerr << "[ PARAMETER ] " << i << "\n";
        std::cerr << parameter << "\n";

        // Check 1D and Str structural type
        auto rank = parameter->get_partial_shape().rank();
        auto rt_info = parameter->get_rt_info();
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


using ov::opset9::Constant;

template <typename T>
shared_ptr<Constant> const_value (const T& value, size_t rank = 0, element::Type et = element::i32) {
    return make_shared<Constant>(et, Shape(rank, 1), value);
}


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
    auto node = ngraph::pattern::wrap_type<ov::opset9::Reshape>(OutputVector{input, any_input()});

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
            FRONT_END_GENERAL_CHECK(rank.is_static(), "Rank is dynamic, not supported");
            FRONT_END_GENERAL_CHECK(target_shape.rank().is_static(), "Expected static rank after Reshape op");
            auto target_rank = target_shape.rank().get_length();
            if(rank.get_length() == 0) {
                std::cerr << "[ 2 ]\n";
                // Scalar case, represented as a single input tensor of rank 1
                FRONT_END_GENERAL_CHECK(input_inputs.size() == 1, "Expected one input to StructPack when output type is scalar Str");

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
                FRONT_END_GENERAL_CHECK(input_inputs.size() == 3, "Expected three inputs to StructPack when output type is not a scalar Str");

                if(target_rank == 0) {
                    FRONT_END_GENERAL_CHECK(false, "Not a scalar to scalar reshape for Str tensors is not supported");
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
        std::cerr << "[ PARAMETER ] " << i << "\n";
        std::cerr << parameter << "\n";

        // Check consumers, find for WordpieceTokenize
        auto consumers = parameter->get_output_target_inputs(0);
        for(std::set<Input<Node>>::iterator i = consumers.begin(); i != consumers.end(); ++i) {
            if(std::dynamic_pointer_cast<WordpieceTokenizeWithOffsets>(i->get_node())) {
                std::cerr << "BINGO!\n";
            }
        }
    }

    return true;
}


}
}
}
}
