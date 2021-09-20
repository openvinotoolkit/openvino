// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>
#include <ngraph/pass/manager.hpp>
#include <numeric>

#include "ngraph/op/util/logical_reduction.hpp"
#include "ngraph/pass/constant_folding.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/pass_config.hpp"
#include "ngraph/slice_plan.hpp"
//#include <ngraph/pass/transpose_sinking.h>
#include <ngraph/pass/constant_folding.hpp>
#include <tensorflow_frontend/place.hpp>

#include "default_opset.h"
#include "graph.hpp"
#include "ngraph_builder.h"
#include "ngraph_conversions.h"
#include "op_table.hpp"
#include "utils.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {
namespace ngraph_bridge {

    void Builder::SetTracingInfo(const std::string& op_name, const ng::Output<ng::Node> ng_node) {
    auto node = ng_node.get_node_shared_ptr();
    node->set_friendly_name(op_name);
    node->add_provenance_tag(op_name);
}

const Builder::ConstMap& Builder::TF_NGRAPH_CONST_MAP() {
    static const Builder::ConstMap the_map = {
            {ng::element::f32, make_pair(MakeConstOp<float>, ng::element::f32)},
            {ng::element::f64, make_pair(MakeConstOp<double>, ng::element::f64)},
            {ng::element::i8, make_pair(MakeConstOp<int8_t>, ng::element::i8)},
            {ng::element::i16, make_pair(MakeConstOp<int16_t>, ng::element::i16)},
#if 0
            {DataType::DT_QINT8, make_pair(MakeConstOp<qint8>, ng::element::i8)},
  {DataType::DT_QUINT8, make_pair(MakeConstOp<quint8>, ng::element::u8)},
  {DataType::DT_QUINT16, make_pair(MakeConstOp<quint16>, ng::element::u16)},
#endif
            {ng::element::i32, make_pair(MakeConstOp<int32_t>, ng::element::i32)},
            {ng::element::i64, make_pair(MakeConstOp<int64_t>, ng::element::i64)},
            {ng::element::u8, make_pair(MakeConstOp<uint8_t>, ng::element::u8)},
            {ng::element::u16, make_pair(MakeConstOp<uint16_t>, ng::element::u16)},
            {ng::element::boolean, make_pair(MakeConstOp<bool, char>, ng::element::boolean)}
    };
    return the_map;
}

void Builder::TranslateGraph(
    std::shared_ptr<ngraph::frontend::InputModelTensorflow> tf_model,
    const std::vector<const ngraph::frontend::tensorflow::detail::TensorWrapper*>& static_input_map,
    const std::string name,
    std::shared_ptr<ngraph::Function>& ng_function) {
    //
    // The op map holds a mapping from TensorFlow op names (strings) to
    // vector of generated nGraph Output<TFNodeDecoder>.
    //
    Builder::OpMap ng_op_map;

    ngraph::ParameterVector params;
    ngraph::ResultVector results;

    const auto& ops = tf_model->get_op_places();
    const auto& inputs = tf_model->partialShapes;
    const auto& indexed_shapes = tf_model->input_shapes;

    //
    // Now create the nGraph ops from TensorFlow ops.
    //
    const auto CREATORS_MAP = get_supported_ops();
    for (auto& op_place : ops) {
        auto op = op_place->get_desc();
        auto op_name = op_place->get_names()[0];
#if 0
    // TODO: Investigate why do we need it
      if (n->IsSink() || n->IsSource()) {
      continue;
    }
#endif
        if (op->IsControlFlow()) {
            throw errors::Unimplemented("Encountered a control flow op in the nGraph bridge: " + op->DebugString());
        }

        NGRAPH_VLOG(2) << "Constructing op " << op_name << " which is " << op->type_string() << "\n";

        // const function<Status(const TFNodeDecoder*, const std::vector<const
        // ngraph::frontend::tensorflow::detail::TensorWrapper*>&,
        //                      Builder::OpMap&)>* op_fun;
        auto creator_it = CREATORS_MAP.find(op->type_string());

        // todo: replace it with TF exception
        NGRAPH_CHECK(creator_it != CREATORS_MAP.end(), "No creator found for ", op->type_string(), " node.");

        try {
            // Pre-processing: prepare a list of ng inputs for the node
            ngraph::OutputVector ng_inputs;
            for (size_t i = 0; i < op->num_inputs(); ++i) {
                std::string input_name;
                size_t port_idx;
                try {
                    op->input_node(i, &input_name, &port_idx);
                    ng_inputs.push_back(ng_op_map.at(input_name).at(port_idx));
                } catch (const std::exception& e) {
                    std::cerr << "[ ERROR ] Exception happened when preparing input " << i << " for op '" << op->name()
                              << "', expected input name: '" << input_name
                              << "', expected input port index: " << port_idx << '\n';
                    throw;
                }
            }
            NodeContext node_context(ng_inputs, op, inputs, indexed_shapes);

            // Next line does the conversion for a node by means of calling specific conversion rule
            auto outputs = (creator_it->second)(node_context);

            // Post-processing: register outputs to the map and detect the edge ops
            auto& node_record = ng_op_map[op_name];
            for (auto output : outputs) {
                if (auto result = std::dynamic_pointer_cast<opset::Result>(output.get_node_shared_ptr())) {
                    results.push_back(result);
                    // Do not add to ng_op_map
                } else {
                    if (auto param = std::dynamic_pointer_cast<opset::Parameter>(output.get_node_shared_ptr())) {
                        params.push_back(param);
                    }
                    node_record.push_back(output);
                }
            }
        } catch (const Status& e) {
            throw errors::Internal("Unhandled exception in op handler: " + op_name + " (" + op->type_string() + ")\n" +
                                   op->DebugString() + "\nDetails: " + e.message);
        } catch (const std::exception& e) {
            throw errors::Internal("Unhandled exception in op handler: " + op_name + " (" + op->type_string() + ")\n" +
                                   op->DebugString() + "\n" + "what(): " + e.what());
        } catch (...) {
            throw errors::Internal("Unhandled exception in op handler: " + op_name + " (" + op->type_string() + ")\n" +
                                   op->DebugString());
        }
    }

    if (results.empty()) {  // TODO: Provide a control to trigger this at FE level, currently this is heuristics
        // Find all terminal nodes in ngraph graph to complete list of results
        for (const auto& p : ng_op_map) {
            for (auto output : p.second) {
                if (output.get_target_inputs().empty() &&
                    !std::dynamic_pointer_cast<opset::Result>(
                        output.get_node_shared_ptr()))  // Exclude existing Results
                    results.push_back(std::make_shared<default_opset::Result>(output));
            }
        }
    }

    // TODO: Reorder results and params according to indices given in RT info (if any)

    //
    // Create the nGraph function.
    //
    ng_function = make_shared<ng::Function>(results, params, name);
    //
    // Request row-major layout on results.
    //
    // TODO: Why do we need this?
    // for (auto result : ng_function->get_results()) {
    //  result->set_needs_default_layout(true);
    //}
    NGRAPH_VLOG(5) << "Done with translations";
}

}  // namespace ngraph_bridge
}  // namespace tensorflow
