// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <memory>
#include <vector>
#include <unordered_set>
#include <regex>
#include <sstream>

#include <cnn_network_ngraph_impl.hpp>
#include "ngraph_ops/convolution_ie.hpp"
#include "ngraph_ops/deconvolution_ie.hpp"
#include "legacy/ngraph_ops/eltwise.hpp"
#include "legacy/ngraph_ops/fully_connected.hpp"
#include "legacy/ngraph_ops/gather_ie.hpp"
#include "legacy/ngraph_ops/gather_tree_ie.hpp"
#include "legacy/ngraph_ops/gru_cell_ie.hpp"
#include "legacy/ngraph_ops/interp.hpp"
#include "legacy/ngraph_ops/lrn_ie.hpp"
#include "legacy/ngraph_ops/lstm_cell_ie.hpp"
#include "legacy/ngraph_ops/normalize_ie.hpp"
#include "legacy/ngraph_ops/pad_ie.hpp"
#include "legacy/ngraph_ops/onehot_ie.hpp"
#include "legacy/ngraph_ops/power.hpp"
#include "legacy/ngraph_ops/prior_box_clustered_ie.hpp"
#include "legacy/ngraph_ops/prior_box_ie.hpp"
#include "legacy/ngraph_ops/proposal_ie.hpp"
#include "legacy/ngraph_ops/relu_ie.hpp"
#include "legacy/ngraph_ops/scaleshift.hpp"
#include "legacy/ngraph_ops/tile_ie.hpp"
#include "legacy/ngraph_ops/hard_sigmoid_ie.hpp"
#include "legacy/ngraph_ops/nms_ie.hpp"
#include "legacy/ngraph_ops/crop_ie.hpp"
#include "legacy/ngraph_ops/selu_ie.hpp"
#include "legacy/ngraph_ops/rnn_cell_ie.hpp"
#include "legacy/ngraph_ops/topk_ie.hpp"
#include "legacy/ngraph_ops/rnn_sequence_ie.hpp"
#include "legacy/ngraph_ops/lstm_sequence_ie.hpp"
#include "legacy/ngraph_ops/gru_sequence_ie.hpp"
#include "exec_graph_info.hpp"

#include "caseless.hpp"
#include <debug.h>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset5.hpp>
#include "transformations/utils/utils.hpp"
#include "transformations/rt_info/fused_names_attribute.hpp"
#include "transformations/rt_info/primitives_priority_attribute.hpp"
#include "cpp/ie_cnn_network.h"

#include "legacy/convert_function_to_cnn_network.hpp"
#include "legacy/graph_tools.hpp"
#include "legacy/net_pass.h"
#include "ie_legacy_itt.hpp"

namespace Builder {

template <class T>
std::string asString(const T& value) {
    return std::to_string(value);
}

template <typename T>
std::string asString(const std::vector<T>& value) {
    std::string result;
    for (const auto& item : value) {
        if (!result.empty()) result += ",";
        result += asString(item);
    }
    return result;
}

template <>
std::string asString<double>(const double& value) {
    std::ostringstream sStrm;
    sStrm.precision(std::numeric_limits<double>::digits10);
    sStrm << std::fixed << value;
    std::string result = sStrm.str();

    auto pos = result.find_last_not_of("0");
    if (pos != std::string::npos) result.erase(pos + 1);

    pos = result.find_last_not_of(".");
    if (pos != std::string::npos) result.erase(pos + 1);

    return result;
}

template <>
std::string asString<float>(const float& value) {
    return asString(static_cast<double>(value));
}

}  // namespace Builder

namespace InferenceEngine {
namespace details {

// helper for adding creators with a specific exception
#define REQUIRED_IE_CONVERSION_CREATOR(type_name, ie_type_name)\
    addSpecificCreator({type_name}, [](const std::shared_ptr<::ngraph::Node>& node, \
        const std::map<std::string, std::string>& params) -> CNNLayerPtr {\
        IE_THROW() << type_name  << " operation has a form that is not supported. " << node->get_friendly_name()\
        << " should be converted to " << ie_type_name << " operation.";\
        return nullptr;\
    });\

/// \brief Creates legacy representation of CNNLayer for SubGraphOp.
/// \param layer node type
/// \return pointer to CNNLayer with legacy representation of SubGraphOp.
CNNLayer::Ptr createSubGraphLayer(const std::shared_ptr<ngraph::Node>& layer) {
    auto sub_graph = std::dynamic_pointer_cast<ngraph::op::util::SubGraphOp>(layer);
    if (!sub_graph) {
        IE_THROW() << "Cannot cast layer to SubGraphOp.";
    }

    // inputs/outputs of TensorIterator (ngraph representation)
    auto parameters = sub_graph->get_function()->get_parameters();
    auto results = sub_graph->get_function()->get_results();

    // Convert body (ngraph representation) to CNNNetwork.
    // This network will contain nodes of type = "Input" and data nodes with wrong names.
    // IE TensorIterator doesn't include such nodes so we create CNNNetwork in a separate scope
    // to call the destructor and delete these "Input"/data nodes.

    TensorIterator::Body body;
    {
        InferenceEngine::CNNNetwork body_net(sub_graph->get_function());
        IE_SUPPRESS_DEPRECATED_START
        // TODO: fix convertFunctionToICNNNetwork
        InferenceEngine::CNNNetwork net(InferenceEngine::details::convertFunctionToICNNNetwork(body_net.getFunction(), body_net));
        IE_SUPPRESS_DEPRECATED_END
        // Paranoid check for cycles
        bool res = CNNNetForestDFS(
            CNNNetGetAllInputLayers(net), [](const CNNLayerPtr& layer) {}, false);
        if (!res) {
            IE_THROW() << "Loop detected. SubGraphOp body should not contain loops.";
        }

        // Get inputs/outputs of cnn network
        auto in_info_map_with_parameters = net.getInputsInfo();
        auto out_info_map = net.getOutputsInfo();

        IE_ASSERT(in_info_map_with_parameters.size() == parameters.size());
        IE_ASSERT(out_info_map.size() == results.size());

        InferenceEngine::TensorIterator::Body temp_body;
        temp_body.inputs.resize(in_info_map_with_parameters.size());
        temp_body.outputs.resize(out_info_map.size());

        // Fill inputs/outs in order aligned with ng representation
        uint64_t counter = 0;
        for (const auto& param : parameters) {
            auto info = in_info_map_with_parameters.at(param->get_friendly_name());
            temp_body.inputs[counter++] = info->getInputData();
        }

        auto map_ng_result_to_ie_name = [] (std::shared_ptr<ngraph::op::v0::Result> res_op) {
            auto result = res_op->input(0).get_source_output();

            std::string name = result.get_node()->get_friendly_name();
            if (result.get_node()->get_output_size() > 1) {
                name += "." + std::to_string(result.get_index());
            }
            return name;
        };

        counter = 0;
        for (const auto& result : results) {
            auto data = out_info_map.at(map_ng_result_to_ie_name(result));
            temp_body.outputs[counter++] = data;
        }

        // This deep copy will hold all unreachable constants. See the comment in CopyTIBody function.
        body = InferenceEngine::NetPass::CopyTIBody(temp_body);

        // Check if data is really const layer holder
        auto is_constant_holder = [] (const DataPtr data) {
            return data->getPrecision() == Precision::UNSPECIFIED;
        };

        // Strip unreached node holder from Inputs node.
        auto holder = body.inputs.back();
        if (is_constant_holder(holder)) {
            auto& holder_map = getInputTo(holder);

            for (auto it = holder_map.begin(); it != holder_map.end(); ) {
                if (it->second->type == "Input")
                    it = holder_map.erase(it);
                else
                    ++it;
            }
        }

        // TODO: Disable this WA after total switch onto Ngraph
        //   WA: Some plugins (like GPU) require matching of Data object name and producer Layer name.
        //       Data name is expected in format "[layer_name]" or "[layer_name].[port_idx]" in case
        //       of multiple inputs. We have to restore it if possible and ignore original names of
        //       Ngraph parameter and result ops.
        //       Will not change data name if:
        //        - data has several consumer layers
        //        - data has no consumer (example if data is straight used as output)
        //
        for (auto &in : body.inputs) {
            if (is_constant_holder(in))
                continue;

            const auto input_to = getInputTo(in);
            if (input_to.size() != 1)
                continue;

            const auto consumer_layer = input_to.begin()->second;
            const auto consumer_in_port_set = consumer_layer->insData;
            const auto found = std::find_if(consumer_in_port_set.begin(), consumer_in_port_set.end(),
                                      [&in] (const DataWeakPtr &wptr) { return wptr.lock() == in; });
            IE_ASSERT(found != consumer_in_port_set.end());
            const auto consumer_port_idx = std::distance(consumer_in_port_set.begin(), found);

            auto new_name = consumer_layer->name;
            if (consumer_in_port_set.size() > 1) {
                new_name += '.' + std::to_string(consumer_port_idx);
            }
            in->setName(new_name);
        }

        // TODO: this WA restore original precisions of outputs.
        //       convertFunctionToICNNNetwork has internal fallback policy for unsupported
        //       precisions for inputs/outputs ports. Particular for U8 will be translated
        //       to FP32. However Loop body has strong requirements for continue_condition
        //       port, it should be BOOL(U8).
        //
        for (size_t i = 0; i < results.size(); i++) {
            auto result = results[i];
            auto output = body.outputs[i];
            if (result->get_element_type() == ngraph::element::u8) {
                output->setPrecision(InferenceEngine::Precision::U8);
            }
        }
    }

    // Create Inference Engine representation of TensorIterator
    LayerParams params = {layer->get_friendly_name(), "TensorIterator",
                          details::convertPrecision(layer->get_output_element_type(0))};
    auto res = std::make_shared<InferenceEngine::TensorIterator>(params);
    if (res == nullptr) {
        IE_THROW() << "Can't create TensorIterator";
    }
    res->body = body;

    // Port map: outputs
    for (const auto& desc : sub_graph->get_output_descriptions()) {
        auto body_output_idx = desc->m_body_value_index;

        std::string type_name = desc->get_type_info().name;
        if (type_name == "ConcatOutputDescription") {
            auto output_desc = ::ngraph::as_type_ptr<ngraph::op::util::SubGraphOp::ConcatOutputDescription>(desc);
            IE_ASSERT(output_desc != nullptr);

            res->output_port_map.emplace_back(InferenceEngine::TensorIterator::PortMap {
                static_cast<int>(output_desc->m_output_index), static_cast<int>(body_output_idx),
                static_cast<int>(output_desc->m_axis), static_cast<int>(output_desc->m_stride),
                static_cast<int>(output_desc->m_start), static_cast<int>(output_desc->m_end),
                static_cast<int>(output_desc->m_part_size)});
        } else if (type_name == "BodyOutputDescription") {
            auto output_desc = ::ngraph::as_type_ptr<ngraph::op::util::SubGraphOp::BodyOutputDescription>(desc);
            IE_ASSERT(output_desc != nullptr);

            res->output_port_map.emplace_back(InferenceEngine::TensorIterator::PortMap {
                static_cast<int>(output_desc->m_output_index), static_cast<int>(body_output_idx), -1, 1, 0, -1, 1});
        } else {
            IE_THROW() << "Incorrect type of the output description.";
        }
    }

    // Port map : inputs and back edges
    for (const auto& desc : sub_graph->get_input_descriptions()) {
        auto body_input_index = desc->m_body_parameter_index;

        if (const auto slice_desc = std::dynamic_pointer_cast<ngraph::op::util::SubGraphOp::SliceInputDescription>(desc)) {
            res->input_port_map.emplace_back(InferenceEngine::TensorIterator::PortMap {
                static_cast<int>(slice_desc->m_input_index), static_cast<int>(body_input_index),
                static_cast<int>(slice_desc->m_axis), static_cast<int>(slice_desc->m_stride),
                static_cast<int>(slice_desc->m_start), static_cast<int>(slice_desc->m_end),
                static_cast<int>(slice_desc->m_part_size)});
        } else if (const auto merge_desc = std::dynamic_pointer_cast<ngraph::op::util::SubGraphOp::MergedInputDescription>(desc)) {
            res->input_port_map.emplace_back(InferenceEngine::TensorIterator::PortMap {
                static_cast<int>(merge_desc->m_input_index), static_cast<int>(body_input_index), -1, 1, 0, -1, 1});

            auto body_output_idx = merge_desc->m_body_value_index;

            res->back_edges.emplace_back(InferenceEngine::TensorIterator::PortMap {
                static_cast<int>(body_output_idx), static_cast<int>(body_input_index), -1, 1, 0, -1, 1});
        } else if (const auto inv_desc = std::dynamic_pointer_cast<ngraph::op::util::SubGraphOp::InvariantInputDescription>(desc)) {
            res->input_port_map.emplace_back(InferenceEngine::TensorIterator::PortMap {
                    static_cast<int>(inv_desc->m_input_index), static_cast<int>(body_input_index), -1, 1, 0, -1, 1});
        } else {
            IE_THROW() << "Incorrect type of the input description.";
        }
    }

    if (const auto loop_op = std::dynamic_pointer_cast<const ngraph::opset5::Loop>(layer)) {
        auto spec_port = loop_op->get_special_body_ports();
        if (spec_port.current_iteration_input_idx != -1) {
            auto ie_port_idx = spec_port.current_iteration_input_idx;
            res->params["loop_body_current_iteration_idx"] = std::to_string(ie_port_idx);
        }
        if (spec_port.body_condition_output_idx != -1) {
            auto body_output_idx = spec_port.body_condition_output_idx;
            res->params["loop_body_condition_output_idx"] = std::to_string(body_output_idx);
        }
        res->params["loop_trip_count_idx"] = "0";
        res->params["loop_execution_condition_idx"] = "1";
    }

    return res;
}

/**
 * @brief Creator for CNNLayer from nGraph op
 */
class CNNLayerCreator : public ::ngraph::AttributeVisitor {
public:
    using CreatorFor = std::function<CNNLayerPtr(const std::shared_ptr<::ngraph::Node>& node,
                                                 const std::map<std::string, std::string>& param)>;
    explicit CNNLayerCreator(const std::shared_ptr<::ngraph::Node>& node);

    CNNLayerPtr create();

    void on_adapter(const std::string& name, ::ngraph::ValueAccessor<bool> &value) override {
        params[name] = value.get() ? "true" : "false";
    }

    void addSpecificCreator(const std::vector<std::string>& forTypes, const CreatorFor& creator) {
        for (const auto type : forTypes) {
            creators[type] = creator;
        }
    }

    void on_adapter(const std::string& name, ::ngraph::ValueAccessor<std::string>& adapter) override {
        std::string data = adapter.get();
        std::transform(data.begin(), data.end(), data.begin(), [](unsigned char c) {
            return std::tolower(c);
        });
        params[name] = data;
    }

    void on_adapter(const std::string& name, ::ngraph::ValueAccessor<std::vector<int32_t>>& adapter) override {
        auto shape = adapter.get();
        params[name] = joinVec(shape);
    }

    void on_adapter(const std::string& name, ::ngraph::ValueAccessor<std::vector<int64_t>>& adapter) override {
        auto shape = adapter.get();
        params[name] = joinVec(shape);
    }

    void on_adapter(const std::string& name, ::ngraph::ValueAccessor<double>& adapter) override {
        std::ostringstream stream;
        stream.precision(8);
        stream << std::fixed << adapter.get();
        params[name] = stream.str();
    }

    void on_adapter(const std::string& name, ::ngraph::ValueAccessor<int64_t>& adapter) override {
        params[name] = std::to_string(adapter.get());
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<std::string>>& adapter) override {
        std::vector<std::string> data = adapter.get();
        for (auto& str : data) {
            std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) {
                return std::tolower(c);
            });
        }

        std::stringstream ss;
        std::copy(data.begin(), data.end(), std::ostream_iterator<std::string>(ss, ","));
        params[name] = ss.str();
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<float>>& adapter) override {
        auto data = adapter.get();
        params[name] = joinVec(data);
    }

    void on_adapter(const std::string& name, ::ngraph::ValueAccessor<std::shared_ptr<ngraph::Function>>& adapter) override {
    }

    void on_adapter(const std::string& name, ::ngraph::ValueAccessor<void>& adapter) override;

private:
    std::shared_ptr<::ngraph::Node> node;
    std::map<std::string, std::string> params;
    std::map<std::string, CreatorFor> creators;
};

void InferenceEngine::details::CNNLayerCreator::on_adapter(const std::string& name,
                                                           ::ngraph::ValueAccessor<void>& adapter) {
    if (auto a = ::ngraph::as_type<::ngraph::AttributeAdapter<::ngraph::element::Type>>(&adapter)) {
        auto type = static_cast<::ngraph::element::Type&>(*a);
        params[name] = details::convertPrecision(type).name();
    } else if (auto a = ::ngraph::as_type<::ngraph::AttributeAdapter<::ngraph::PartialShape>>(&adapter)) {
        std::string dims;
        auto shape = static_cast<::ngraph::PartialShape&>(*a);
        for (int64_t i = 0; i < shape.rank().get_length(); i++) {
            if (!dims.empty()) dims += ",";
            dims += std::to_string(shape[i].get_length());
        }
        params[name] = dims;
    } else if (auto a = ::ngraph::as_type<::ngraph::AttributeAdapter<::ngraph::Shape>>(&adapter)) {
        auto shape = static_cast<::ngraph::Shape&>(*a);
        params[name] = joinVec(shape);
    } else if (auto a = ::ngraph::as_type<::ngraph::AttributeAdapter<::ngraph::Strides>>(&adapter)) {
        auto shape = static_cast<::ngraph::Strides&>(*a);
        params[name] = joinVec(shape);
    } else if (auto a = ::ngraph::as_type<::ngraph::AttributeAdapter<std::vector<size_t>>>(& adapter)) {
        auto data = a->get();
        params[name] = joinVec(data);
    } else if (auto a = ::ngraph::as_type<::ngraph::AttributeAdapter<std::shared_ptr<::ngraph::Variable>>>(& adapter)) {
        params[name] = a->get()->get_info().variable_id;
    } else if (auto a = ::ngraph::as_type<::ngraph::AttributeAdapter<std::vector<std::shared_ptr<
    ngraph::op::util::SubGraphOp::InputDescription>>>>(& adapter)) {
        (void)a;
    } else if (auto a = ::ngraph::as_type<::ngraph::AttributeAdapter<std::vector<std::shared_ptr<
    ngraph::op::util::SubGraphOp::OutputDescription>>>>(& adapter)) {
        (void)a;
    } else if (auto a = ::ngraph::as_type<::ngraph::AttributeAdapter<ngraph::op::v5::Loop::SpecialBodyPorts>>(& adapter)) {
        (void)a;
    } else if (auto a = ::ngraph::as_type<::ngraph::AttributeAdapter<std::shared_ptr<ngraph::runtime::AlignedBuffer>>>(& adapter)) {
        if (std::string(node->get_type_name()) != "Constant") {
            const auto data_beg = static_cast<char*>(a->get()->get_ptr());
            params[name] = std::string(data_beg, a->get()->size());
        }
    } else {
        IE_THROW() << "Error converting ngraph to CNN network. "
                              "Attribute adapter can not be found for " << name << " parameter";
    }
}

InferenceEngine::details::CNNLayerCreator::CNNLayerCreator(const std::shared_ptr<::ngraph::Node>& node): node(node) {
    addSpecificCreator({"Parameter"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                         const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "Input",
            details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<CNNLayer>(attrs);
        return res;
    });
    // TODO - Remove "GreaterEq" once ngraph transitions to GreaterEqual
    addSpecificCreator({"Eltwise", "Subtract", "Power", "Maximum", "Minimum", "Divide", "Greater", "GreaterEqual", "FloorMod", "LogicalOr",
                        "LogicalAnd", "LogicalXor", "GreaterEq", "Less", "LessEqual", "Equal", "NotEqual", "Multiply", "Add"},
                        [](const std::shared_ptr<::ngraph::Node>& node, const std::map<std::string, std::string>& params) -> CNNLayerPtr {
            LayerParams attrs = {node->get_friendly_name(), "Eltwise",
                details::convertPrecision(node->get_output_element_type(0))};
            auto res = std::make_shared<EltwiseLayer>(attrs);
            res->params = params;
            if (node->description() == "Maximum") {
                res->params["operation"] = "max";
            } else if (node->description() == "Minimum") {
                res->params["operation"] = "min";
            } else if (node->description() == "Power") {
                res->params["operation"] = "pow";
            } else if (node->description() == "Subtract") {
                res->params["operation"] = "sub";
            } else if (node->description() == "Divide") {
                res->params["operation"] = "div";
            } else if (node->description() == "LessEqual") {
                res->params["operation"] = "less_equal";
            } else if (node->description() == "Less") {
                res->params["operation"] = "less";
            } else if (node->description() == "Equal") {
                res->params["operation"] = "equal";
            } else if (node->description() == "NotEqual") {
                res->params["operation"] = "not_equal";
            } else if (node->description() == "FloorMod") {
                res->params["operation"] = "floor_mod";
            } else if (node->description() == "Multiply") {
                res->params["operation"] = "prod";
            } else if (node->description() == "Add") {
                res->params["operation"] = "sum";
            } else if (node->description() == "Greater") {
                res->params["operation"] = "greater";
            } else if (node->description() == "GreaterEq") {
                res->params["operation"] = "greater_equal";
            } else if (node->description() == "GreaterEqual") {
                res->params["operation"] = "greater_equal";
            } else if (node->description() == "LogicalOr") {
                res->params["operation"] = "logical_or";
            } else if (node->description() == "LogicalAnd") {
                res->params["operation"] = "logical_and";
            } else if (node->description() == "LogicalXor") {
                res->params["operation"] = "logical_xor";
            } else if (node->description() == "Eltwise") {
                auto castedLayer = std::dynamic_pointer_cast<::ngraph::op::Eltwise>(node);
                if (castedLayer == nullptr) IE_THROW() << "Cannot get " << attrs.type << " layer " << attrs.name;
                std::string type;
                switch (castedLayer->eltwise_type) {
                case ELTWISE_TYPE::Sum:
                    type = "sum";
                    break;
                case ELTWISE_TYPE::Sub:
                    type = "sub";
                    break;
                case ELTWISE_TYPE::Prod:
                    type = "prod";
                    break;
                default:
                    IE_THROW() << "Not supported eltwise type!";
                }

                res->params["operation"] = type;
            }
            return res;
        });
    addSpecificCreator({"Concat"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                      const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), node->description(),
                             details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<ConcatLayer>(attrs);
        res->params = params;
        auto axis = std::stoi(res->params["axis"]);
        res->params["axis"] = Builder::asString(axis < 0 ? axis + node->get_input_shape(0).size() : axis);
        return res;
    });
    addSpecificCreator({"AvgPool", "MaxPool"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                                  const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "Pooling",
            details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<PoolingLayer>(attrs);
        res->params = params;
        if (res->params.find("auto_pad") != res->params.end() &&
            details::CaselessEq<std::string>()(res->params["auto_pad"], "EXPLICIT"))
            res->params.erase("auto_pad");

        if (res->params.find("exclude_pad") != res->params.end()) {
            res->params["exclude-pad"] = res->params["exclude_pad"];
            res->params.erase("exclude_pad");
        }

        if (node->description() == "MaxPool") {
            res->params["pool-method"] = "max";
        } else if (node->description() == "AvgPool") {
            res->params["pool-method"] = "avg";
        }
        return res;
    });
    addSpecificCreator({"Select"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                      const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), node->description(),
                             details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<SelectLayer>(attrs);
        res->params = params;
        return res;
    });
    addSpecificCreator({"BinaryConvolution"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                      const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), node->description(),
                             details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<BinaryConvolutionLayer>(attrs);

        // todo: investigate difference between ngraph parameters for BinConvolution and the implementation above
        // this leads to accuracy issue for Precollected_ONNX_ResNet50_88percentinto1bit e2e test
        // res->params = params;

        auto castedLayer = ::ngraph::as_type_ptr<::ngraph::op::v1::BinaryConvolution>(node);
        IE_ASSERT(castedLayer) << " Operation " << node->description() << " with name "
            << node->get_friendly_name() << " cannot be casted to ngraph::op::v1::BinaryConvolution";

        std::string value;
        for (const auto& val : castedLayer->get_pads_begin()) {
            if (!value.empty()) value += ",";
            value += Builder::asString(val);
        }
        res->params["pads_begin"] = value;

        value.clear();
        for (const auto& val : castedLayer->get_pads_end()) {
            if (!value.empty()) value += ",";
            value += Builder::asString(val);
        }
        res->params["pads_end"] = value;

        switch (castedLayer->get_auto_pad()) {
            case ::ngraph::op::PadType::SAME_UPPER:
                res->params["auto_pad"] = "same_upper";
                break;
            case ::ngraph::op::PadType::SAME_LOWER:
                res->params["auto_pad"] = "same_lower";
                break;
            case ::ngraph::op::PadType::VALID:
                res->params["auto_pad"] = "valid";
                break;
            default:
                break;
        }

        value.clear();
        for (const auto& val : castedLayer->get_strides()) {
            if (!value.empty()) value += ",";
            value += Builder::asString(val);
        }
        res->params["strides"] = value;

        value.clear();
        for (const auto& val : castedLayer->get_dilations()) {
            if (!value.empty()) value += ",";
            value += Builder::asString(val);
        }
        res->params["dilations"] = value;

        // Restore kernel size and output
        const auto& shape = castedLayer->get_input_shape(1);
        res->params["output"] = Builder::asString(shape[0]);

        value.clear();
        for (size_t i = 2; i < shape.size(); i++) {
            if (!value.empty()) value += ",";
            value += Builder::asString(shape[i]);
        }
        res->params["kernel"] = value;

        switch (castedLayer->get_mode()) {
            case ::ngraph::op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT:
                res->params["mode"] = "xnor-popcount";
        }

        IE_ASSERT(castedLayer->input(1).get_partial_shape().is_static()) << " Weights for binary convolution "
            << castedLayer->get_friendly_name() << " should have static shapes!";
        auto weights_shape = castedLayer->input(1).get_source_output().get_shape();
        res->params["input"] = Builder::asString(weights_shape[1]);
        res->params["pad_value"] = Builder::asString(castedLayer->get_pad_value());

        const auto weightsNode = castedLayer->input(1).get_source_output().get_node_shared_ptr();
        InferenceEngine::details::addBlob(weightsNode, res, InferenceEngine::details::weights);

        return res;
    });

    addSpecificCreator({"SpaceToBatch"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                      const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), node->description(),
                             details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<SpaceToBatchLayer>(attrs);
        res->params = params;
        return res;
    });

    addSpecificCreator({"BatchToSpace"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                      const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), node->description(),
                             details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<BatchToSpaceLayer>(attrs);
        res->params = params;
        return res;
    });

    addSpecificCreator({"Assign"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                            const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "Memory",
                             details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<CNNLayer>(attrs);
        res->params["id"] = params.at("variable_id");
        res->params["index"] = "0";
        res->params["size"] = "2";
        return res;
    });

    addSpecificCreator({"ReadValue"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                            const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "Memory",
                             details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<CNNLayer>(attrs);
        res->params["id"] = params.at("variable_id");
        res->params["index"] = "1";
        res->params["size"] = "2";
        return res;
    });

    addSpecificCreator({"DepthToSpace"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                            const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), node->description(),
                             details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<DepthToSpaceLayer>(attrs);
        res->params = params;
        return res;
    });

    addSpecificCreator({"SpaceToDepth"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                            const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), node->description(),
                             details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<SpaceToDepthLayer>(attrs);
        res->params = params;
        return res;
    });

    addSpecificCreator({"DeconvolutionIE"},
                       [](const std::shared_ptr<::ngraph::Node> &node,
                          const std::map<std::string, std::string> &params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "Deconvolution",
                             details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<DeconvolutionLayer>(attrs);

        res->params = params;
        const auto& shape = node->get_input_shape(1);
        res->params["output"] = Builder::asString(shape[1]);
        std::string kernel_value;
        for (size_t i = 2; i < shape.size(); i++) {
            if (!kernel_value.empty()) kernel_value += ",";
            kernel_value += Builder::asString(shape[i]);
        }
        res->params["kernel"] = kernel_value;

        const auto weightsNode = node->input_value(1).get_node_shared_ptr();
        if (InferenceEngine::details::addBlob(weightsNode, res, InferenceEngine::details::weights)) {
            if (node->inputs().size() == 3) {
                const auto biasNode = node->input_value(2).get_node_shared_ptr();
                InferenceEngine::details::addBlob(biasNode, res, InferenceEngine::details::biases);
            }
        }
        return res;
    });

    addSpecificCreator({"DetectionOutput"},
                       [](const std::shared_ptr<::ngraph::Node> &node,
                          const std::map<std::string, std::string> &params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "DetectionOutput",
                            details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<InferenceEngine::CNNLayer>(attrs);
        res->params = params;

        if (res->params["code_type"] == "caffe.priorboxparameter.center_size") {
            res->params["code_type"] = "caffe.PriorBoxParameter.CENTER_SIZE";
        } else {
            res->params["code_type"] =  "caffe.PriorBoxParameter.CORNER";
        }
        res->params["variance_encoded_in_target"] = res->getBoolStrParamAsIntStr("variance_encoded_in_target");
        res->params["share_location"] = res->getBoolStrParamAsIntStr("share_location");
        res->params["clip_after_nms"] = res->getBoolStrParamAsIntStr("clip_after_nms");
        res->params["clip_before_nms"] = res->getBoolStrParamAsIntStr("clip_before_nms");
        res->params["decrease_label_id"] = res->getBoolStrParamAsIntStr("decrease_label_id");
        res->params["normalized"] = res->getBoolStrParamAsIntStr("normalized");
        return res;
    });

    addSpecificCreator({"LogicalNot"},
                       [](const std::shared_ptr<::ngraph::Node>& node,
                          const std::map<std::string, std::string> params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "Activation",
                              details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<InferenceEngine::CNNLayer>(attrs);
        res->params["type"] = "not";
        return res;
    });

    addSpecificCreator({"LSTMCellIE"},
                        [](const std::shared_ptr<::ngraph::Node>& node,
                           const std::map<std::string, std::string> params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "LSTMCell",
                             details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<LSTMCell>(attrs);
        res->params = params;
        const auto weightsNode = node->input_value(3).get_node_shared_ptr();
        InferenceEngine::details::addBlob(weightsNode, res, InferenceEngine::details::weights);

        const auto biasNode = node->input_value(4).get_node_shared_ptr();
        InferenceEngine::details::addBlob(biasNode, res, InferenceEngine::details::biases);

        return res;
    });

    addSpecificCreator({"RNNCellIE"},
                       [](const std::shared_ptr<::ngraph::Node>& node,
                          const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "RNNCell",
                             details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<RNNCell>(attrs);
        res->params = params;

        const auto weightsNode = node->input_value(2).get_node_shared_ptr();
        InferenceEngine::details::addBlob(weightsNode, res, InferenceEngine::details::weights);

        const auto biasNode = node->input_value(3).get_node_shared_ptr();
        InferenceEngine::details::addBlob(biasNode, res, InferenceEngine::details::biases);

        return res;
    });

    addSpecificCreator({"GRUCellIE"},
                       [](const std::shared_ptr<::ngraph::Node>& node,
                          const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "GRUCell",
                             details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<GRUCell>(attrs);
        res->params = params;

        const auto weightsNode = node->input_value(2).get_node_shared_ptr();
        InferenceEngine::details::addBlob(weightsNode, res, InferenceEngine::details::weights);

        const auto biasNode = node->input_value(3).get_node_shared_ptr();
        InferenceEngine::details::addBlob(biasNode, res, InferenceEngine::details::biases);

        return res;
    });

    addSpecificCreator({"PRelu"},
                       [](const std::shared_ptr<::ngraph::Node>& node,
                          const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "PReLU",
                             details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<PReLULayer>(attrs);
        res->params = params;

        const auto weightsNode = node->input_value(1).get_node_shared_ptr();
        InferenceEngine::details::addBlob(weightsNode, res, InferenceEngine::details::weights);

        return res;
    });

    addSpecificCreator({"TileIE"},
                       [](const std::shared_ptr<::ngraph::Node>& node,
                          const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "Tile",
            details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<TileLayer>(attrs);
        res->params = params;
        return res;
    });

    addSpecificCreator({"PriorBoxIE"},
                       [](const std::shared_ptr<::ngraph::Node>& node,
                          const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "PriorBox",
            details::convertPrecision(node->get_output_element_type(0))};

        auto res = std::make_shared<CNNLayer>(attrs);
        res->params = params;
        res->params["clip"] = res->getBoolStrParamAsIntStr("clip");
        res->params["flip"] = res->getBoolStrParamAsIntStr("flip");
        res->params["scale_all_sizes"] = res->getBoolStrParamAsIntStr("scale_all_sizes");

        auto scale_all_sizes = std::stoi(res->params["scale_all_sizes"]);
        if (!scale_all_sizes) {
            auto data_pshape = node->get_input_partial_shape(0);
            if (data_pshape.is_dynamic()) IE_THROW() << "Dynamic 0-port input of PriorBox is not supported";
            auto data_shape = data_pshape.to_shape();
            if (data_shape.size() != 4) IE_THROW() << "PriorBox has " << data_shape.size() << " items in 0-port input, 4 expected";
            auto img_pshape = node->get_input_partial_shape(1);
            if (img_pshape.is_dynamic()) IE_THROW() << "Dynamic 1-port input of PriorBox is not supported";
            auto img_shape = img_pshape.to_shape();
            if (img_shape.size() != 4) IE_THROW() << "PriorBox has " << data_shape.size() << " items in 1-port input, 4 expected";

            // mxnet-like PriorBox
            auto img_H = img_shape[2];
            auto data_H = data_shape[2];

            auto step = std::stof(res->params["step"]);
            if (step == -1)
                step = img_H / static_cast<float>(data_H);
            else
                step *= img_H;
            res->params["step"] = Builder::asString(step);

            auto min_size = details::split(res->params["min_size"], ",");
            for (auto &size : min_size) {
                size = Builder::asString(std::stof(size) * img_H);
            }
            res->params["min_size"] = details::joinVec(min_size);
        }
        return res;
    });

    addSpecificCreator({"PriorBoxClusteredIE"},
                       [](const std::shared_ptr<::ngraph::Node>& node,
                          const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "PriorBoxClustered",
            details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<CNNLayer>(attrs);
        res->params = params;
        res->params["clip"] =
            res->getBoolStrParamAsIntStr("clip");

        auto step_h = std::stof(res->params["step_h"]);
        auto step_w = std::stof(res->params["step_w"]);
        if (std::abs(step_h - step_w) < 1e-5) {
            res->params["step"] = res->params["step_w"];
        }
        return res;
    });

    addSpecificCreator({"ProposalIE"},
                       [](const std::shared_ptr<::ngraph::Node>& node,
                          const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "Proposal",
            details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<CNNLayer>(attrs);
        res->params = params;
        res->params["clip_before_nms"] =
            res->getBoolStrParamAsIntStr("clip_before_nms");
        res->params["clip_after_nms"] =
            res->getBoolStrParamAsIntStr("clip_after_nms");
        res->params["normalize"] = res->getBoolStrParamAsIntStr("normalize");
        return res;
    });

    addSpecificCreator({"Relu", "ReLUIE"},
                       [](const std::shared_ptr<::ngraph::Node>& node,
                          const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "ReLU",
            details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<ReLULayer>(attrs);
        res->params = params;
        return res;
    });

    addSpecificCreator({"Reshape"},
                       [](const std::shared_ptr<::ngraph::Node>& node,
                          const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "Reshape",
            details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<ReshapeLayer>(attrs);
        return res;
    });

    addSpecificCreator({"ReverseSequence"},
                       [](const std::shared_ptr<::ngraph::Node>& node,
                          const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "ReverseSequence",
            details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<ReverseSequenceLayer>(attrs);
        res->params = params;
        return res;
    });

    addSpecificCreator({"SeluIE"},
                       [](const std::shared_ptr<::ngraph::Node>& node,
                          const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "Selu",
            details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<CNNLayer>(attrs);
        res->params = params;
        return res;
    });

    addSpecificCreator({"Softmax"},
                       [](const std::shared_ptr<::ngraph::Node>& node,
                          const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "SoftMax",
            details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<SoftMaxLayer>(attrs);
        res->params = params;
        return res;
    });

    addSpecificCreator({"Split"},
                       [](const std::shared_ptr<::ngraph::Node>& node,
                          const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "Split",
            details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<SplitLayer>(attrs);

        auto axis_node = node->input_value(1).get_node_shared_ptr();
        const auto axis_node_const = std::dynamic_pointer_cast<ngraph::op::Constant>(axis_node);
        if (!axis_node_const) {
            IE_THROW() << "Split " << node->get_friendly_name() << " has no axes as Constant";
        }
        auto axis = axis_node_const->cast_vector<int64_t>()[0];
        if (axis < 0) {
            axis += node->get_input_shape(0).size();
        }
        res->params["axis"] = Builder::asString(axis);

        return res;
    });

    addSpecificCreator({"Tanh"},
                       [](const std::shared_ptr<::ngraph::Node>& node,
                          const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "TanH",
            details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<CNNLayer>(attrs);
        res->params = params;
        return res;
    });

    addSpecificCreator({"ScatterElementsUpdate"}, [](const std::shared_ptr<::ngraph::Node>& node,
        const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), node->description(),
            details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<ScatterElementsUpdateLayer>(attrs);
        res->params = params;
        return res;
    });

    addSpecificCreator({"ScatterUpdate"}, [](const std::shared_ptr<::ngraph::Node>& node,
        const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), node->description(),
            details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<ScatterUpdateLayer>(attrs);
        res->params = params;
        return res;
    });

    addSpecificCreator({"StaticShapeTopK"}, [](const std::shared_ptr<::ngraph::Node>& node,
        const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "TopK",
            details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<TopKLayer>(attrs);
        res->params = params;
        return res;
    });

    addSpecificCreator({"StridedSlice"}, [](const std::shared_ptr<::ngraph::Node> &node,
        const std::map<std::string, std::string> &params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "StridedSlice",
            details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<InferenceEngine::StridedSliceLayer>(attrs);
        auto stridedSliceInvertMaskStr = [](const std::string& str) -> std::string {
            std::string value;
            auto found_numbers = details::split(str, ",");
            for (const auto &val : found_numbers) {
                if (!value.empty())
                    value += ",";
                value += Builder::asString((1 - std::stoi(val)));
            }
            return value;
        };

        res->params = params;
        // plugins require reversed value of begin_mask and end_mask
        res->params["begin_mask"] = stridedSliceInvertMaskStr(res->params["begin_mask"]);
        res->params["end_mask"] = stridedSliceInvertMaskStr(res->params["end_mask"]);

        return res;
    });

    addSpecificCreator({"TopK", "TopKIE"}, [](const std::shared_ptr<::ngraph::Node>& node,
        const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "TopK",
                          details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<InferenceEngine::TopKLayer>(attrs);
        res->params = params;
        return res;
    });

    addSpecificCreator({"Transpose"}, [](const std::shared_ptr<::ngraph::Node>& node,
        const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "Permute",
            details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<InferenceEngine::CNNLayer>(attrs);
        res->params = params;
        if (auto transpose_const = std::dynamic_pointer_cast<ngraph::op::Constant>(node->input_value(1).get_node_shared_ptr())) {
            res->params["order"] = Builder::asString(transpose_const->cast_vector<int64_t>());
        }
        return res;
    });

    addSpecificCreator({"SwishIE"}, [](const std::shared_ptr<::ngraph::Node>& node,
        const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "Swish",
            details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<InferenceEngine::CNNLayer>(attrs);
        res->params = params;
        return res;
    });

    addSpecificCreator({"NonMaxSuppressionIE3"}, [](const std::shared_ptr<::ngraph::Node>& node,
        const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "NonMaxSuppression",
            details::convertPrecision(node->get_output_element_type(0))};

        auto castedLayer = ::ngraph::as_type_ptr<::ngraph::op::NonMaxSuppressionIE3>(node);
        IE_ASSERT(castedLayer) << " Operation " << node->description() << " with name "
            << node->get_friendly_name() << " cannot be casted to ngraph::op::NonMaxSuppressionIE3";

        auto res = std::make_shared<InferenceEngine::NonMaxSuppressionLayer>(attrs);
        res->params = params;

        res->params["center_point_box"] = castedLayer->m_center_point_box ? "true" : "false";
        res->params["sort_result_descending"] = castedLayer->m_sort_result_descending ? "true" : "false";

        auto output_type = details::convertPrecision(castedLayer->m_output_type);
        std::string output_type_str;
        switch (output_type) {
        case Precision::I32:
            output_type_str = "I32";
            break;
        case Precision::I64:
            output_type_str = "I64";
            break;
        default:
            IE_THROW() << "Unsupported output type";
        }
        res->params["output_type"] = output_type_str;

        return res;
    });

    addSpecificCreator({"NonMaxSuppression"}, [](const std::shared_ptr<::ngraph::Node>& node,
        const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "NonMaxSuppression",
            details::convertPrecision(node->get_output_element_type(0))};

        auto castedLayer = ::ngraph::as_type_ptr<::ngraph::op::v5::NonMaxSuppression>(node);
        IE_ASSERT(castedLayer) << " Operation " << node->description() << " with name "
            << node->get_friendly_name() << " cannot be casted to ngraph::op::v5::NonMaxSuppression";

        auto res = std::make_shared<InferenceEngine::NonMaxSuppressionLayer>(attrs);
        res->params = params;

        auto box_encoding = castedLayer->get_box_encoding();
        switch (box_encoding) {
            case ngraph::op::v5::NonMaxSuppression::BoxEncodingType::CORNER:
                res->params["center_point_box"] = "false";
                break;
            case ngraph::op::v5::NonMaxSuppression::BoxEncodingType::CENTER:
                res->params["center_point_box"] = "true";
                break;
            default:
                IE_THROW() << "Unsupported box encoding for NonMaxSuppression op";
                break;
        }

        auto output_type = details::convertPrecision(castedLayer->get_output_type());
        std::string output_type_str;
        switch (output_type) {
        case Precision::I32:
            output_type_str = "I32";
            break;
        case Precision::I64:
            output_type_str = "I64";
            break;
        default:
            IE_THROW() << "Unsupported output type";
        }
        res->params["output_type"] = output_type_str;

        bool sort_result_descending = castedLayer->get_sort_result_descending();
        res->params["sort_result_descending"] = sort_result_descending ? "true" : "false";

        return res;
    });

    addSpecificCreator({"NonMaxSuppressionIE"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                                 const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "NonMaxSuppression", details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<InferenceEngine::NonMaxSuppressionLayer>(attrs);
        res->params = params;
        return res;
    });

    addSpecificCreator({"GRUSequenceIE"}, [](const std::shared_ptr<::ngraph::Node>& node,
                    const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "GRUSequence",
                             details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<RNNSequenceLayer>(attrs);
        res->params = params;
        res->axis = std::stoi(res->params["axis"]);
        if (res->params["direction"] == "reverse")
            res->params["direction"] = "Backward";
        else if (res->params["direction"] == "forward")
            res->params["direction"] = "Forward";
        else
            res->params["direction"] = "Bidirectional";

        res->cellType = RNNSequenceLayer::CellType::GRU;
        if (res->params["linear_before_reset"] == "true") {
            res->cellType = RNNSequenceLayer::CellType::GRU_LBR;
        }

        const auto weightsNode = node->input_value(3).get_node_shared_ptr();
        InferenceEngine::details::addBlob(weightsNode, res, InferenceEngine::details::weights);

        const auto biasNode = node->input_value(4).get_node_shared_ptr();
        InferenceEngine::details::addBlob(biasNode, res, InferenceEngine::details::biases);

        return res;
    });

    addSpecificCreator({"RNNSequenceIE"}, [](const std::shared_ptr<::ngraph::Node>& node,
                    const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "RNNSequence",
                             details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<RNNSequenceLayer>(attrs);
        res->params = params;

        res->cellType = RNNSequenceLayer::CellType::RNN;
        res->axis = std::stoi(res->params["axis"]);
        if (res->params["direction"] == "reverse")
            res->params["direction"] = "Backward";
        else if (res->params["direction"] == "forward")
            res->params["direction"] = "Forward";
        else
            res->params["direction"] = "Bidirectional";

        const auto weightsNode = node->input_value(3).get_node_shared_ptr();
        InferenceEngine::details::addBlob(weightsNode, res, InferenceEngine::details::weights);

        const auto biasNode = node->input_value(4).get_node_shared_ptr();
        InferenceEngine::details::addBlob(biasNode, res, InferenceEngine::details::biases);

        return res;
    });

    addSpecificCreator({"LSTMSequenceIE"}, [](const std::shared_ptr<::ngraph::Node>& node,
                    const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "LSTMSequence",
                             details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<RNNSequenceLayer>(attrs);
        res->params = params;

        res->cellType = RNNSequenceLayer::CellType::LSTM;
        res->axis = std::stoi(res->params["axis"]);
        if (res->params["direction"] == "reverse")
            res->params["direction"] = "Backward";
        else if (res->params["direction"] == "forward")
            res->params["direction"] = "Forward";
        else
            res->params["direction"] = "Bidirectional";

        const auto weightsNode = node->input_value(4).get_node_shared_ptr();
        InferenceEngine::details::addBlob(weightsNode, res, InferenceEngine::details::weights);

        const auto biasNode = node->input_value(5).get_node_shared_ptr();
        InferenceEngine::details::addBlob(biasNode, res, InferenceEngine::details::biases);

        return res;
    });

    REQUIRED_IE_CONVERSION_CREATOR("Broadcast", "Tile");
    REQUIRED_IE_CONVERSION_CREATOR("Interpolate", "Interp");
    REQUIRED_IE_CONVERSION_CREATOR("NormalizeL2", "NormalizeIE");
    REQUIRED_IE_CONVERSION_CREATOR("GroupConvolution", "ConvolutionIE");
    REQUIRED_IE_CONVERSION_CREATOR("GroupConvolutionBackpropData", "DeconvolutionIE");

    addSpecificCreator({ "Convolution", "GatherTree", "GRUCell", "GRUSequence", "HardSigmoid",
                      "LRN", "LSTMCell", "LSTMSequence", "NonMaxSuppression", "RNNCell", "RNNSequence", "OneHot",
                      "Pad", "PriorBoxClustered", "PriorBox", "Proposal", "Selu", "Swish", "Tile"},
            [](const std::shared_ptr<::ngraph::Node>& node, const std::map<std::string, std::string>& params)
            -> CNNLayerPtr {
        const std::string& type_name = node->get_type_name();
        IE_THROW() << type_name << " operation has a form that is not supported. " << node->get_friendly_name()
                           << " should be converted to " << type_name + "IE operation.";
        return nullptr;
    });

    addSpecificCreator({"ReduceMin", "ReduceMax", "ReduceMean", "ReduceProd", "ReduceSum", "ReduceL1", "ReduceL2"},
                       [](const std::shared_ptr<::ngraph::Node>& node, const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), node->description(), details::convertPrecision(node->get_output_element_type(0))};
        auto reduce_node = std::dynamic_pointer_cast<ngraph::op::util::ArithmeticReductionKeepDims>(node);
        if (reduce_node == nullptr)
            IE_THROW() << "Node '" << node->get_name() << "' is not an instance of ArithmeticReductionKeepDims.";
        auto res = std::make_shared<InferenceEngine::ReduceLayer>(attrs);
        res->params = params;
        res->params["keep_dims"] = reduce_node->get_keep_dims() ? "True" : "False";
        return res;
    });

    addSpecificCreator({"ReduceLogicalAnd"}, [](const std::shared_ptr<::ngraph::Node>& node, const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "ReduceAnd", details::convertPrecision(node->get_output_element_type(0))};
        auto reduce_node = std::dynamic_pointer_cast<ngraph::op::util::LogicalReductionKeepDims>(node);
        if (reduce_node == nullptr)
            IE_THROW() << "Node '" << node->get_name() << "' is not an instance of LogicalReductionKeepDims.";
        auto res = std::make_shared<InferenceEngine::ReduceLayer>(attrs);
        res->params = params;
        res->params["keep_dims"] = reduce_node->get_keep_dims() ? "True" : "False";
        return res;
    });

    addSpecificCreator({"ReduceLogicalOr"}, [](const std::shared_ptr<::ngraph::Node>& node, const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "ReduceOr", details::convertPrecision(node->get_output_element_type(0))};
        auto reduce_node = std::dynamic_pointer_cast<ngraph::op::util::LogicalReductionKeepDims>(node);
        if (reduce_node == nullptr)
            IE_THROW() << "Node '" << node->get_name() << "' is not an instance of LogicalReductionKeepDims.";
        auto res = std::make_shared<InferenceEngine::ReduceLayer>(attrs);
        res->params = params;
        res->params["keep_dims"] = reduce_node->get_keep_dims() ? "True" : "False";
        return res;
    });

    addSpecificCreator({"Constant"}, [](const std::shared_ptr<::ngraph::Node>& node, const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "Const", details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<InferenceEngine::CNNLayer>(attrs);
        auto castedLayer = ngraph::as_type_ptr<ngraph::op::Constant>(node);
        if (!res) IE_THROW() << "Cannot get " << attrs.type << " layer " << attrs.name;

        res->blobs["custom"] = InferenceEngine::details::shareWeights(castedLayer);

        return res;
    });

    addSpecificCreator({"Convert"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                       const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "Convert",
                             details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<InferenceEngine::CNNLayer>(attrs);

        auto p = details::convertPrecision(node->get_output_element_type(0));
        std::string precision_str;
        switch (p) {
        case Precision::FP16:
            precision_str = "FP16";
            break;
        case Precision::BF16:
            precision_str = "BF16";
            break;
        case Precision::FP32:
            precision_str = "FP32";
            break;
        case Precision::FP64:
            precision_str = "FP64";
            break;
        case Precision::I8:
            precision_str = "I8";
            break;
        case Precision::I16:
            precision_str = "I16";
            break;
        case Precision::I32:
            precision_str = "I32";
            break;
        case Precision::I64:
            precision_str = "I64";
            break;
        case Precision::U8:
            precision_str = "U8";
            break;
        case Precision::U16:
            precision_str = "U16";
            break;
        case Precision::U32:
            precision_str = "U32";
            break;
        case Precision::U64:
            precision_str = "U64";
            break;
        case Precision::BOOL:
            precision_str = "BOOL";
            break;
        default:
            IE_THROW() << "Unsupported type";
        }

        res->params["precision"] = precision_str;
        return res;
    });

    addSpecificCreator({"MVN"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                   const std::map<std::string, std::string> &params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "MVN",
                            details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<InferenceEngine::MVNLayer>(attrs);

        res->params["normalize_variance"] = params.at("normalize_variance");
        res->params["normalize_variance"] = res->getBoolStrParamAsIntStr("normalize_variance");
        res->params["eps"] = params.at("eps");
        const auto& acrossChannelsIt = params.find("across_channels");
        if (acrossChannelsIt != params.end()) {
            res->params["across_channels"] = params.at("across_channels");
            res->params["across_channels"] = res->getBoolStrParamAsIntStr("across_channels");
        }
        return res;
    });

    addSpecificCreator({"NormalizeIE"}, [](const std::shared_ptr<::ngraph::Node> &node,
                                           const std::map<std::string, std::string> &params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "Normalize",
                             details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<InferenceEngine::NormLayer>(attrs);

        res->params = params;
        res->params["channel_shared"] = res->getBoolStrParamAsIntStr("channel_shared");
        res->params["across_spatial"] = res->getBoolStrParamAsIntStr("across_spatial");

        const auto weightsNode = node->input_value(1).get_node_shared_ptr();
        if (auto castedLayer = ngraph::as_type_ptr<ngraph::op::Constant>(weightsNode)) {
            res->blobs["weights"] = InferenceEngine::details::shareWeights(castedLayer);
        }
        return res;
    });

    addSpecificCreator({"Clamp"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                     const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "Clamp", details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<InferenceEngine::ClampLayer>(attrs);
        res->params = params;
        return res;
    });

    addSpecificCreator({"LRN_IE"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                      const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "Norm", details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<InferenceEngine::NormLayer>(attrs);
        res->params = params;
        return res;
    });

    addSpecificCreator({"Elu"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                   const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "elu", details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<InferenceEngine::CNNLayer>(attrs);
        res->params = params;
        return res;
    });

    addSpecificCreator({"MatMul"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                      const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "Gemm", details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<InferenceEngine::GemmLayer>(attrs);
        res->params = params;
        return res;
    });

    addSpecificCreator({"GatherIE"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                        const std::map<std::string, std::string>& params) ->CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "Gather", details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<InferenceEngine::GatherLayer>(attrs);

        auto castedLayer = std::dynamic_pointer_cast<ngraph::op::GatherIE>(node);
        if (castedLayer == nullptr) IE_THROW() << "Cannot get " << attrs.type << " layer " << attrs.name;

        res->params["axis"] = Builder::asString(castedLayer->get_axis());

        return res;
    });

    addSpecificCreator({"GatherTreeIE"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                            const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "GatherTree", details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<InferenceEngine::CNNLayer>(attrs);
        return res;
    });

    addSpecificCreator({"GRN"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                   const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "GRN", details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<InferenceEngine::GRNLayer>(attrs);
        res->params = params;
        return res;
    });

    addSpecificCreator({"OneHotIE"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                        const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "OneHot", details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<InferenceEngine::OneHotLayer>(attrs);
        res->params = params;
        return res;
    });

    addSpecificCreator({"HardSigmoid_IE"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                              const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "HardSigmoid", details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<InferenceEngine::CNNLayer>(attrs);

        auto castedLayer = std::dynamic_pointer_cast<ngraph::op::HardSigmoid_IE>(node);
        if (!castedLayer)
            IE_THROW() << "Cannot get " << attrs.type << " layer " << attrs.name;

        res->params["alpha"] = Builder::asString(castedLayer->get_alpha());
        res->params["beta"] = Builder::asString(castedLayer->get_beta());
        return res;
    });

    addSpecificCreator({"Interp"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                      const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "Interp", details::convertPrecision(node->get_output_element_type(0))};
        auto castedLayer = std::dynamic_pointer_cast<ngraph::op::Interp>(node);
        if (!castedLayer) IE_THROW() << "Cannot get " << attrs.type << " layer " << attrs.name;

        auto interp_attrs = castedLayer->get_attrs();

        if (interp_attrs.antialias) {
            IE_THROW() << "Interp do not support antialias";
        }
        if (interp_attrs.mode != "linear") {
            IE_THROW() << "Interp do not support mode '" << interp_attrs.mode << "'";
        }

        auto res = std::make_shared<InferenceEngine::CNNLayer>(attrs);
        res->params = params;
        res->params["align_corners"] = interp_attrs.align_corners ? "1" : "0";
        return res;
    });

    addSpecificCreator({"PadIE"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                     const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "Pad", details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<InferenceEngine::PadLayer>(attrs);

        res->params["pad_mode"] = params.at("pad_mode");
        res->params["pads_begin"] = params.at("pads_begin");
        res->params["pads_end"] = params.at("pads_end");

        if (params.at("pad_mode") == "constant") {
            res->params["pad_value"] = params.at("pad_value");
        }

        return res;
    });

    addSpecificCreator({"Subtract"}, [](const std::shared_ptr<::ngraph::Node> &node,
                                        const std::map<std::string, std::string> &params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "Eltwise",
                             details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<InferenceEngine::EltwiseLayer>(attrs);
        res->params["operation"] = "sub";
        return res;
    });

    addSpecificCreator({"FakeQuantize"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                            const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "FakeQuantize", details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<InferenceEngine::QuantizeLayer>(attrs);
        res->params = params;
        return res;
    });

    addSpecificCreator({"ConvolutionIE"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                             const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "Convolution", details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<InferenceEngine::ConvolutionLayer>(attrs);
        res->params = params;

        auto && rt_info = node->get_rt_info();
        bool keep_constants(false);
        if (auto attr = std::dynamic_pointer_cast<ngraph::VariantWrapper<int64_t>>(rt_info["keep_constants"])) {
            keep_constants = attr->get();
        }

        // Restore output and kernel size
        auto shape = node->get_input_shape(1);
        shape.erase(shape.begin(), shape.begin() + 2);

        res->params["kernel"] = Builder::asString(static_cast<std::vector<size_t>&>(shape));
        res->params["output"] = Builder::asString(node->get_shape()[1]);

        // forward auto_pad only when its value is different than explicit
        if (params.at("auto_pad") == "explicit") {
            res->params.erase("auto_pad");
        }

        const auto weightsNode = node->input_value(1).get_node_shared_ptr();
        if (!keep_constants && InferenceEngine::details::addBlob(weightsNode, res, InferenceEngine::details::weights)) {
            if (node->inputs().size() == 3) {
                const auto biasNode = node->input_value(2).get_node_shared_ptr();
                InferenceEngine::details::addBlob(biasNode, res, InferenceEngine::details::biases);
            }
        }
        return res;
    });

    addSpecificCreator({"DeformableConvolution"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                                     const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "DeformableConvolution", details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<InferenceEngine::DeformableConvolutionLayer>(attrs);

        res->params = params;

        auto shape = node->get_input_shape(2);
        std::string value;

        res->params["output"] = Builder::asString(shape[0]);

        for (size_t i = 2; i < shape.size(); i++) {
            if (!value.empty()) value += ",";
            value += Builder::asString(shape[i]);
        }
        res->params["kernel"] = value;

        if (res->params["auto_pad"] == "explicit") {
            res->params.erase("auto_pad");
        }

        const auto weightsNode = node->input_value(2).get_node_shared_ptr();
        InferenceEngine::details::addBlob(weightsNode, res, InferenceEngine::details::weights);

        return res;
    });

    addSpecificCreator({"DeformablePSROIPooling"}, [](const std::shared_ptr<::ngraph::Node> &node,
                                                      const std::map<std::string, std::string> &params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "PSROIPooling", details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<InferenceEngine::CNNLayer>(attrs);
        res->params = params;
        res->params["no_trans"] = node->get_input_size() == 2 ? "1" : "0";
        // v1::DeformablePRSOIPooling treats group_size attribute as pooled sizes
        res->params["pooled_height"] = params.at("group_size");
        res->params["pooled_width"] = params.at("group_size");
        return res;
    });

    addSpecificCreator({"CTCGreedyDecoder"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                                const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "CTCGreedyDecoder", details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<InferenceEngine::CNNLayer>(attrs);
        res->params = params;
        res->params["ctc_merge_repeated"] = res->getBoolStrParamAsIntStr("ctc_merge_repeated");
        return res;
    });

    addSpecificCreator({"TensorIterator", "StaticShapeLoop"},
        [](const std::shared_ptr<::ngraph::Node>& node, const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        auto res = createSubGraphLayer(node);
        res->type = "TensorIterator";
        return res;
    });

    addSpecificCreator({"Loop"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                    const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        auto res = createSubGraphLayer(node);
        res->type = "Loop";
        return res;
    });

    addSpecificCreator({"SquaredDifference"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                                 const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "Eltwise", details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<InferenceEngine::EltwiseLayer>(attrs);
        res->params["operation"] = "squared_diff";
        return res;
    });

    addSpecificCreator({"RegionYolo"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                          const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "RegionYolo", details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<InferenceEngine::CNNLayer>(attrs);
        res->params = params;
        res->params["do_softmax"] = res->getBoolStrParamAsIntStr("do_softmax");
        return res;
    });

    addSpecificCreator({"VariadicSplit"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                             const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "Split", details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<InferenceEngine::SplitLayer>(attrs);
        auto castedLayer = std::dynamic_pointer_cast<ngraph::op::VariadicSplit>(node);
        if (!castedLayer) IE_THROW() << "Cannot get " << attrs.type << " layer " << attrs.name;

        auto axis_node = castedLayer->input_value(1).get_node_shared_ptr();
        const auto axis_node_const = ngraph::as_type_ptr<ngraph::op::Constant>(axis_node);
        if (!axis_node_const) {
            IE_THROW() << "Split " << castedLayer->get_friendly_name() << " has no axes as Constant";
        }

        auto axis = axis_node_const->cast_vector<int64_t>()[0];
        if (axis < 0) {
            axis += castedLayer->get_input_shape(0).size();
        }

        res->params["axis"] = Builder::asString(axis);
        return res;
    });

    addSpecificCreator({"Interpolate"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                           const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "Interpolate", details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<InferenceEngine::CNNLayer>(attrs);
        res->params = params;
        return res;
    });

    addSpecificCreator({"CropIE"}, [](const std::shared_ptr<::ngraph::Node> &node,
                                      const std::map<std::string, std::string> &params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "Crop", details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<InferenceEngine::CropLayer>(attrs);
        res->params = params;
        return res;
    });

    addSpecificCreator({"ScaleShiftIE"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                            const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "ScaleShift", details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<InferenceEngine::ScaleShiftLayer>(attrs);
        res->params = params;
        const auto weightsNode = node->input_value(1).get_node_shared_ptr();
        InferenceEngine::details::addBlob(weightsNode, res, InferenceEngine::details::weights);
        const auto biasNode = node->input_value(2).get_node_shared_ptr();
        InferenceEngine::details::addBlob(biasNode, res, InferenceEngine::details::biases);
        return res;
    });

    addSpecificCreator({"ExecutionNode"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                             const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        auto& rtInfo = node->get_rt_info();

        if (rtInfo.count(ExecGraphInfoSerialization::LAYER_TYPE) == 0) {
            IE_THROW() << "No " << ExecGraphInfoSerialization::LAYER_TYPE
                << " attribute is set in " << node->get_friendly_name() << " node";
        }

        auto getStringValue = [] (const std::shared_ptr<ngraph::Variant> & variant) {
            auto castedVariant = std::dynamic_pointer_cast<ngraph::VariantImpl<std::string>>(variant);
            IE_ASSERT(castedVariant != nullptr);
            return castedVariant->get();
        };

        LayerParams attrs = {node->get_friendly_name(),
                            getStringValue(rtInfo[ExecGraphInfoSerialization::LAYER_TYPE]),
                            details::convertPrecision(node->get_output_element_type(0))};
        rtInfo.erase(ExecGraphInfoSerialization::LAYER_TYPE);

        auto res = std::make_shared<InferenceEngine::CNNLayer>(attrs);
        res->params = params;

        for (const auto & kvp : rtInfo) {
            auto castedVariant = std::dynamic_pointer_cast<ngraph::VariantImpl<std::string>>(kvp.second);
            // skip RT info which holds fusedNames, etc
            if (castedVariant)
                res->params[kvp.first] = getStringValue(castedVariant);
        }

        return res;
    });

    addSpecificCreator({"ResampleV2"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                          const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "Resample", details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<InferenceEngine::CNNLayer>(attrs);
        res->params = params;

        res->params["antialias"] = res->getBoolStrParamAsIntStr("antialias");
        if (res->params["type"] == "nearest") {
            res->params["type"] = "caffe.ResampleParameter.NEAREST";
        } else if (res->params["type"] == "cubic") {
            res->params["type"] = "caffe.ResampleParameter.CUBIC";
        } else if (res->params["type"] == "area") {
            res->params["type"] = "caffe.ResampleParameter.AREA";
        } else if (res->params["type"] == "linear") {
            res->params["type"] = "caffe.ResampleParameter.LINEAR";
        }
        return res;
    });

    addSpecificCreator({"FullyConnected"}, [](const std::shared_ptr<::ngraph::Node> &node,
                                              const std::map<std::string, std::string> &params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "FullyConnected", details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<InferenceEngine::FullyConnectedLayer>(attrs);
        res->params = params;

        auto & rt_info = node->get_rt_info();
        bool keep_constants(false);
        if (auto attr = std::dynamic_pointer_cast<ngraph::VariantWrapper<int64_t>>(rt_info["keep_constants"])) {
            keep_constants = attr->get();
        }
        const auto weightsNode = node->input_value(1).get_node_shared_ptr();
        if (!keep_constants && InferenceEngine::details::addBlob(weightsNode, res, InferenceEngine::details::weights)) {
            const auto biasNode = node->input_value(2).get_node_shared_ptr();
            InferenceEngine::details::addBlob(biasNode, res, InferenceEngine::details::biases);
        }
        return res;
    });

    addSpecificCreator({"ShuffleChannels"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                               const std::map<std::string, std::string>& params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "ShuffleChannels", details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<InferenceEngine::ShuffleChannelsLayer>(attrs);
        res->params = params;
        return res;
    });

    addSpecificCreator({"PowerIE"}, [](const std::shared_ptr<::ngraph::Node> &node,
                                       const std::map<std::string, std::string> &params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "Power", details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<InferenceEngine::PowerLayer>(attrs);

        auto castedLayer = ngraph::as_type_ptr<ngraph::op::PowerIE>(node);
        if (castedLayer == nullptr) IE_THROW() << "Cannot get " << attrs.type << " layer " << attrs.name;
        res->params = params;
        // This is needed as scale parameter requires high precision
        res->params["scale"] = Builder::asString(castedLayer->scale);
        return res;
    });
}

CNNLayerPtr InferenceEngine::details::CNNLayerCreator::create() {
    LayerParams attrs = {node->get_friendly_name(), node->description(),
                         details::convertPrecision(node->get_output_element_type(0))};
    if (creators.find(node->description()) != creators.end())
        return creators[node->description()](node, params);

    auto res = std::make_shared<CNNLayer>(attrs);
    res->params = params;
    return res;
}

void convertFunctionToICNNNetwork(const std::shared_ptr<const ::ngraph::Function> &graph,
                                  const CNNNetwork &network,
                                  CNNNetworkImpl* cnnNetworkImpl,
                                  bool keep_constant_inputs) {
    OV_ITT_SCOPED_TASK(itt::domains::IELegacy, "details::convertFunctionToICNNNetwork");

    const auto createCNNLayer = [](const std::shared_ptr<::ngraph::Node> &node) -> CNNLayerPtr {
        class NGraphCNNLayer: public CNNLayer {
        public:
            void setNode(const std::shared_ptr<::ngraph::Node>& node) {
                this->node = node;
            }
        };
        CNNLayerPtr result;

        CNNLayerCreator visitor(node);
        if (node->visit_attributes(visitor)) {
            result = visitor.create();
        }

        if (!result)
            IE_THROW() << "Cannot cast ngraph node " << node->get_friendly_name() << " to CNNLayer!";
        NGraphCNNLayer * layer = reinterpret_cast<NGraphCNNLayer*>(result.get());
        layer->setNode(node);
        return result;
    };

    const auto isInternalConstLayer = [](const std::shared_ptr<::ngraph::op::Constant> &constLayer,
                                         const std::shared_ptr<::ngraph::Node> &consumerLayer,
                                         bool keep_constants) -> bool {
        if (((::ngraph::as_type_ptr<::ngraph::op::ConvolutionIE>(consumerLayer) ||
            ::ngraph::as_type_ptr<::ngraph::op::FullyConnected>(consumerLayer)) && !keep_constants) ||
            ::ngraph::as_type_ptr<::ngraph::op::v1::BinaryConvolution>(consumerLayer) ||
            ::ngraph::as_type_ptr<::ngraph::op::DeconvolutionIE>(consumerLayer) ||
            ::ngraph::as_type_ptr<::ngraph::op::v1::DeformableConvolution>(consumerLayer) ||
            ::ngraph::as_type_ptr<::ngraph::op::Elu>(consumerLayer) ||
            ::ngraph::as_type_ptr<::ngraph::op::NormalizeIE>(consumerLayer) ||
            ::ngraph::as_type_ptr<::ngraph::op::PRelu>(consumerLayer) ||
            ::ngraph::as_type_ptr<::ngraph::op::v1::Split>(consumerLayer) ||
            ::ngraph::as_type_ptr<::ngraph::op::VariadicSplit>(consumerLayer) ||
            ::ngraph::as_type_ptr<::ngraph::op::ScaleShiftIE>(consumerLayer) ||
            ::ngraph::as_type_ptr<::ngraph::op::Transpose>(consumerLayer) ||
            ::ngraph::as_type_ptr<::ngraph::op::LSTMSequenceIE>(consumerLayer) ||
            ::ngraph::as_type_ptr<::ngraph::op::RNNSequenceIE>(consumerLayer) ||
            ::ngraph::as_type_ptr<::ngraph::op::GRUSequenceIE>(consumerLayer) ||
            ::ngraph::as_type_ptr<::ngraph::op::RNNCellIE>(consumerLayer) ||
            ::ngraph::as_type_ptr<::ngraph::op::GRUCellIE>(consumerLayer)) {
            // Check that all input nodes except zero input are Constants for all ops except DeformableConvolutions
            // for which the input with index 1 is also dynamic
            size_t inputID = 1;
            if (::ngraph::as_type_ptr<::ngraph::op::v1::DeformableConvolution>(consumerLayer) ||
                             ::ngraph::as_type_ptr<::ngraph::op::GRUCellIE>(consumerLayer) ||
                             ::ngraph::as_type_ptr<::ngraph::op::RNNCellIE>(consumerLayer) ||
                    ::ngraph::as_type_ptr<::ngraph::op::GRUSequenceIE>(consumerLayer) ||
                    ::ngraph::as_type_ptr<::ngraph::op::RNNSequenceIE>(consumerLayer)) {
                inputID = 2;
            } else if (::ngraph::as_type_ptr<::ngraph::op::LSTMSequenceIE>(consumerLayer)) {
                inputID = 3;
            }

            for (; inputID < consumerLayer->inputs().size(); ++inputID) {
                auto inputLayer = consumerLayer->input(inputID).get_source_output().get_node_shared_ptr();
                if (inputLayer == constLayer) {
                    return true;
                }
            }
        } else if (::ngraph::as_type_ptr<::ngraph::op::LSTMCellIE>(consumerLayer)) {
            for (size_t inputID = 3; inputID < consumerLayer->inputs().size(); ++inputID) {
                auto inputLayer = consumerLayer->input(inputID).get_source_output().get_node_shared_ptr();
                if (inputLayer == constLayer) {
                    return true;
                }
            }
        }
        return false;
    };

    // Checks that node is internal layer for all layers from specific function
    const auto isInternalLayer = [=](const std::shared_ptr<::ngraph::Node> &node,
                                     bool keep_constant) -> bool {
        if (auto constantNode = ::ngraph::as_type_ptr<::ngraph::op::Constant>(node)) {
            for (const auto &consumerInputPort : constantNode->output(0).get_target_inputs()) {
                const auto &consumerLayer = consumerInputPort.get_node()->shared_from_this();
                if (!isInternalConstLayer(constantNode, consumerLayer, keep_constant))
                    return false;
            }
            return true;
        }

        return ::ngraph::as_type_ptr<::ngraph::op::Result>(node) != nullptr;
    };

    const auto keep_input_info = [](CNNNetworkImpl *network, const DataPtr &inData) {
        InputInfo::Ptr info(new InputInfo());
        info->setInputData(inData);
        network->setInputInfo(info);
    };

    // Check if some of function nodes has dynamic input or output shape
    // we collect this nodes and then throw an exception with the list
    // of dynamic nodes.
    std::stringstream err_log;
    for (const auto & node : graph->get_ordered_ops()) {
        bool is_dynamic = false;
        for (const auto & input : node->inputs()) {
            if (input.get_partial_shape().is_dynamic()) {
                is_dynamic = true;
                break;
            }
        }
        for (const auto & output : node->outputs()) {
            if (output.get_partial_shape().is_dynamic()) {
                is_dynamic = true;
                break;
            }
        }
        if (is_dynamic) err_log << node << std::endl;
    }
    if (!err_log.str().empty()) {
        IE_THROW() << "\nUnsupported dynamic ops: \n" << err_log.str();
    }

    IE_SUPPRESS_DEPRECATED_START
    const auto & icnnnetwork = static_cast<const ICNNNetwork &>(network);
    IE_SUPPRESS_DEPRECATED_END
    const CNNNetworkNGraphImpl* nGraphImpl = dynamic_cast<const CNNNetworkNGraphImpl*>(&icnnnetwork);

    InputsDataMap thisInputDataMap = network.getInputsInfo();

    // Construct network
    cnnNetworkImpl->setName(graph->get_friendly_name());

    const ngraph::NodeVector& nodes = graph->get_ops();
    bool keep_constants = keep_constant_inputs || ::ngraph::op::util::has_op_with_type<::ngraph::op::FakeQuantize>(graph);

    std::unordered_map<std::string, std::shared_ptr<ngraph::Node>> unique_names;
    auto can_change_name = [](const std::shared_ptr<ngraph::Node> & node) -> bool {
        if (ngraph::as_type_ptr<ngraph::op::Parameter>(node) ||
            ngraph::as_type_ptr<ngraph::op::Result>(node)) {
            return false;
        }
        for (const auto & output : node->outputs()) {
            for (const auto & consumer : output.get_target_inputs()) {
                if (ngraph::is_type<ngraph::op::Result>(consumer.get_node())) {
                    return false;
                }
            }
        }
        return true;
    };

    auto generate_unique_name = [&unique_names](std::string name) -> std::string {
        size_t suffix = 1;
        while (unique_names.count(name + "/" + std::to_string(suffix))) {
            ++suffix;
        }
        return name + "/" + std::to_string(suffix);
    };

    // normalize nodes names to be unique
    for (auto & node : nodes) {
        // skip Result operations as they have the same friendly name as their parent
        if (ngraph::is_type<ngraph::op::Result>(node.get())) {
            continue;
        }

        auto & duplicate = unique_names[node->get_friendly_name()];
        if (!duplicate) {
            duplicate = node;
            continue;
        }

        if (!can_change_name(duplicate) && !can_change_name(node)) {
            IE_THROW() << "Detected two output operations with the same name: " << duplicate << " and " << node;
        }

        auto & renamed = can_change_name(duplicate) ? duplicate : node;
        renamed->set_friendly_name(generate_unique_name(renamed->get_friendly_name()));

        unique_names[duplicate->get_friendly_name()] = duplicate;
        unique_names[node->get_friendly_name()] = node;
    }

    // Create layers and output data
    for (const auto &layer : nodes) {
        if (isInternalLayer(layer, keep_constants)) continue;

        // TODO: remove this rt info when all blobs will be inputs
        auto &rt_info = layer->get_rt_info();
        rt_info["keep_constants"] = std::make_shared<::ngraph::VariantWrapper<int64_t>> (keep_constants);

        CNNLayerPtr cnnLayer = createCNNLayer(layer);

        // Set originalLayersNames from FusedNames
        std::string originalNames = ::ngraph::getFusedNames(layer);
        if (!originalNames.empty()) {
            cnnLayer->params[ExecGraphInfoSerialization::ORIGINAL_NAMES] = originalNames;
        }

        std::string primitivesPriority = ::ngraph::getPrimitivesPriority(layer);
        if (!primitivesPriority.empty()) {
            cnnLayer->params["PrimitivesPriority"] = primitivesPriority;
        }

        // Copy runtime info attributes from Nodes to CNNLayers if they have VariantWrapper<std::string> type
        using VariantString = ::ngraph::VariantWrapper<std::string>;
        for (const auto &rt : rt_info) {
            if (auto str_attr = std::dynamic_pointer_cast<VariantString>(rt.second)) {
                if (details::CaselessEq<std::string>()(rt.first, "affinity")) {
                    cnnLayer->affinity = str_attr->get();
                } else {
                    cnnLayer->params[rt.first] = str_attr->get();
                }
            }
        }

        size_t inputCount(0);
        for (size_t i = 0; i < layer->get_input_size(); i++) {
            const auto &constant = ngraph::as_type_ptr<ngraph::op::Constant>(layer->input(i).get_source_output().get_node_shared_ptr());
            if (constant && isInternalConstLayer(constant, layer, keep_constants)) {
                continue;
            }
            inputCount++;
        }

        if (cnnLayer->type == "Memory" && cnnLayer->params["index"] == "1") {
            inputCount = 0;
        }

        cnnLayer->insData.resize(inputCount);

        for (size_t i = 0; i < layer->get_output_size(); i++) {
            // Memory node with index = 1 has no inputs according to the specification.
            // For proper conversion, we must cut off all the layers and data nodes above ReadValue,
            // if they are connected only with this layer.
            // Now MO generates only constants or constant sub-graphs as input to ReadValue op.
            if (std::dynamic_pointer_cast<::ngraph::op::Constant>(layer)) {
                bool all_to_read_value = !layer->output(i).get_target_inputs().empty();
                for (const auto &output_input : layer->output(i).get_target_inputs()) {
                    all_to_read_value
                            &= dynamic_cast<ngraph::op::ReadValueBase *>(output_input.get_node()) != nullptr;
                }
                if (all_to_read_value)
                    continue;
            }

            if (cnnLayer->type == "Memory" && cnnLayer->params["index"] == "0") {
                cnnLayer->outData.clear();
                continue;
            }
            NGRAPH_SUPPRESS_DEPRECATED_START
            auto outName = layer->output(i).get_tensor().get_name();
            NGRAPH_SUPPRESS_DEPRECATED_END
            if (outName.empty()) {
                outName = ngraph::op::util::create_ie_output_name(layer->output(i));
            }

            DataPtr &ptr = cnnNetworkImpl->getData(outName.c_str());
            IE_ASSERT(layer->get_output_partial_shape(i).is_static()) << " nGraph "
                << layer->description() << " operation with name: "
                << layer->get_friendly_name() << " cannot be converted to " << cnnLayer->type
                << " layer with name: " << cnnLayer->name << " because output with index "
                << i << " contains dynamic shapes: " << layer->get_output_partial_shape(i)
                << ". Try to use CNNNetwork::reshape() method in order to specialize shapes "
                << "before the conversion.";
            SizeVector dims = layer->get_output_shape(i);
            for (const auto &dim : dims) {
                if (!dim)
                    IE_THROW() << cnnLayer->type << " layer " << cnnLayer->name
                        << " has incorrect dimensions in the output data " << i;
            }
            if (!ptr && nGraphImpl && nGraphImpl->_data.find(outName) != nGraphImpl->_data.end()) {
                ptr = nGraphImpl->_data.at(outName);
                {
                    const auto layout =
                        dims.size() == ptr->getTensorDesc().getDims().size() ?
                        ptr->getTensorDesc().getLayout() :
                        TensorDesc::getLayoutByDims(dims);

                    ptr->reshape(dims, layout);
                }
                cnnNetworkImpl->addData(outName.c_str(), ptr);
            }

            if (!ptr) {
                ptr.reset(new Data(outName,
                                   {details::convertPrecision(layer->get_output_element_type(i)), dims,
                                    TensorDesc::getLayoutByDims(dims)}));
            }

            getCreatorLayer(ptr) = cnnLayer;
            cnnLayer->outData.push_back(ptr);
            if (std::dynamic_pointer_cast<::ngraph::op::Parameter>(layer)) {
                keep_input_info(cnnNetworkImpl, ptr);
            }
        }
        cnnNetworkImpl->addLayer(cnnLayer);
    }

    // Set input data
    for (const auto &layer : graph->get_ordered_ops()) {
        if (std::dynamic_pointer_cast<::ngraph::op::ReadValueBase>(layer))
            continue;
        if (std::dynamic_pointer_cast<::ngraph::op::Result>(layer)) {
            IE_ASSERT(layer->get_input_size() == 1);
            const auto &input = layer->input_value(0);
            NGRAPH_SUPPRESS_DEPRECATED_START
            auto name = input.get_tensor().get_name();
            NGRAPH_SUPPRESS_DEPRECATED_END
            if (!name.empty())
                cnnNetworkImpl->addOutput(name);
            else
                cnnNetworkImpl->addOutput(ngraph::op::util::create_ie_output_name(input));
            continue;
        }

        uint64_t count_of_skipped = 0;
        for (size_t i = 0; i < layer->get_input_size(); i++) {
            const auto &output_port = layer->input_value(i);
            const auto &input = output_port.get_node_shared_ptr();

            if (auto const_node = std::dynamic_pointer_cast<::ngraph::op::Constant>(input)) {
                if (isInternalConstLayer(const_node, layer, keep_constants)) {
                    count_of_skipped++;
                    continue;
                }
            }

            CNNLayerPtr prevCnnLayer;
            StatusCode ret = cnnNetworkImpl->getLayerByName(input->get_friendly_name().c_str(), prevCnnLayer, nullptr);
            if (ret != OK)
                IE_THROW() << "Cannot find layer with name: " << input->get_friendly_name();

            CNNLayerPtr cnnLayer;
            ret = cnnNetworkImpl->getLayerByName(layer->get_friendly_name().c_str(), cnnLayer, nullptr);
            if (ret != OK)
                IE_THROW() << "Cannot find layer with name: " << layer->get_friendly_name();

            auto inIndex = layer->input(i).get_index();
            if (cnnLayer->insData.size() <= (inIndex - count_of_skipped) ||
                prevCnnLayer->outData.size() <= output_port.get_index() || count_of_skipped > inIndex)
                IE_THROW() << "Cannot create CNNNetwork. Network structure is incorrect! "
                                   << "Input port " << inIndex << " (max " << cnnLayer->insData.size() << ") of "
                                   << cnnLayer->type << " layer " << cnnLayer->name
                                   << " cannot be connected with output port " << output_port.get_index()
                                   << " (max " << prevCnnLayer->outData.size() << ") of " << prevCnnLayer->type
                                   << " layer " << prevCnnLayer->name;
            cnnLayer->insData[inIndex - count_of_skipped] = prevCnnLayer->outData[output_port.get_index()];
            getInputTo(prevCnnLayer->outData[output_port.get_index()])[cnnLayer->name] = cnnLayer;
        }
    }

    // check all input ports are occupied
    for (const auto &kvp : cnnNetworkImpl->allLayers()) {
        const CNNLayer::Ptr &layer = kvp.second;
        size_t inSize = layer->insData.size();

        for (unsigned i = 0; i < inSize; i++) {
            if (!layer->insData[i].lock()) {
                IE_THROW() << "Layer " << layer->name.c_str() << " input port " << i
                                   << " is not connected to any data";
            }
        }

        // execution ngraph is fake graph and should not be validated
        if (layer->params.count(ExecGraphInfoSerialization::PERF_COUNTER) == 0) {
            layer->parseParams();
        }
    }

    if (!cnnNetworkImpl) IE_THROW() << "Cannot convert nGraph function to CNNNetworkImpl!";

    // update input preprocessing info
    InputsDataMap resultInputDataMap;
    cnnNetworkImpl->getInputsInfo(resultInputDataMap);
    IE_ASSERT(resultInputDataMap.size() == thisInputDataMap.size());
    for (auto i : resultInputDataMap) {
        auto &thisInputData = *thisInputDataMap[i.first];
        i.second->setPrecision(thisInputData.getPrecision());
        i.second->setLayout(thisInputData.getLayout());
        i.second->getPreProcess() = thisInputData.getPreProcess();
    }
}

std::shared_ptr<CNNNetworkImpl> convertFunctionToICNNNetwork(const std::shared_ptr<const ::ngraph::Function> &graph,
                                                             const CNNNetwork &network,
                                                             bool keep_constant_inputs) {
    auto cnnNetworkImpl = std::make_shared<details::CNNNetworkImpl>();
    convertFunctionToICNNNetwork(graph, network, cnnNetworkImpl.get(), keep_constant_inputs);
    return cnnNetworkImpl;
}

}  // namespace details
}  // namespace InferenceEngine
