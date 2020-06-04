// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_function_to_cnn_network.hpp"

#include <string>
#include <memory>
#include <vector>
#include <unordered_set>

#include <cnn_network_ngraph_impl.hpp>
#include "ngraph_ops/convolution_ie.hpp"
#include "ngraph_ops/deconvolution_ie.hpp"
#include "ngraph_ops/eltwise.hpp"
#include "ngraph_ops/fully_connected.hpp"
#include "ngraph_ops/gather_ie.hpp"
#include "ngraph_ops/gather_tree_ie.hpp"
#include "ngraph_ops/gru_cell_ie.hpp"
#include "ngraph_ops/interp.hpp"
#include "ngraph_ops/lrn_ie.hpp"
#include "ngraph_ops/lstm_cell_ie.hpp"
#include "ngraph_ops/normalize_ie.hpp"
#include "ngraph_ops/pad_ie.hpp"
#include "ngraph_ops/onehot_ie.hpp"
#include "ngraph_ops/power.hpp"
#include "ngraph_ops/proposal_ie.hpp"
#include "ngraph_ops/relu_ie.hpp"
#include "ngraph_ops/scaleshift.hpp"
#include "ngraph_ops/tile_ie.hpp"
#include "ngraph_ops/hard_sigmoid_ie.hpp"
#include "ngraph_ops/nms_ie.hpp"
#include "ngraph_ops/crop_ie.hpp"
#include "ngraph_ops/selu_ie.hpp"
#include "ngraph_ops/rnn_cell_ie.hpp"
#include "ngraph_ops/topk_ie.hpp"
#include "generic_ie.hpp"

#include "ie_profiling.hpp"
#include "ie_cnn_layer_builder_ngraph.h"

#include <debug.h>
#include <ngraph/opsets/opset1.hpp>
#include "transformations/convert_opset1_to_legacy/convert_opset1_to_legacy.hpp"
#include "transformations/utils/utils.hpp"
#include "transformations/rt_info/fused_names_attribute.hpp"
#include "transformations/rt_info/primitives_priority_attribute.hpp"

namespace InferenceEngine {
namespace details {

/**
 * @brief Creator for CNNLayer from nGraph op
 */
class CNNLayerCreator : public ::ngraph::AttributeVisitor {
public:
    using CreatorFor = std::function<CNNLayerPtr(const std::shared_ptr<::ngraph::Node>& node,
                                                 const std::map<std::string, std::string> param)>;
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

    void on_adapter(const std::string& name, ::ngraph::ValueAccessor<std::vector<int64_t>>& adapter) override {
        auto shape = adapter.get();
        params[name] = joinVec(shape);
    }

    void on_adapter(const std::string& name, ::ngraph::ValueAccessor<double>& adapter) override {
        params[name] = std::to_string(adapter.get());
    }

    void on_adapter(const std::string& name, ::ngraph::ValueAccessor<int64_t>& adapter) override {
        params[name] = std::to_string(adapter.get());
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
        for (size_t i = 0; i < shape.rank().get_length(); i++) {
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
    }
}

InferenceEngine::details::CNNLayerCreator::CNNLayerCreator(const std::shared_ptr<::ngraph::Node>& node): node(node) {
    addSpecificCreator({"Parameter"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                         const std::map<std::string, std::string> params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "Input",
            details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<CNNLayer>(attrs);
        return res;
    });
    // TODO - Remove "GreaterEq" once ngraph transitions to GreaterEqual
    addSpecificCreator({"Eltwise", "Subtract", "Power", "Maximum", "Divide", "Greater", "GreaterEqual", "FloorMod", "LogicalOr", "LogicalAnd", "LogicalXor",
        "GreaterEq", "Less", "LessEqual", "Equal", "NotEqual", "Multiply", "Add"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                                                 const std::map<std::string, std::string> params) -> CNNLayerPtr {
            LayerParams attrs = {node->get_friendly_name(), "Eltwise",
                details::convertPrecision(node->get_output_element_type(0))};
            auto res = std::make_shared<EltwiseLayer>(attrs);
            res->params = params;
            if (node->description() == "Maximum") {
                res->params["operation"] = "max";
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
                if (castedLayer == nullptr) THROW_IE_EXCEPTION << "Cannot get " << attrs.type << " layer " << attrs.name;
                std::string type;
                switch (castedLayer->eltwise_type) {
                case ELTWISE_TYPE::Sum:
                    type = "sum";
                    break;
                case ELTWISE_TYPE::Prod:
                    type = "prod";
                    break;
                default:
                    THROW_IE_EXCEPTION << "Not supported eltwise type!";
                }

                res->params["operation"] = type;
            }
            return res;
        });
    addSpecificCreator({"Concat"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                      const std::map<std::string, std::string> params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), node->description(),
            details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<ConcatLayer>(attrs);
        res->params = params;
        return res;
    });
    addSpecificCreator({"AvgPool", "MaxPool"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                                  const std::map<std::string, std::string> params) -> CNNLayerPtr {
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
                                      const std::map<std::string, std::string> params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), node->description(),
                             details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<SelectLayer>(attrs);
        res->params = params;
        return res;
    });
    addSpecificCreator({"BinaryConvolution"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                      const std::map<std::string, std::string> params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), node->description(),
                             details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<BinaryConvolutionLayer>(attrs);

        // todo: investigate difference between ngraph parameters for BinConvolution and the implementation above
        // this leads to accuracy issue for Precollected_ONNX_ResNet50_88percentinto1bit e2e test
        // res->params = params;

        auto castedLayer = ::ngraph::as_type_ptr<::ngraph::op::v1::BinaryConvolution>(node);

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

        auto weights_shape = castedLayer->input(1).get_source_output().get_shape();
        res->params["input"] = Builder::asString(weights_shape[1]);
        res->params["pad_value"] = Builder::asString(castedLayer->get_pad_value());

        Builder::NodeConverter<::ngraph::op::Constant> converter;

        const auto weightsNode = castedLayer->get_inputs()[1].get_output().get_node();
        if (converter.canCreate(weightsNode)) {
            const auto& weights = converter.createLayer(weightsNode);
            res->blobs["weights"] = weights->blobs["custom"];
            res->_weights = weights->blobs["custom"];
        }
        return res;
    });

    addSpecificCreator({"SpaceToBatch"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                      const std::map<std::string, std::string> params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), node->description(),
                             details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<SpaceToBatchLayer>(attrs);
        res->params = params;
        return res;
    });

    addSpecificCreator({"BatchToSpace"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                      const std::map<std::string, std::string> params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), node->description(),
                             details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<BatchToSpaceLayer>(attrs);
        res->params = params;
        return res;
    });
    
    addSpecificCreator({"Assign"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                            const std::map<std::string, std::string> params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "Memory",
                             details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<CNNLayer>(attrs);
        res->params["id"] = params.at("variable_id");
        res->params["index"] = "0";
        res->params["size"] = "2";
        return res;
    });

    addSpecificCreator({"ReadValue"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                            const std::map<std::string, std::string> params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "Memory",
                             details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<CNNLayer>(attrs);
        res->params["id"] = params.at("variable_id");
        res->params["index"] = "1";
        res->params["size"] = "2";
        return res;
    });

    addSpecificCreator({"DepthToSpace"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                            const std::map<std::string, std::string> params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), node->description(),
                             details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<DepthToSpaceLayer>(attrs);
        res->params = params;
        return res;
    });

    addSpecificCreator({"SpaceToDepth"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                            const std::map<std::string, std::string> params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), node->description(),
                             details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<SpaceToDepthLayer>(attrs);
        res->params = params;
        return res;
    });

    addSpecificCreator({"RNNCell"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                            const std::map<std::string, std::string> params) -> CNNLayerPtr {
        THROW_IE_EXCEPTION << "RNNCell operation has a form that is not supported." << node->get_friendly_name()
                           << " should be converted to RNNCellIE operation";
        return nullptr;
    });

    addSpecificCreator({"GRUCell"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                            const std::map<std::string, std::string> params) -> CNNLayerPtr {
        THROW_IE_EXCEPTION << "GRUCell operation has a form that is not supported." << node->get_friendly_name()
                           << " should be converted to GRUCellIE operation";
        return nullptr;
    });

    addSpecificCreator({"RNNCellIE"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                      const std::map<std::string, std::string> params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "RNNCell",
                             details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<RNNCell>(attrs);
        res->params = params;

        Builder::NodeConverter<ngraph::op::Constant> converter;
        const auto weightsNode = node->input_value(2).get_node_shared_ptr();
        if (converter.canCreate(weightsNode)) {
            const auto& weights = converter.createLayer(weightsNode);
            res->blobs["weights"] = weights->blobs["custom"];
            res->_weights = weights->blobs["custom"];
        }

        const auto biasNode = node->input_value(3).get_node_shared_ptr();
        if (converter.canCreate(biasNode)) {
            const auto& bias = converter.createLayer(biasNode);
            res->blobs["biases"] = bias->blobs["custom"];
            res->_biases = bias->blobs["custom"];
        }
        return res;
    });

    addSpecificCreator({"GRUCellIE"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                         const std::map<std::string, std::string> params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "GRUCell",
                             details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<GRUCell>(attrs);
        res->params = params;

        Builder::NodeConverter<ngraph::op::Constant> converter;
        const auto weightsNode = node->input_value(2).get_node_shared_ptr();
        if (converter.canCreate(weightsNode)) {
            const auto& weights = converter.createLayer(weightsNode);
            res->blobs["weights"] = weights->blobs["custom"];
            res->_weights = weights->blobs["custom"];
        }

        const auto biasNode = node->input_value(3).get_node_shared_ptr();
        if (converter.canCreate(biasNode)) {
            const auto& bias = converter.createLayer(biasNode);
            res->blobs["biases"] = bias->blobs["custom"];
            res->_biases = bias->blobs["custom"];
        }
        return res;
    });

    addSpecificCreator({"ScatterElementsUpdate"}, [](const std::shared_ptr<::ngraph::Node>& node,
        const std::map<std::string, std::string> params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), node->description(),
            details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<ScatterElementsUpdateLayer>(attrs);
        res->params = params;
        return res;
    });

    addSpecificCreator({"ScatterUpdate"}, [](const std::shared_ptr<::ngraph::Node>& node,
        const std::map<std::string, std::string> params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), node->description(),
            details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<ScatterUpdateLayer>(attrs);
        res->params = params;
        return res;
    });

    addSpecificCreator({"StaticShapeTopK"}, [](const std::shared_ptr<::ngraph::Node>& node,
        const std::map<std::string, std::string> params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "TopK",
            details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<TopKLayer>(attrs);
        res->params = params;
        return res;
    });

    addSpecificCreator({"Transpose"}, [](const std::shared_ptr<::ngraph::Node>& node,
        const std::map<std::string, std::string> params) -> CNNLayerPtr {
        LayerParams attrs = {node->get_friendly_name(), "Permute",
            details::convertPrecision(node->get_output_element_type(0))};
        auto res = std::make_shared<InferenceEngine::CNNLayer>(attrs);
        res->params = params;
        if (auto transpose_const = std::dynamic_pointer_cast<ngraph::op::Constant>(node->input_value(1).get_node_shared_ptr())) {
            res->params["order"] = Builder::asString(transpose_const->cast_vector<int64_t>());
        }
        return res;

    });

    addSpecificCreator({"PriorBox"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                       const std::map<std::string, std::string> params) -> CNNLayerPtr {
        THROW_IE_EXCEPTION << "PriorBox operation has a form that is not supported." << node->get_friendly_name()
                           << " should be replaced by constant during constant folding.";
        return nullptr;
    });

    addSpecificCreator({"PriorBoxClustered"}, [](const std::shared_ptr<::ngraph::Node>& node,
                                       const std::map<std::string, std::string> params) -> CNNLayerPtr {
        THROW_IE_EXCEPTION << "PriorBoxClustered operation has a form that is not supported." << node->get_friendly_name()
                           << " should be replaced by constant during constant folding.";
        return nullptr;
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

std::shared_ptr<CNNNetworkImpl> convertFunctionToICNNNetwork(const std::shared_ptr<const ::ngraph::Function>& graph, const ICNNNetwork &network) {
    IE_PROFILING_AUTO_SCOPE(convertFunctionToICNNNetwork)
    const auto createCNNLayer = [](const std::shared_ptr<::ngraph::Node> &node) -> CNNLayerPtr {
        class NGraphCNNLayer: public CNNLayer {
        public:
            void setNode(const std::shared_ptr<::ngraph::Node>& node) {
                this->node = node;
            }
        };
        static std::vector<std::shared_ptr<Builder::INodeConverter>> convertors = {
                std::make_shared<Builder::NodeConverter<::ngraph::op::Abs>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Acos>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::Add>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Asin>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Atan>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::AvgPool>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::BatchNormInference>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::Broadcast>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Clamp>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Concat>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Constant>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::ConvolutionIE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::DeconvolutionIE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Cos>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Cosh>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::CropIE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Convert>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::CTCGreedyDecoder>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::DetectionOutput>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::DeformableConvolution>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::DeformablePSROIPooling>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::Divide>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::Reshape>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Eltwise>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Elu>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Erf>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Exp>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::FakeQuantize>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Floor>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Ceiling>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::GatherIE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::GatherTree>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::GatherTreeIE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Interp>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Interpolate>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Log>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::LRN>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::LRN_IE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::MVN>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::FullyConnected>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::MatMul>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::GenericIE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::GRN>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::MaxPool>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::Maximum>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::Minimum>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::Multiply>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::NonMaxSuppression>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::NonMaxSuppressionIE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::NormalizeL2>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::NormalizeIE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::OneHotIE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::PRelu>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::PadIE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::Power>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::PowerIE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Proposal>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::ProposalIE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Relu>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::SeluIE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::ReLUIE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Range>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::ReverseSequence>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::ReduceMin>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::ReduceMax>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::ReduceMean>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::ReduceProd>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::ReduceSum>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::ResampleV2>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::RegionYolo>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::ReorgYolo>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::ROIPooling>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::PSROIPooling>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::ScaleShiftIE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::ShapeOf>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Sigmoid>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Sin>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Sign>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Sinh>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::SquaredDifference>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::Softmax>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::Split>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::VariadicSplit>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::StridedSlice>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Squeeze>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Sqrt>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Subtract>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Tan>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Tanh>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::TileIE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::TopK>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::TopKIE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::Unsqueeze>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::TensorIterator>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::LSTMCellIE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::HardSigmoid>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::HardSigmoid_IE>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::LogicalNot>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::ReduceLogicalAnd>>(),
                std::make_shared<Builder::NodeConverter<::ngraph::op::v1::ReduceLogicalOr>>(),
        };
        CNNLayerPtr result;

        for (auto &convertor : convertors) {
            if (!convertor->canCreate(node)) continue;
            result = convertor->createLayer(node);
            break;
        }

        if (!result) {
            CNNLayerCreator visitor(node);
            if (node->visit_attributes(visitor)) result = visitor.create();
        }

        if (!result)
            THROW_IE_EXCEPTION << "Cannot cast ngraph node " << node->get_friendly_name() << " to CNNLayer!";
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
            ::ngraph::as_type_ptr<::ngraph::op::RNNCellIE>(consumerLayer) ||
            ::ngraph::as_type_ptr<::ngraph::op::GRUCellIE>(consumerLayer)) {
            // Check that all input nodes except zero input are Constants for all ops except DeformableConvolutions
            // for which the input with index 1 is also dynamic
            size_t inputID = ::ngraph::as_type_ptr<::ngraph::op::v1::DeformableConvolution>(consumerLayer) ||
                             ::ngraph::as_type_ptr<::ngraph::op::GRUCellIE>(consumerLayer) ||
                             ::ngraph::as_type_ptr<::ngraph::op::RNNCellIE>(consumerLayer)? 2 : 1;
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
                                     const std::unordered_set<std::string> &names,
                                     bool keep_constant) -> bool {
        if (auto constantNode = ::ngraph::as_type_ptr<::ngraph::op::Constant>(node)) {
            for (const auto &consumerInputPort : constantNode->get_outputs()[0].get_inputs()) {
                const auto &consumerLayer = consumerInputPort->get_node();
                if (names.find(consumerLayer->get_name()) == names.end())
                    continue;
                if (!isInternalConstLayer(constantNode, consumerLayer, keep_constant))
                    return false;
            }
            return true;
        }

        return ::ngraph::as_type_ptr<::ngraph::op::Result>(node) != nullptr;
    };

    const auto keep_input_info = [](std::shared_ptr<details::CNNNetworkImpl> &network, const DataPtr &inData) {
        InputInfo::Ptr info(new InputInfo());
        info->setInputData(inData);
        network->setInputInfo(info);
    };

    const CNNNetworkNGraphImpl* nGraphImpl = dynamic_cast<const CNNNetworkNGraphImpl*>(&network);

    InputsDataMap thisInputDataMap;
    network.getInputsInfo(thisInputDataMap);

    // Create network
    auto cnnNetworkImpl = std::make_shared<details::CNNNetworkImpl>();
    cnnNetworkImpl->setName(graph->get_friendly_name());
    // In generic case all nGraph functions have MIXED precision
    // Network precision should be deprecated
    cnnNetworkImpl->setPrecision(Precision::MIXED);

    // Collect all names from current graph
    // It is necessary in order to differentiate outputs from constant layers when we share constants
    // (Constant operations contains outputs for converted and original functions)
    const ngraph::NodeVector& nodes = graph->get_ops();

    std::unordered_set<std::string> op_names;
    for (const auto &layer : nodes)
        op_names.insert(layer->get_name());

    bool keep_constants = ::ngraph::op::util::has_op_with_type<::ngraph::op::FakeQuantize>(graph);

    // Create layers and output data
    for (const auto &layer : nodes) {
        if (isInternalLayer(layer, op_names, keep_constants)) continue;

        // TODO: remove this rt info when all blobs will be inputs
        auto &rt_info = layer->get_rt_info();
        rt_info["keep_constants"] = std::make_shared<::ngraph::VariantWrapper<int64_t>> (keep_constants);

        CNNLayerPtr cnnLayer = createCNNLayer(layer);

        // Set originalLayersNames from FusedNames
        std::string originalNames = ::ngraph::getFusedNames(layer);
        if (!originalNames.empty()) {
            cnnLayer->params["originalLayersNames"] = originalNames;
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
            const auto &constant = ngraph::as_type_ptr<ngraph::op::Constant>(layer->get_inputs()[i].get_output().get_node());
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
            if (cnnLayer->type == "Memory" && cnnLayer->params["index"] == "0") {
                cnnLayer->outData.clear();
                continue;
            }
            std::string outName = layer->get_friendly_name();
            if (layer->get_output_size() != 1) outName += "." + std::to_string(i);
            DataPtr &ptr = cnnNetworkImpl->getData(outName.c_str());

            SizeVector dims;
            dims = layer->get_output_shape(i);
            for (const auto &dim : dims) {
                if (!dim)
                    THROW_IE_EXCEPTION << cnnLayer->type << " layer " << cnnLayer->name
                        << " has incorrect dimensions in the output data " << i;
            }
            if (!ptr && nGraphImpl && nGraphImpl->_data.find(outName) != nGraphImpl->_data.end()) {
                ptr = nGraphImpl->_data.at(outName);
                if (auto nData = std::dynamic_pointer_cast<InferenceEngine::details::NGraphData>(ptr)) {
                    const auto layout =
                        dims.size() == nData->getTensorDesc().getDims().size() ?
                        nData->getTensorDesc().getLayout() :
                        TensorDesc::getLayoutByDims(dims);

                    nData->reset();
                    nData->reshape(dims, layout);
                }
                cnnNetworkImpl->addData(outName.c_str(), ptr);
            }

            if (!ptr) {
                ptr.reset(new Data(outName,
                                   {details::convertPrecision(layer->get_output_element_type(i)), dims,
                                    TensorDesc::getLayoutByDims(dims)}));
            }

            ptr->getCreatorLayer() = cnnLayer;
            cnnLayer->outData.push_back(ptr);
            if (std::dynamic_pointer_cast<::ngraph::op::Parameter>(layer)) {
                keep_input_info(cnnNetworkImpl, ptr);
            }
        }
        cnnNetworkImpl->addLayer(cnnLayer);
    }

    // Set input data
    for (const auto &layer : graph->get_ordered_ops()) {
        if (std::dynamic_pointer_cast<::ngraph::op::ReadValue>(layer))
            continue;
        if (std::dynamic_pointer_cast<::ngraph::op::Result>(layer)) {
            IE_ASSERT(layer->get_inputs().size() == 1);
            const auto &input = layer->input_value(0);
            std::string outName = input.get_node_shared_ptr()->get_friendly_name();
            if (input.get_node_shared_ptr()->get_output_size() != 1)
                outName += "." + std::to_string(input.get_index());
            cnnNetworkImpl->addOutput(outName);
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
                THROW_IE_EXCEPTION << "Cannot find layer with name: " << input->get_friendly_name();

            CNNLayerPtr cnnLayer;
            ret = cnnNetworkImpl->getLayerByName(layer->get_friendly_name().c_str(), cnnLayer, nullptr);
            if (ret != OK) THROW_IE_EXCEPTION << "Cannot find layer with name: " << layer->get_friendly_name();

            auto inIndex = layer->input(i).get_index();
            if (cnnLayer->insData.size() <= (inIndex - count_of_skipped) ||
                prevCnnLayer->outData.size() <= output_port.get_index() || count_of_skipped > inIndex)
                THROW_IE_EXCEPTION << "Cannot create ICNNNetwork. Network structure is incorrect! "
                                   << "Input port " << inIndex << " (max " << cnnLayer->insData.size() << ") of "
                                   << cnnLayer->type << " layer " << cnnLayer->name
                                   << " cannot be connected with output port " << output_port.get_index()
                                   << " (max " << prevCnnLayer->outData.size() << ") of " << prevCnnLayer->type
                                   << " layer " << prevCnnLayer->name;
            cnnLayer->insData[inIndex - count_of_skipped] = prevCnnLayer->outData[output_port.get_index()];
            prevCnnLayer->outData[output_port.get_index()]->getInputTo()[cnnLayer->name] = cnnLayer;
        }
    }

    // check all input ports are occupied
    for (const auto &kvp : cnnNetworkImpl->allLayers()) {
        const CNNLayer::Ptr &layer = kvp.second;
        size_t inSize = layer->insData.size();

        for (unsigned i = 0; i < inSize; i++) {
            if (!layer->insData[i].lock()) {
                THROW_IE_EXCEPTION << "Layer " << layer->name.c_str() << " input port " << i
                                   << " is not connected to any data";
            }
        }
        layer->validateLayer();
    }

    if (!cnnNetworkImpl) THROW_IE_EXCEPTION << "Cannot convert nGraph function to CNNNetworkImpl!";

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

    for (const auto &ext : ::ngraph::op::GenericIE::getExtensions(graph)) {
        cnnNetworkImpl->AddExtension(ext, nullptr);
    }
    return cnnNetworkImpl;
}
}  // namespace details
}  // namespace InferenceEngine
