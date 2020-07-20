// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_layer_validators.hpp"

#include <ie_iextension.h>

#include <cmath>
#include <details/ie_exception.hpp>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "debug.h"
#include "ie_layers.h"
#include "xml_parse_utils.h"

#ifdef __clang__
#pragma clang diagnostic ignored "-Wunused-variable"
#endif

namespace InferenceEngine {

using namespace details;
using std::map;
using std::string;
using std::vector;

template <typename T, typename P>
inline bool one_of(T val, P item) {
    return val == item;
}
template <typename T, typename P, typename... Args>
inline bool one_of(T val, P item, Args... item_others) {
    return val == item || one_of(val, item_others...);
}

void CNNLayer::validateLayer() {
    try {
        LayerValidator::Ptr validator = LayerValidators::getInstance()->getValidator(type);
        validator->parseParams(this);
        validator->checkParams(this);
        InOutDims shapes;
        getInOutShapes(this, shapes);
        validator->checkShapes(this, shapes.inDims);
    } catch (const InferenceEngineException& ie_e) {
        THROW_IE_EXCEPTION << "Error of validate layer: " << this->name << " with type: " << this->type << ". "
                           << ie_e.what();
    }
}

void checkNumOfInput(const std::vector<SizeVector>& inShapes, const vector<int>& expected_num_of_shapes) {
    bool shape_was_found = false;
    for (const auto& i : expected_num_of_shapes) {
        if (inShapes.size() == i) {
            shape_was_found = true;
            break;
        }
    }
    if (!shape_was_found) {
        THROW_IE_EXCEPTION << "Number of inputs (" << inShapes.size()
                           << ") is not equal to expected ones: " << expected_num_of_shapes.size();
    }
}

LayerValidators* LayerValidators::getInstance() {
    static LayerValidators instance;
    return &instance;
}

LayerValidator::Ptr LayerValidators::getValidator(const std::string& type) {
    if (_validators.find(type) == _validators.end()) {
        return std::make_shared<GeneralValidator>(type);
    }
    return _validators[type];
}

GeneralValidator::GeneralValidator(const std::string& _type): LayerValidator(_type) {}

//////////////////////////////////////////////////////////

SparseFillEmptyRowsValidator::SparseFillEmptyRowsValidator(const std::string& _type): LayerValidator(_type) {}

void SparseFillEmptyRowsValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<SparseFillEmptyRowsLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << layer->name << " Layer is not instance of SparseFillEmptyRows class";
    }
}

void SparseFillEmptyRowsValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

void SparseFillEmptyRowsValidator::checkShapes(const CNNLayer* layer, const vector<SizeVector>& inShapes) const {
    auto casted = dynamic_cast<const SparseFillEmptyRowsLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << layer->name << " Layer is not instance of SparseFillEmptyRows class";
    }

    size_t numInputs = inShapes.size();
    if (numInputs != 4)
        THROW_IE_EXCEPTION << layer->name
                           << " SparseFillEmptyRows must have 4 inputs, but actually it has: " << numInputs;

    // Check dimensions of a tensor with input indices
    if (inShapes[0].size() != 2)
        THROW_IE_EXCEPTION << layer->name << " Input indices of SparseFillEmptyRows must be 2-D tensor";
    if (inShapes[0][1] != 2) THROW_IE_EXCEPTION << layer->name << " Input indices must be two-dimensional";

    // Check dimensions of a tensor with input values
    if (inShapes[1].size() != 1)
        THROW_IE_EXCEPTION << layer->name << " Input values of SparseFillEmptyRows must be 1-D tensor";
    if (inShapes[1][0] != inShapes[0][0])
        THROW_IE_EXCEPTION << layer->name << " Number of input indices and values must match";

    // Check dimensions of a tensor with a dense shape
    if (inShapes[2].size() != 1)
        THROW_IE_EXCEPTION << layer->name << " Dense shape of SparseFillEmptyRows must be 1-D tensor";
    // TODO: check that dense shape value is set

    // Check dimensions of a tensor with default value
    if (inShapes[3].size() != 1)
        THROW_IE_EXCEPTION << layer->name << " Default value of SparseFillEmptyRows must be 1-D tensor";
}

//////////////////////////////////////////////////////////

SparseSegmentReduceValidator::SparseSegmentReduceValidator(const std::string& _type): LayerValidator(_type) {}

void SparseSegmentReduceValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<SparseSegmentReduceLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << layer->name << " Layer is not instance of SparseSegmentReduce class";
    }
}

void SparseSegmentReduceValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

void SparseSegmentReduceValidator::checkShapes(const CNNLayer* layer, const vector<SizeVector>& inShapes) const {
    auto casted = dynamic_cast<const SparseSegmentReduceLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << layer->name << " Layer is not instance of SparseSegmentReduce class";
    }

    size_t numInputs = inShapes.size();
    if (numInputs != 3)
        THROW_IE_EXCEPTION << layer->name
                           << " SparseSegmentReduce must take three inputs, but actually it has: " << numInputs;

    // check that the second and the third inputs are one-dimensional
    if (inShapes[1].size() != 1) {
        THROW_IE_EXCEPTION << layer->name << " The second input of SparseSegmentReduce must be one-dimensional";
    }
    if (inShapes[2].size() != 1) {
        THROW_IE_EXCEPTION << layer->name << " The third input of SparseSegmentReduce must be one-dimensional";
    }
}

//////////////////////////////////////////////////////////

SparseToDenseValidator::SparseToDenseValidator(const std::string& _type) : LayerValidator(_type) {}

void SparseToDenseValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<SparseToDenseLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << layer->name << " Layer is not instance of SparseToDense class";
    }
}

void SparseToDenseValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

void SparseToDenseValidator::checkShapes(const CNNLayer* layer, const vector<SizeVector>& inShapes) const {
    auto casted = dynamic_cast<const SparseToDenseLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << layer->name << " Layer is not instance of SparseToDense class";
    }

    size_t numInputs = inShapes.size();
    if (numInputs != 3 && numInputs != 4)
        THROW_IE_EXCEPTION << layer->name
                           << " SparseToDense must take three or four inputs, but actually it has: "
                           << numInputs;

    // check shapes of inputs
    if (inShapes[0].size() != 2) {
        THROW_IE_EXCEPTION << layer->name
                           << " The first input with indices of SparseToDense must be two-dimensional";
    }
    if (inShapes[1].size() != 1 || inShapes[1][0] != inShapes[0][1]) {
        THROW_IE_EXCEPTION << layer->name
                           << " The second input with a dense shape of SparseToDense must be one-dimensional";
    }
    if (inShapes[2].size() != 1 || inShapes[2][0] != inShapes[0][0]) {
        THROW_IE_EXCEPTION << layer->name
                           << " The third input with values of SparseToDense must be one-dimensional";
    }
    if (numInputs == 4 && inShapes[3].size() != 0) {
        THROW_IE_EXCEPTION << layer->name
            << " The fourth input with default value of SparseToDense must be a scalar";
    }

    // check precisions of inputs
    const size_t INPUT_INDICES_PORT = 0;
    const size_t INPUT_DENSE_SHAPE = 1;
    const size_t INPUT_VALUES_PORT = 2;
    const size_t INPUT_DEFAULT_VALUE = 3;

    Precision input_indices_precision = layer->insData[INPUT_INDICES_PORT].lock()->getTensorDesc().getPrecision();
    if (input_indices_precision != Precision::I32)
        THROW_IE_EXCEPTION << layer->name << " Incorrect input indices precision. Only I32 are supported!";
    Precision input_dense_shape_precision = layer->insData[INPUT_DENSE_SHAPE].lock()->getTensorDesc().getPrecision();
    if (input_dense_shape_precision != Precision::I32)
        THROW_IE_EXCEPTION << layer->name << " Incorrect input dense shape precision. Only I32 are supported!";
    Precision input_values_precision = layer->insData[INPUT_VALUES_PORT].lock()->getTensorDesc().getPrecision();
    if (input_values_precision != Precision::I32)
        THROW_IE_EXCEPTION << layer->name << " Incorrect input values precision. Only I32 are supported!";
    if (numInputs == 4) {
        Precision input_default_value_precision = layer->insData[INPUT_DEFAULT_VALUE].lock()->getTensorDesc().getPrecision();
        if (input_default_value_precision != Precision::I32)
            THROW_IE_EXCEPTION << layer->name << " Incorrect input default value precision. Only I32 are supported!";
    }
}

/****************************************/
/*** RNN specific validators ************/
/****************************************/

static RNNCellBase::CellType cell_type_from(string type_name) {
    const vector<string> to_remove {"Cell", "Sequence"};
    for (auto& sub : to_remove) {
        auto idx = type_name.find(sub);
        if (idx != string::npos) type_name.erase(idx);
    }

    if (!one_of(type_name, "LSTM", "RNN", "GRU"))
        THROW_IE_EXCEPTION << "Unknown RNN cell type " << type_name << ". "
                           << "Expected one of [ LSTM | RNN | GRU ].";

    return type_name == "LSTM"
               ? RNNSequenceLayer::LSTM
               : type_name == "GRU" ? RNNSequenceLayer::GRU
                                    : type_name == "RNN" ? RNNSequenceLayer::RNN : RNNSequenceLayer::LSTM;
}

static RNNSequenceLayer::Direction direction_from(string direction_name) {
    if (!one_of(direction_name, "Forward", "Backward", "Bidirectional"))
        THROW_IE_EXCEPTION << "Unknown RNN direction type " << direction_name << ". "
                           << "Expected one of [ Forward | Backward | Bidirectional ].";

    return direction_name == "Forward"
               ? RNNSequenceLayer::FWD
               : direction_name == "Backward"
                     ? RNNSequenceLayer::BWD
                     : direction_name == "Bidirecttional" ? RNNSequenceLayer::BDR : RNNSequenceLayer::FWD;
}

RNNBaseValidator::RNNBaseValidator(const std::string& _type, RNNSequenceLayer::CellType CELL): LayerValidator(_type) {
    if (RNNSequenceLayer::LSTM == CELL) {
        def_acts = {"sigmoid", "tanh", "tanh"};
        def_alpha = {0, 0, 0};
        def_beta = {0, 0, 0};
        G = 4;
        NS = 2;
    } else if (RNNSequenceLayer::GRU == CELL) {
        def_acts = {"sigmoid", "tanh"};
        def_alpha = {0, 0};
        def_beta = {0, 0};
        G = 3;
        NS = 1;
    } else if (RNNSequenceLayer::RNN == CELL) {
        def_acts = {"tanh"};
        def_alpha = {0};
        def_beta = {0};
        G = 1;
        NS = 1;
    } else {
        IE_ASSERT(false);
    }
}

void RNNBaseValidator::parseParams(CNNLayer* layer) {
    auto rnn = dynamic_cast<RNNCellBase*>(layer);
    if (!rnn) THROW_IE_EXCEPTION << "Layer is not instance of RNNLayer class";

    rnn->cellType = cell_type_from(layer->type);
    rnn->hidden_size = rnn->GetParamAsInt("hidden_size");
    rnn->clip = rnn->GetParamAsFloat("clip", 0.0f);
    rnn->activations = rnn->GetParamAsStrings("activations", def_acts);
    rnn->activation_alpha = rnn->GetParamAsFloats("activation_alpha", def_alpha);
    rnn->activation_beta = rnn->GetParamAsFloats("activation_beta", def_beta);

    if (rnn->cellType == RNNCellBase::GRU) {
        auto lbr = rnn->GetParamAsBool("linear_before_reset", false);
        if (lbr) rnn->cellType = RNNCellBase::GRU_LBR;
    }
}

void RNNBaseValidator::checkParams(const InferenceEngine::CNNLayer* layer) {
    auto rnn = dynamic_cast<const RNNCellBase*>(layer);
    if (!rnn) THROW_IE_EXCEPTION << "Layer is not instance of RNNLayer class";

    if (rnn->clip < 0.0f) THROW_IE_EXCEPTION << "Clip parameter should be positive";

    for (auto& act : rnn->activations)
        if (!one_of(act, "sigmoid", "tanh", "relu"))
            THROW_IE_EXCEPTION << "Unsupported activation function (" << act << ") for RNN layer.";

    int act_num_required = def_acts.size();
    if (rnn->activations.size() != act_num_required)
        THROW_IE_EXCEPTION << "Expected " << act_num_required << " activations, but provided "
                           << rnn->activations.size();

    if (rnn->activation_alpha.size() != act_num_required)
        THROW_IE_EXCEPTION << "Expected " << act_num_required << " activation alpha parameters, "
                           << "but provided " << rnn->activation_alpha.size();
    if (rnn->activation_beta.size() != act_num_required)
        THROW_IE_EXCEPTION << "Expected " << act_num_required << " activation beta parameters, "
                           << "but provided " << rnn->activation_beta.size();
}

void RNNBaseValidator::checkCorrespondence(const CNNLayer* layer, const map<string, Blob::Ptr>& blobs,
                                           const vector<SizeVector>& inShapes) const {
    auto rnn = dynamic_cast<const RNNCellBase*>(layer);
    if (!rnn) THROW_IE_EXCEPTION << "Layer is not instance of RNNLayer class";

    if (blobs.size() != 2)
        THROW_IE_EXCEPTION << "Expected only 2 blobs with trained parameters (weights and biases), "
                           << "but provided only " << blobs.size();
    if (inShapes.empty()) THROW_IE_EXCEPTION << "No input tensors.";

    size_t D = inShapes[0].back();
    size_t S = rnn->hidden_size;
    size_t expectetd_w_size = G * S * (D + S);
    size_t expectetd_b_size = G * S;

    if (rnn->cellType == RNNCellBase::GRU_LBR) expectetd_b_size = (G + 1) * S;

    auto w = blobs.find("weights");
    if (w == blobs.end()) THROW_IE_EXCEPTION << "Weights blob is not provided";

    if (w->second->size() != expectetd_w_size)
        THROW_IE_EXCEPTION << "Weights blob has wrang size. Expected " << expectetd_w_size;

    auto b = blobs.find("biases");
    if (b == blobs.end()) THROW_IE_EXCEPTION << "Biases blob is not provided";

    if (b->second->size() != expectetd_b_size)
        THROW_IE_EXCEPTION << "Biases blob has wrang size. Expected " << expectetd_b_size;
}

template <RNNSequenceLayer::CellType CELL>
RNNCellValidator<CELL>::RNNCellValidator(const std::string& _type): RNNBaseValidator(_type, CELL) {}

template <RNNSequenceLayer::CellType CELL>
void RNNCellValidator<CELL>::checkShapes(const CNNLayer* layer, const vector<SizeVector>& inShapes) const {
    auto rnn = dynamic_cast<const RNNCellBase*>(layer);
    if (!rnn) THROW_IE_EXCEPTION << "Layer is not instance of RNNSequenceLayer class";

    const size_t& NS = RNNCellValidator<CELL>::NS;

    if (inShapes.size() != NS + 1) THROW_IE_EXCEPTION << "Wrong number of input tensors. Expected " << NS + 1;

    if (inShapes[0].size() != 2) THROW_IE_EXCEPTION << "First input data tensor should be 2D";

    size_t N = inShapes[0][0];
    size_t D = inShapes[0][1];
    size_t S = rnn->hidden_size;

    SizeVector expected_state_shape {N, S};

    if (inShapes[1] != expected_state_shape) THROW_IE_EXCEPTION << "Wrong shape of first initial state tensors.";
    //                         << " Expected " << expected_state_shape << " but provided " << inShapes[1];

    if (NS == 2 && inShapes[2] != expected_state_shape)
        THROW_IE_EXCEPTION << "Wrong shape of second initial state tensors.";
    //                         << " Expected " << expected_state_shape << " but provided " << inShapes[2];
}

template class details::RNNCellValidator<RNNSequenceLayer::RNN>;
template class details::RNNCellValidator<RNNSequenceLayer::GRU>;

//////////////////////////////////////////////////////////

void ProposalValidator::parseParams(CNNLayer* layer) {
    if (layer->params.find("num_outputs") == layer->params.end()) {
        layer->params["num_outputs"] = std::to_string(layer->outData.size());
    }
}

void ProposalValidator::checkParams(const CNNLayer* layer) {
    unsigned int post_nms_topn_ = layer->GetParamAsUInt("post_nms_topn");

    if (layer->CheckParamPresence("feat_stride")) unsigned int feat_stride_ = layer->GetParamAsUInt("feat_stride");
    if (layer->CheckParamPresence("base_size")) unsigned int base_size_ = layer->GetParamAsUInt("base_size");
    if (layer->CheckParamPresence("min_size")) unsigned int min_size_ = layer->GetParamAsUInt("min_size");
    if (layer->CheckParamPresence("pre_nms_topn")) unsigned int pre_nms_topn_ = layer->GetParamAsUInt("pre_nms_topn");
    if (layer->CheckParamPresence("nms_thresh")) {
        float nms_thresh_ = layer->GetParamAsFloat("nms_thresh");
        if (nms_thresh_ < 0) {
            THROW_IE_EXCEPTION << "The value of Proposal layer nms_thresh_ parameter is invalid";
        }
    }
}

void ProposalValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    checkNumOfInput(inShapes, {3});
}

ProposalValidator::ProposalValidator(const std::string& _type): LayerValidator(_type) {}

//////////////////////////////////////////////////////////

void SimplerNMSValidator::checkParams(const CNNLayer* layer) {
    unsigned int post_nms_topn_ = layer->GetParamAsUInt("post_nms_topn");

    if (layer->CheckParamPresence("min_bbox_size")) unsigned int min_box_size_ = layer->GetParamAsUInt("min_bbox_size");
    if (layer->CheckParamPresence("feat_stride")) unsigned int feat_stride_ = layer->GetParamAsUInt("feat_stride");
    if (layer->CheckParamPresence("pre_nms_topn")) unsigned int pre_nms_topn_ = layer->GetParamAsUInt("pre_nms_topn");
    if (layer->CheckParamPresence("iou_threshold")) {
        float iou_threshold_ = layer->GetParamAsFloat("iou_threshold");
        if (iou_threshold_ < 0) {
            THROW_IE_EXCEPTION << "The value of SimplerNMS layer iou_threshold_ parameter is invalid";
        }
    }
    if (layer->CheckParamPresence("scale")) std::vector<unsigned int> scale = layer->GetParamAsUInts("scale", {});
    if (layer->CheckParamPresence("cls_threshold")) {
        float cls_threshold = layer->GetParamAsFloat("cls_threshold");
        if (cls_threshold < 0) {
            THROW_IE_EXCEPTION << "The value of SimplerNMS layer cls_threshold parameter is invalid";
        }
    }
}

void SimplerNMSValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    checkNumOfInput(inShapes, {3});
}

SimplerNMSValidator::SimplerNMSValidator(const std::string& _type): LayerValidator(_type) {}

//////////////////////////////////////////////////////////

void SpatialTransformerValidator::checkParams(const CNNLayer* layer) {
    LayerValidator::checkParams(layer);
}

void SpatialTransformerValidator::checkShapes(const CNNLayer* layer, const std::vector<SizeVector>& inShapes) const {
    checkNumOfInput(inShapes, {2});
}

SpatialTransformerValidator::SpatialTransformerValidator(const std::string& _type): LayerValidator(_type) {}

//////////////////////////////////////////////////////////

UniqueValidator::UniqueValidator(const std::string& _type): LayerValidator(_type) {}

void UniqueValidator::parseParams(CNNLayer* layer) {
    auto casted = dynamic_cast<UniqueLayer*>(layer);
    if (!casted) {
        THROW_IE_EXCEPTION << layer->name << " Layer is not instance of Unique class";
    }

    casted->sorted = layer->GetParamAsBool("sorted");
    casted->return_inverse = layer->GetParamAsBool("return_inverse");
    casted->return_counts = layer->GetParamAsBool("return_counts");
}

void UniqueValidator::checkShapes(const CNNLayer* layer, const vector<SizeVector>& inShapes) const {
    size_t numInputs = inShapes.size();
    if (numInputs != 1)
        THROW_IE_EXCEPTION << layer->name << " Unique can take only 1 input, but actually it has: " << numInputs;
}

//////////////////////////////////////////////////////////

#define REG_LAYER_VALIDATOR_FOR_TYPE(__validator, __type) _validators[#__type] = std::make_shared<__validator>(#__type)

LayerValidators::LayerValidators() {
    REG_LAYER_VALIDATOR_FOR_TYPE(ProposalValidator, Proposal);
    REG_LAYER_VALIDATOR_FOR_TYPE(SimplerNMSValidator, SimplerNMS);
    REG_LAYER_VALIDATOR_FOR_TYPE(SpatialTransformerValidator, SpatialTransformer);
    REG_LAYER_VALIDATOR_FOR_TYPE(SparseFillEmptyRowsValidator, SparseFillEmptyRows);
    REG_LAYER_VALIDATOR_FOR_TYPE(SparseSegmentReduceValidator, SparseSegmentMean);
    REG_LAYER_VALIDATOR_FOR_TYPE(SparseSegmentReduceValidator, SparseSegmentSqrtN);
    REG_LAYER_VALIDATOR_FOR_TYPE(SparseSegmentReduceValidator, SparseSegmentSum);
    REG_LAYER_VALIDATOR_FOR_TYPE(RNNCellValidator<RNNSequenceLayer::RNN>, RNNCell);
    REG_LAYER_VALIDATOR_FOR_TYPE(RNNCellValidator<RNNSequenceLayer::GRU>, GRUCell);
    REG_LAYER_VALIDATOR_FOR_TYPE(SparseToDenseValidator, SparseToDense);
    REG_LAYER_VALIDATOR_FOR_TYPE(UniqueValidator, Unique);
}

}  // namespace InferenceEngine
