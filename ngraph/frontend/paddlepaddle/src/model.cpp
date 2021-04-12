// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/paddlepaddle_frontend/model.hpp"

#include "framework.pb.h"
#include "utility.hpp"

namespace ngraph {
namespace frontend {

class InputModelPDPD::InputModelPDPDImpl {  
    std::vector<std::vector<std::shared_ptr<OpPlacePDPD>>> op_places;
    std::vector<std::map<std::string, std::shared_ptr<VarPlacePDPD>>> var_places;
    std::shared_ptr<paddle::framework::proto::ProgramDesc> fw_ptr;
    std::ifstream weights_stream;
    bool weights_composed = false;
public:
    std::string path;

    InputModelPDPDImpl (const std::string& _path);
    std::vector<Place::Ptr> getInputs () const;
    std::vector<Place::Ptr> getOutputs () const;
    Place::Ptr getPlaceByTensorName (const std::string& tensorName);
    void overrideAllOutputs (const std::vector<Place::Ptr>& outputs);
    void overrideAllInputs (const std::vector<Place::Ptr>& inputs);
    void extractSubgraph (const std::vector<Place::Ptr>& inputs, const std::vector<Place::Ptr>& outputs);
    void setDefaultShape (Place::Ptr place, const ngraph::Shape&);
    void setPartialShape (Place::Ptr place, const ngraph::PartialShape&);
    
    template<typename T>
    std::vector<T> getWeight(const std::string& name, int64_t tensor_length);
    std::vector<std::shared_ptr<OpPlacePDPD>> getOpPlaces(int i) const { return op_places[i]; } 
    std::map<std::string, std::shared_ptr<VarPlacePDPD>> getVarPlaces(int i) const { return var_places[i]; }
    size_t getBlockNumber() const { return op_places.size(); }
};

InputModelPDPD::InputModelPDPDImpl::InputModelPDPDImpl(const std::string& _path) : path(_path), fw_ptr{std::make_shared<paddle::framework::proto::ProgramDesc>()} {
    std::string ext = ".pdmodel";            
    std::string model_file(path);
    if (path.length() >= ext.length() && (0 == path.compare(path.length() - ext.length(), ext.length(), ext)))
    {
        weights_composed = true;
        auto weights_file = path.replace(path.size() - ext.size(), ext.size(), ".pdiparams");
        weights_stream = std::ifstream(weights_file, std::ios::binary);
        if (!weights_stream || !weights_stream.is_open())
        {
            std::cerr << "Model file cannot be opened" << std::endl;
        }
    } else {
        weights_composed = false;
        model_file += "/__model__";
    }

    std::ifstream pb_stream(model_file, std::ios::binary);
    std::cout << "Model Parsed: " << fw_ptr->ParseFromIstream(&pb_stream) << std::endl;

    std::cout << "Blocks number: " << fw_ptr->blocks().size() << std::endl;
    for (const auto& block : fw_ptr->blocks()) {
        var_places.push_back(std::map<std::string, std::shared_ptr<VarPlacePDPD>>());
        for (int i = 0; i < block.vars().size(); i++) {
            var_places.back()[block.vars()[i].name()] = std::make_shared<VarPlacePDPD>(VarPlacePDPD(&(block.vars()[i])));
        }

        op_places.push_back(std::vector<std::shared_ptr<OpPlacePDPD>>());
        for (int i = 0; i < block.ops_size(); i++) {
            auto& op = block.ops()[i];
            auto op_place = std::make_shared<OpPlacePDPD>(OpPlacePDPD(&op));
            for (const auto &output : op.outputs()) {
                std::vector<std::weak_ptr<VarPlacePDPD>> out_vars;
                for (auto& var_name : output.arguments()) {
                    auto& var = var_places.back().at(var_name);
                    var->producing_ops.push_back(op_place);
                    out_vars.push_back(var);
                }
                op_place->outputs[output.parameter()] = out_vars;
            }
            std::map<std::string, google::protobuf::RepeatedPtrField<std::string>> inputs_dict;
            for (const auto &input : op.inputs()) {
                std::vector<std::weak_ptr<VarPlacePDPD>> in_vars;
                for (auto& var_name : input.arguments()) {
                    auto& var = var_places.back().at(var_name);
                    var->consuming_ops.push_back(op_place);
                    in_vars.push_back(var);
                }
                op_place->inputs[input.parameter()] = in_vars;
            }
            op_places.back().push_back(op_place);
        }
    }
}

template<typename T>
std::vector<T> InputModelPDPD::InputModelPDPDImpl::getWeight(const std::string& name, int64_t tensor_length) {
    std::vector<T> tensor_data(tensor_length, 0);

    std::ifstream is;
    std::ifstream* stream_ptr;
    if (weights_composed) {
        stream_ptr = &weights_stream;
    } else {
        is = std::ifstream(path + "/" + name, std::ios::in | std::ifstream::binary);
        if (!is || !is.is_open())
        {
            std::cout << "File not opened" << std::endl;
        }
        stream_ptr = &is;
    }
    // TODO: validate that this works for types other than FP32
    std::vector<char> header(16, 0);
    stream_ptr->read(&header[0], 16);
    uint32_t dims_len = 0;
    stream_ptr->read(reinterpret_cast<char*>(&dims_len), 4);
    std::vector<char> dims_struct(dims_len, 0);
    stream_ptr->read(&dims_struct[0], dims_len);
    stream_ptr->read(reinterpret_cast<char*>(&tensor_data[0]), tensor_length * sizeof(T));
    return tensor_data;
}

std::vector<Place::Ptr> InputModelPDPD::InputModelPDPDImpl::getInputs () const {
    NOT_IMPLEMENTED("getInputs");
}

std::vector<Place::Ptr> InputModelPDPD::InputModelPDPDImpl::getOutputs () const {
    NOT_IMPLEMENTED("getOutputs");
}

Place::Ptr InputModelPDPD::InputModelPDPDImpl::getPlaceByTensorName (const std::string& tensorName) {
    for (auto var_places_in_block : var_places) {
        if (var_places_in_block.count(tensorName))
            return var_places_in_block.at(tensorName);
    }
    return nullptr;
}

void InputModelPDPD::InputModelPDPDImpl::overrideAllOutputs (const std::vector<Place::Ptr>& outputs) {
    NOT_IMPLEMENTED("overrideAllOutputs");
}

void InputModelPDPD::InputModelPDPDImpl::overrideAllInputs (const std::vector<Place::Ptr>& inputs) {
    NOT_IMPLEMENTED("overrideAllInputs");
}

void InputModelPDPD::InputModelPDPDImpl::extractSubgraph (const std::vector<Place::Ptr>& inputs, const std::vector<Place::Ptr>& outputs) {
    NOT_IMPLEMENTED("extractSubgraph");
}

void InputModelPDPD::InputModelPDPDImpl::setDefaultShape (Place::Ptr place, const ngraph::Shape&) {
    NOT_IMPLEMENTED("setDefaultShape");
}

void InputModelPDPD::InputModelPDPDImpl::setPartialShape (Place::Ptr place, const ngraph::PartialShape&) {
    NOT_IMPLEMENTED("setPartialShape");
}

InputModelPDPD::InputModelPDPD (const std::string& _path) : _impl{std::make_shared<InputModelPDPDImpl>(_path)} {}

//template<typename T>
std::vector<float> InputModelPDPD::getWeight(const std::string& name, int64_t tensor_length) {
    return _impl->getWeight<float>(name, tensor_length);
}

std::vector<std::shared_ptr<OpPlacePDPD>> InputModelPDPD::getOpPlaces(int i) const {
    return _impl->getOpPlaces(i);
}

std::map<std::string, std::shared_ptr<VarPlacePDPD>> InputModelPDPD::getVarPlaces(int i) const {
    return _impl->getVarPlaces(i);
}

size_t InputModelPDPD::getBlockNumber() const {
    return _impl->getBlockNumber();
}

std::vector<Place::Ptr> InputModelPDPD::getInputs () const {
    return _impl->getInputs();
}

std::vector<Place::Ptr> InputModelPDPD::getOutputs () const {
    return _impl->getOutputs();
}

Place::Ptr InputModelPDPD::getPlaceByTensorName (const std::string& tensorName) {
    return _impl->getPlaceByTensorName(tensorName);
}

void InputModelPDPD::overrideAllOutputs (const std::vector<Place::Ptr>& outputs) {
    return _impl->overrideAllOutputs(outputs);
}

void InputModelPDPD::overrideAllInputs (const std::vector<Place::Ptr>& inputs) {
    return _impl->overrideAllInputs(inputs);
}

void InputModelPDPD::extractSubgraph (const std::vector<Place::Ptr>& inputs, const std::vector<Place::Ptr>& outputs) {
    return _impl->extractSubgraph(inputs, outputs);
}

void InputModelPDPD::setDefaultShape (Place::Ptr place, const ngraph::Shape& shape) {
    return _impl->setDefaultShape(place, shape);
}

void InputModelPDPD::setPartialShape (Place::Ptr place, const ngraph::PartialShape& p_shape) {
    return _impl->setPartialShape(place, p_shape);
}

} // namespace frontend
} // namespace ngraph
