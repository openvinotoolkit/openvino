// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/paddlepaddle_frontend/model.hpp"

#include "framework.pb.h"
#include "utility.hpp"

namespace ngraph {
namespace frontend {

class InputModelPDPD::InputModelPDPDImpl {
public:
    std::string path;

    InputModelPDPDImpl (const std::string& _path, const InputModel& input_model);
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
    std::vector<std::shared_ptr<OpPlacePDPD>> getOpPlaces(int i) const { return op_places_blocks[i]; }
    std::map<std::string, std::shared_ptr<TensorPlacePDPD>> getVarPlaces(int i) const { return var_places_blocks[i]; }
    size_t getBlockNumber() const { return op_places_blocks.size(); }

private:
    std::vector<std::vector<std::shared_ptr<OpPlacePDPD>>> op_places_blocks;
    std::vector<std::map<std::string, std::shared_ptr<TensorPlacePDPD>>> var_places_blocks;
    std::shared_ptr<paddle::framework::proto::ProgramDesc> fw_ptr;
    std::ifstream weights_stream;
    bool weights_composed = false;
    const InputModel& m_input_model;
};

InputModelPDPD::InputModelPDPDImpl::InputModelPDPDImpl(const std::string& _path, const InputModel& input_model)
    : path(_path),
      fw_ptr{std::make_shared<paddle::framework::proto::ProgramDesc>()},
      m_input_model(input_model) {
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

    const int cnt_of_blocks = fw_ptr->blocks_size();
    const auto& blocks = fw_ptr->blocks();
    var_places_blocks.resize(cnt_of_blocks);
    op_places_blocks.resize(cnt_of_blocks);

    for (int block_idx = 0; block_idx < cnt_of_blocks; block_idx++) {
        const auto& block = blocks[block_idx];
        auto& var_place_block = var_places_blocks[block_idx];
        auto& op_place_block = op_places_blocks[block_idx];

        for (const auto& var : block.vars()) {
            // storing proto in Places? std::make_shared<paddle::framework::proto::VarDesc>(var)
            var_place_block[var.name()] = std::make_shared<TensorPlacePDPD>(m_input_model, &var);
        }

        for (const auto& op : block.ops()) {
            // storing proto in Places? std::make_shared<paddle::framework::proto::OpDesc>(op)
            auto op_place = std::make_shared<OpPlacePDPD>(m_input_model, &op);
            op_place_block.push_back(op_place);

            for (const auto &output : op.outputs()) {
                for (auto &var_name : output.arguments()) {
                    const auto& var_place = var_place_block.at(var_name);

                    var_place->addInput(op_place);
                    op_place->addOutput(var_place, output.parameter());
                }
            }

            for (const auto &input : op.inputs()) {
                for (const auto &var_name : input.arguments()) {
                    auto &var = var_place_block.at(var_name);

                    op_place->addInput(var, input.parameter());
                    var->addOutput(op_place);
                }
            }
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
    for (auto var_places_in_block : var_places_blocks) {
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

InputModelPDPD::InputModelPDPD (const std::string& _path) : _impl{std::make_shared<InputModelPDPDImpl>(_path, *this)} {}

//template<typename T>
std::vector<float> InputModelPDPD::getWeight(const std::string& name, int64_t tensor_length) {
    return _impl->getWeight<float>(name, tensor_length);
}

std::vector<std::shared_ptr<OpPlacePDPD>> InputModelPDPD::getOpPlaces(int i) const {
    return _impl->getOpPlaces(i);
}

std::map<std::string, std::shared_ptr<TensorPlacePDPD>> InputModelPDPD::getVarPlaces(int i) const {
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
