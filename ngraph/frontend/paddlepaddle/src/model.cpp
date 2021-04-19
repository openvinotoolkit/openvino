// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <paddlepaddle_frontend/model.hpp>
#include <paddlepaddle_frontend/utility.hpp>

#include <fstream>
#include "framework.pb.h"
#include "decoder.hpp"

namespace ngraph {
namespace frontend {

using namespace paddle::framework::proto;

class InputModelPDPD::InputModelPDPDImpl {
public:

    InputModelPDPDImpl (const std::string& _path, const InputModel& input_model);
    std::vector<Place::Ptr> getInputs () const;
    std::vector<Place::Ptr> getOutputs () const;
    Place::Ptr getPlaceByTensorName (const std::string& tensorName) const;
    void overrideAllOutputs (const std::vector<Place::Ptr>& outputs);
    void overrideAllInputs (const std::vector<Place::Ptr>& inputs);
    void extractSubgraph (const std::vector<Place::Ptr>& inputs, const std::vector<Place::Ptr>& outputs);
    void setDefaultShape (Place::Ptr place, const ngraph::Shape&);
    void setPartialShape (Place::Ptr place, const ngraph::PartialShape&);
    void setElementType (Place::Ptr place, const ngraph::element::Type&);
    
    template<typename T>
    std::vector<T> readWeight(const std::string& name, int64_t tensor_length);
    std::vector<std::shared_ptr<OpPlacePDPD>> getOpPlaces(int i) const { return m_op_places_blocks[i]; }
    std::map<std::string, std::shared_ptr<TensorPlacePDPD>> getVarPlaces(int i) const { return m_var_places_blocks[i]; }
    size_t getBlockNumber() const { return m_op_places_blocks.size(); }

private:
    std::vector<std::vector<std::shared_ptr<OpPlacePDPD>>> m_op_places_blocks;
    std::vector<std::map<std::string, std::shared_ptr<TensorPlacePDPD>>> m_var_places_blocks;
    std::shared_ptr<ProgramDesc> m_fw_ptr;
    std::ifstream m_weights_stream;
    bool m_weights_composed = false;
    const InputModel& m_input_model;
    std::vector<Place::Ptr> m_inputs;
    std::vector<Place::Ptr> m_outputs;
    std::string m_path;
};

InputModelPDPD::InputModelPDPDImpl::InputModelPDPDImpl(const std::string& _path, const InputModel& input_model)
    : m_path(_path),
      m_fw_ptr{std::make_shared<ProgramDesc>()},
      m_input_model(input_model) {
    std::string ext = ".pdmodel";            
    std::string model_file(m_path);
    if (m_path.length() >= ext.length() && (0 == m_path.compare(m_path.length() - ext.length(), ext.length(), ext)))
    {
        m_weights_composed = true;
        auto weights_file = m_path.replace(m_path.size() - ext.size(), ext.size(), ".pdiparams");
        m_weights_stream = std::ifstream(weights_file, std::ios::binary);
        if (!m_weights_stream || !m_weights_stream.is_open())
        {
            std::cerr << "Model file cannot be opened" << std::endl;
        }
    } else {
        m_weights_composed = false;
        model_file += "/__model__";
    }

    std::ifstream pb_stream(model_file, std::ios::binary);
    std::cout << "Model Parsed: " << m_fw_ptr->ParseFromIstream(&pb_stream) << std::endl;

    std::cout << "Blocks number: " << m_fw_ptr->blocks().size() << std::endl;

    const int cnt_of_blocks = m_fw_ptr->blocks_size();
    const auto& blocks = m_fw_ptr->blocks();
    m_var_places_blocks.resize(cnt_of_blocks);
    m_op_places_blocks.resize(cnt_of_blocks);

    for (int block_idx = 0; block_idx < cnt_of_blocks; block_idx++) {
        const auto& block = blocks[block_idx];
        auto& var_place_block = m_var_places_blocks[block_idx];
        auto& op_place_block = m_op_places_blocks[block_idx];

        for (const auto& var : block.vars()) {
            var_place_block[var.name()] = std::make_shared<TensorPlacePDPD>(m_input_model, std::make_shared<VarDesc>(var));
        }

        for (const auto& op : block.ops()) {
            auto op_place = std::make_shared<OpPlacePDPD>(m_input_model, std::make_shared<OpDesc>(op));
            op_place_block.push_back(op_place);

            for (const auto &output : op.outputs()) {
                auto out_port = std::make_shared<OutPortPlacePDPD>(m_input_model);
                op_place->addOutPort(out_port, output.parameter());
                out_port->setOp(op_place);
                for (const auto &var_name : output.arguments()) {
                    const auto& tensor = var_place_block.at(var_name);
                    tensor->addProducingPort(out_port);
                    out_port->addTargetTensor(tensor);
                }
            }

            for (const auto &input : op.inputs()) {
                auto in_port = std::make_shared<InPortPlacePDPD>(m_input_model);
                op_place->addInPort(in_port, input.parameter());
                in_port->setOp(op_place);
                for (const auto &var_name : input.arguments()) {
                    const auto& tensor = var_place_block.at(var_name);
                    tensor->addConsumingPort(in_port);
                    in_port->addSourceTensor(tensor);
                }
            }

            // Determine outputs and inputs
            if (op.type() == "feed") {
                const auto& place = op_place->getOutputPortByName("Out")->getTargetTensor(0);
                const auto& var_place = std::dynamic_pointer_cast<TensorPlacePDPD>(place);
                const auto& tensor_desc = var_place->getDesc()->type().lod_tensor().tensor();
                const auto& dims = tensor_desc.dims();

                var_place->setElementType(TYPE_MAP[tensor_desc.data_type()]);
                var_place->setPartialShape(PartialShape(std::vector<Dimension>(dims.begin(), dims.end())));
                m_inputs.push_back(place);
            } else if (op.type() == "fetch") {
                auto place = op_place->getInputPortByName("X")->getSourceTensor(0);
                m_outputs.push_back(place);
            }
        }
    }
}

template<typename T>
std::vector<T> InputModelPDPD::InputModelPDPDImpl::readWeight(const std::string& name, int64_t tensor_length) {
    std::vector<T> tensor_data(tensor_length, 0);

    std::ifstream is;
    std::ifstream* stream_ptr;
    if (m_weights_composed) {
        stream_ptr = &m_weights_stream;
    } else {
        is = std::ifstream(m_path + "/" + name, std::ios::in | std::ifstream::binary);
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
    return m_inputs;
}

std::vector<Place::Ptr> InputModelPDPD::InputModelPDPDImpl::getOutputs () const {
    return m_outputs;
}

Place::Ptr InputModelPDPD::InputModelPDPDImpl::getPlaceByTensorName (const std::string& tensorName) const {
    for (const auto& var_places_in_block : m_var_places_blocks) {
        if (var_places_in_block.count(tensorName))
            return var_places_in_block.at(tensorName);
    }
    return nullptr;
}

void InputModelPDPD::InputModelPDPDImpl::overrideAllInputs (const std::vector<Place::Ptr>& inputs) {
    // TODO: resolve for different kind of places
    m_inputs = inputs;
}

void InputModelPDPD::InputModelPDPDImpl::overrideAllOutputs (const std::vector<Place::Ptr>& outputs) {
    // TODO: resolve for different kind of places
    m_outputs = outputs;
}

void InputModelPDPD::InputModelPDPDImpl::extractSubgraph (const std::vector<Place::Ptr>& inputs, const std::vector<Place::Ptr>& outputs) {
    overrideAllInputs(inputs);
    overrideAllOutputs(outputs);
}

void InputModelPDPD::InputModelPDPDImpl::setDefaultShape (Place::Ptr place, const ngraph::Shape& shape) {
    NOT_IMPLEMENTED("setDefaultShape");
}

void InputModelPDPD::InputModelPDPDImpl::setPartialShape (Place::Ptr place, const ngraph::PartialShape& p_shape) {
    auto var_place = std::dynamic_pointer_cast<TensorPlacePDPD>(place);
    if (var_place) {
        var_place->setPartialShape(p_shape);
    }
    // TODO: resolve for port places
}

void InputModelPDPD::InputModelPDPDImpl::setElementType (Place::Ptr place, const ngraph::element::Type& type) {
    auto var_place = std::dynamic_pointer_cast<TensorPlacePDPD>(place);
    if (var_place) {
        var_place->setElementType(type);
    }
    // TODO: resolve for port places
}

InputModelPDPD::InputModelPDPD (const std::string& _path) : _impl{std::make_shared<InputModelPDPDImpl>(_path, *this)} {}

template<typename T>
std::vector<T> InputModelPDPD::readWeight(const std::string& name, int64_t tensor_length) {
    return _impl->readWeight<T>(name, tensor_length);
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

Place::Ptr InputModelPDPD::getPlaceByTensorName (const std::string& tensorName) const {
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

void InputModelPDPD::setElementType (Place::Ptr place, const ngraph::element::Type& type) {
    return _impl->setElementType(place, type);
}

} // namespace frontend
} // namespace ngraph
