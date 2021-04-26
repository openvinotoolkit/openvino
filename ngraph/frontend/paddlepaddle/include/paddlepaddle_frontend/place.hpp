//*****************************************************************************
// Copyright 2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <frontend_manager/ifrontend_manager.hpp>
#include <paddlepaddle_frontend/utility.hpp>

namespace paddle {
namespace framework {
namespace proto {

class OpDesc;
class VarDesc;

} // proto
} // framework
} // paddle

namespace ngraph {
namespace frontend {

class TensorPlacePDPD;
class OpPlacePDPD;

class PlacePDPD : public IPlace {
public:
    PlacePDPD(const IInputModel& input_model, const std::vector<std::string>& names)
            : m_input_model(input_model),
              m_names(names) {
    }

    explicit PlacePDPD(const IInputModel& input_model) : PlacePDPD(input_model, std::vector<std::string>{}) {
    }

    ~PlacePDPD() override = default;

    bool isInput() const override;

    bool isOutput() const override;

    bool isEqual(Ptr another) const override { return this == another.get(); }

    std::vector<std::string> getNames() const override { return m_names; }

private:
    std::vector<std::string> m_names;
    const IInputModel& m_input_model;
};

class InPortPlacePDPD : public PlacePDPD {
public:
    explicit InPortPlacePDPD(const IInputModel& input_model)
            : PlacePDPD(input_model) {
    }

    void setOp(const std::weak_ptr<OpPlacePDPD>& op) {
        m_op = op;
    }

    void setSourceTensor(const std::weak_ptr<TensorPlacePDPD>& source_tensor) {
        m_source_tensor = source_tensor;
    }

    std::shared_ptr<TensorPlacePDPD> getSourceTensorPDPD() const;

    std::shared_ptr<OpPlacePDPD> getOp();
private:
    std::weak_ptr<TensorPlacePDPD> m_source_tensor;
    std::weak_ptr<OpPlacePDPD> m_op;
};

class OutPortPlacePDPD : public PlacePDPD {
public:
    explicit OutPortPlacePDPD(const IInputModel& input_model)
            : PlacePDPD(input_model) {
    }

    void setOp(const std::weak_ptr<OpPlacePDPD>& op) { m_op = op; }

    void setTargetTensor(const std::weak_ptr<TensorPlacePDPD>& target_tensor) {
        m_target_tensor = target_tensor;
    }

    std::shared_ptr<TensorPlacePDPD> getTargetTensorPDPD() const;

private:
    std::weak_ptr<OpPlacePDPD> m_op;
    std::weak_ptr<TensorPlacePDPD> m_target_tensor;
};

class OpPlacePDPD : public PlacePDPD {
public:
    OpPlacePDPD(const IInputModel& input_model,
                const std::vector<std::string>& names,
                const std::shared_ptr<paddle::framework::proto::OpDesc>& op_desc);

    OpPlacePDPD(const IInputModel& input_model,
                const std::shared_ptr<paddle::framework::proto::OpDesc>& op_desc);

    void addInPort(const std::shared_ptr<InPortPlacePDPD>& input, const std::string& name) {
        m_input_ports[name].push_back(input);
    }

    void addOutPort(const std::shared_ptr<OutPortPlacePDPD>& output, const std::string& name) {
        m_output_ports[name].push_back(output);
    }

    const std::map<std::string, std::vector<std::shared_ptr<OutPortPlacePDPD>>>& getOutputPorts() const {
        return m_output_ports;
    }

    const std::map<std::string, std::vector<std::shared_ptr<InPortPlacePDPD>>>& getInputPorts() const {
        return m_input_ports;
    }

    std::shared_ptr<OutPortPlacePDPD> getOutputPortPDPD(const std::string& name, int idx) {
        return m_output_ports[name][idx];
    }

    std::shared_ptr<InPortPlacePDPD> getInputPortPDPD(const std::string& name, int idx) {
        return m_input_ports[name][idx];
    }

    const std::shared_ptr<paddle::framework::proto::OpDesc>& getDesc() const { return m_op_desc; }

private:
    std::shared_ptr<paddle::framework::proto::OpDesc> m_op_desc;
    std::map<std::string, std::vector<std::shared_ptr<InPortPlacePDPD>>> m_input_ports;
    std::map<std::string, std::vector<std::shared_ptr<OutPortPlacePDPD>>> m_output_ports;
};

class TensorPlacePDPD : public PlacePDPD {
public:
    TensorPlacePDPD(const IInputModel& input_model,
                    const std::vector<std::string>& names,
                    const std::shared_ptr<paddle::framework::proto::VarDesc>& var_desc);

    TensorPlacePDPD(const IInputModel& input_model,
                    const std::shared_ptr<paddle::framework::proto::VarDesc>& var_desc);

    void addProducingPort(const std::shared_ptr<OutPortPlacePDPD>& out_port) {
        m_producing_ports.push_back(out_port);
    }

    void addConsumingPort(const std::shared_ptr<InPortPlacePDPD>& in_port) {
        m_consuming_ports.push_back(in_port);
    }

    std::vector<IPlace::Ptr> getConsumingPorts () const override;

    Ptr getProducingPort () const override;

    const PartialShape& getPartialShape() const { return m_pshape; }

    const element::Type& getElementType() const { return m_type; }

    void setPartialShape(const PartialShape& pshape) { m_pshape = pshape; }

    void setElementType(const element::Type& type) { m_type = type; }

    const std::shared_ptr<paddle::framework::proto::VarDesc>& getDesc() const { return m_var_desc; }

private:
    std::shared_ptr<paddle::framework::proto::VarDesc> m_var_desc;
    PartialShape m_pshape;
    element::Type m_type;

    std::vector<std::weak_ptr<OutPortPlacePDPD>> m_producing_ports;
    std::vector<std::weak_ptr<InPortPlacePDPD>> m_consuming_ports;
};

} // namespace frontend
} // namespace ngraph
