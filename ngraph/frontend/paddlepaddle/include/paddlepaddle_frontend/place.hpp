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

#include <frontend_manager/frontend_manager.hpp>

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

class PlacePDPD : public Place {
public:
    PlacePDPD(const InputModel& input_model, const std::vector<std::string>& names)
            : m_input_model(input_model),
              m_names(names) {
    }

    explicit PlacePDPD(const InputModel& input_model) : PlacePDPD(input_model, std::vector<std::string>{}) {
    }

    ~PlacePDPD() override = default;

    bool isInput() const override;

    bool isOutput() const override;

    bool isEqual(Ptr another) const override { return this == another.get(); }

    std::vector<std::string> getNames() const override { return m_names; }

private:
    std::vector<std::string> m_names;
    const InputModel& m_input_model;
};

class InPortPlacePDPD : public PlacePDPD {
public:
    InPortPlacePDPD(const InputModel& input_model,
                    const std::weak_ptr<TensorPlacePDPD>& from_tensor,
                    const std::weak_ptr<OpPlacePDPD>& consuming_op)
            : PlacePDPD(input_model),
              m_tensor(from_tensor),
              m_consuming_op(consuming_op) {
    }

private:
    std::weak_ptr<TensorPlacePDPD> m_tensor;
    std::weak_ptr<OpPlacePDPD> m_consuming_op;
};

class OutPortPlacePDPD : public PlacePDPD {
public:
    OutPortPlacePDPD(const InputModel& input_model,
                     const std::weak_ptr<OpPlacePDPD>& producing_op,
                     const std::weak_ptr<TensorPlacePDPD>& to_tensor)
            : PlacePDPD(input_model),
              m_producing_op(producing_op),
              m_tensor(to_tensor) {
    }

private:
    std::weak_ptr<OpPlacePDPD> m_producing_op;
    std::weak_ptr<TensorPlacePDPD> m_tensor;
};

class OpPlacePDPD : public PlacePDPD {
public:
    OpPlacePDPD(const InputModel& input_model,
                const std::vector<std::string>& names,
                const std::shared_ptr<paddle::framework::proto::OpDesc>& op_desc);

    OpPlacePDPD(const InputModel& input_model,
                const std::shared_ptr<paddle::framework::proto::OpDesc>& op_desc);

    void addInput(const std::weak_ptr<TensorPlacePDPD>& input, const std::string& name) {
        m_input_names.push_back(name);
        m_input_name_to_idx[name].push_back(m_inputs_tensors.size());
        m_inputs_tensors.push_back(input);
    }

    void addOutput(const std::weak_ptr<TensorPlacePDPD>& output, const std::string& name) {
        m_output_names.push_back(name);
        m_output_name_to_idx[name].push_back(m_outputs_tensors.size());
        m_outputs_tensors.push_back(output);
    }

    const std::vector<std::weak_ptr<TensorPlacePDPD>>& getOutputTensors() const {
        return m_outputs_tensors;
    }

    const std::vector<std::weak_ptr<TensorPlacePDPD>>& getInputTensors() const {
        return m_inputs_tensors;
    }

    std::shared_ptr<TensorPlacePDPD> getOutputTensorByName(const std::string& name, int idx = 0) {
        return m_outputs_tensors[m_output_name_to_idx.at(name)[idx]].lock();
    }

    std::shared_ptr<TensorPlacePDPD> getInputTensorByName(const std::string& name, int idx = 0) {
        return m_inputs_tensors[m_input_name_to_idx.at(name)[idx]].lock();
    }

    std::string getInputNameByIdx(int idx) {
        return m_input_names[idx];
    }

    std::string getOutputNameByIdx(int idx) {
        return m_output_names[idx];
    }

    const std::shared_ptr<paddle::framework::proto::OpDesc>& getDesc() const { return m_op_desc; }

private:
    std::shared_ptr<paddle::framework::proto::OpDesc> m_op_desc;
    std::map<std::reference_wrapper<const std::string>, std::vector<int>, std::less<std::string>> m_input_name_to_idx;
    std::map<std::reference_wrapper<const std::string>, std::vector<int>, std::less<std::string>> m_output_name_to_idx;

    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_names;
    std::vector<std::weak_ptr<TensorPlacePDPD>> m_inputs_tensors;
    std::vector<std::weak_ptr<TensorPlacePDPD>> m_outputs_tensors;
};

class TensorPlacePDPD : public PlacePDPD {
public:
    TensorPlacePDPD(const InputModel& input_model,
                    const std::vector<std::string>& names,
                    const std::shared_ptr<paddle::framework::proto::VarDesc>& var_desc);

    TensorPlacePDPD(const InputModel& input_model,
                    const std::shared_ptr<paddle::framework::proto::VarDesc>& var_desc);

    void addInput(const std::weak_ptr<OpPlacePDPD>& input) {
        m_producing_ops.push_back(input);
    }

    void addOutput(const std::weak_ptr<OpPlacePDPD>& output) {
        m_consuming_ops.push_back(output);
    }

    const PartialShape& getPartialShape() const { return m_pshape; }

    const element::Type& getElementType() const { return m_type; }

    void setPartialShape(const PartialShape& pshape) { m_pshape = pshape; }

    void setElementType(const element::Type& type) { m_type = type; }

    const std::shared_ptr<paddle::framework::proto::VarDesc>& getDesc() const { return m_var_desc; }

private:
    std::shared_ptr<paddle::framework::proto::VarDesc> m_var_desc;
    PartialShape m_pshape;
    element::Type m_type;

    std::vector<std::weak_ptr<OpPlacePDPD>> m_producing_ops;
    std::vector<std::weak_ptr<OpPlacePDPD>> m_consuming_ops;
};

} // namespace frontend
} // namespace ngraph
