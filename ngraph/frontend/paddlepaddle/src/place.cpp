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


#include <paddlepaddle_frontend/place.hpp>

using namespace ngraph;
using namespace frontend;

bool PlacePDPD::isInput() const {
    const auto &model_ins = m_input_model.getInputs();

    const auto cmp = [this](const Place::Ptr &p) {
        return p.get() == this;
    };
    return std::find_if(model_ins.begin(), model_ins.end(), cmp) != model_ins.end();
}

bool PlacePDPD::isOutput() const {
    const auto &model_outs = m_input_model.getOutputs();
    const auto cmp = [this](const Place::Ptr &p) {
        return p.get() == this;
    };
    return std::find_if(model_outs.begin(), model_outs.end(), cmp) != model_outs.end();
}

OpPlacePDPD::OpPlacePDPD(const InputModel &input_model, const std::vector<std::string> &names,
                         const std::shared_ptr<paddle::framework::proto::OpDesc> &op_desc)
        : PlacePDPD(input_model, names),
          m_op_desc(op_desc) {

}

OpPlacePDPD::OpPlacePDPD(const InputModel &input_model,
                         const std::shared_ptr<paddle::framework::proto::OpDesc> &op_desc)
        : OpPlacePDPD(input_model, {}, op_desc) {
}

TensorPlacePDPD::TensorPlacePDPD(const InputModel &input_model, const std::vector<std::string> &names,
                                 const std::shared_ptr<paddle::framework::proto::VarDesc> &var_desc)
        : PlacePDPD(input_model, names),
          m_var_desc(var_desc) {
}

TensorPlacePDPD::TensorPlacePDPD(const InputModel &input_model,
                                 const std::shared_ptr<paddle::framework::proto::VarDesc> &var_desc)
        : TensorPlacePDPD(input_model, {}, var_desc) {
}
