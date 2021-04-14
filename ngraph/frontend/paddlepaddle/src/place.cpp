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

bool PDPDPlace::isInput() const {
    const auto &model_ins = m_input_model.getInputs();

    const auto cmp = [this](const Place::Ptr &p) {
        return p.get() == this;
    };
    return std::find_if(model_ins.begin(), model_ins.end(), cmp) != model_ins.end();
}

bool PDPDPlace::isOutput() const {
    const auto &model_outs = m_input_model.getOutputs();
    const auto cmp = [this](const Place::Ptr &p) {
        return p.get() == this;
    };
    return std::find_if(model_outs.begin(), model_outs.end(), cmp) != model_outs.end();
}
