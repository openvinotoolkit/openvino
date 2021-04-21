//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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

#include <ngraph/opsets/opset7.hpp>
#include "split.hpp"
#include <paddlepaddle_frontend/utility.hpp>

namespace ngraph {
namespace frontend {
namespace pdpd {
namespace op {
    NamedOutputs split(const NodeContext& node) {
        using namespace ngraph;
        using namespace opset7;
        const auto& data = node.get_ng_input("X");
        auto dim = node.get_attribute<int32_t>("axis");
        // todo: 'num' can be list of values, in this case we should create VariadicSplit
        // todo: support VariadicSplit
        auto num_or_sections = node.get_attribute<int32_t>("num");
        auto axis = std::make_shared<Constant>(ngraph::element::i32, Shape{}, dim);

        NamedOutputs named_outputs;
        auto split_outputs = std::make_shared<Split>(data, axis, num_or_sections)->outputs();
        auto out_names = node.get_output_names();
        PDPD_ASSERT(out_names.size() == 1, "Unexpected number of outputs");

        auto it = std::find(out_names.begin(), out_names.end(), "Out");
        PDPD_ASSERT(it != out_names.end(), "Expected output not found");
        for (const auto& split_output : split_outputs) {
            named_outputs[*it].push_back(split_output);
        }
        return named_outputs;
    }
} // namespace op
} // namespace pdpd
} // namespace frontend
} // namespace ngraph