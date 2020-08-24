//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include "ngraph/node.hpp"

namespace ngraph
{
    namespace op
    {
        namespace util
        {
            ///
            /// \brief      Validates static rank and dimension for provided input parameters.
            ///             Additionally input_size dimension is checked for X and W inputs.
            ///             Applies to LSTM, GRU and RNN Sequences.
            ///
            ///
            /// \param[in]  node         Sequence node for which validatin is requested
            /// \param[in]  input        Vector with input parameters
            ///
            void validate_seq_input_rank_dimension(const std::vector<ngraph::PartialShape>& input);
        }
    }
}
