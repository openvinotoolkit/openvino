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

#pragma once

#include <vector>

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
            /// \param[in]  input        Vector with RNNSequence-like op inputs in following order:
            ///                          X, initial_hidden_state, sequence_lengths, W, R and B.
            ///
            void validate_seq_input_rank_dimension(const std::vector<ngraph::PartialShape>& input);
        } // namespace util
    }     // namespace op
} // namespace ngraph
