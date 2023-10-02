// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <string>

namespace CPUTestUtils {
    typedef enum {
        undef,
        a,
        ab,
        acb,
        aBc8b,
        aBc16b,
        abcd,
        acdb,
        aBcd8b,
        aBcd16b,
        abcde,
        acdeb,
        aBcde8b,
        aBcde16b,
        // RNN layouts
        abc,
        bac,
        abdc,
        abdec,

        x = a,
        nc = ab,
        ncw = abc,
        nchw = abcd,
        ncdhw = abcde,
        nwc = acb,
        nhwc = acdb,
        ndhwc = acdeb,
        nCw8c = aBc8b,
        nCw16c = aBc16b,
        nChw8c = aBcd8b,
        nChw16c = aBcd16b,
        nCdhw8c = aBcde8b,
        nCdhw16c = aBcde16b,
        // RNN layouts
        tnc = abc,
        /// 3D RNN data tensor in the format (batch, seq_length, input channels).
        ntc = bac,
        /// 4D RNN states tensor in the format (num_layers, num_directions,
        /// batch, state channels).
        ldnc = abcd,
        /// 5D RNN weights tensor in the format (num_layers, num_directions,
        ///  input_channels, num_gates, output_channels).
        ///
        ///  - For LSTM cells, the gates order is input, forget, candidate
        ///    and output gate.
        ///  - For GRU cells, the gates order is update, reset and output gate.
        ldigo = abcde,
        /// 5D RNN weights tensor in the format (num_layers, num_directions,
        /// num_gates, output_channels, input_channels).
        ///
        ///  - For LSTM cells, the gates order is input, forget, candidate
        ///    and output gate.
        ///  - For GRU cells, the gates order is update, reset and output gate.
        ldgoi = abdec,
        /// 4D LSTM projection tensor in the format (num_layers, num_directions,
        /// num_channels_in_hidden_state, num_channels_in_recurrent_projection).
        ldio = abcd,
        /// 4D LSTM projection tensor in the format (num_layers, num_directions,
        /// num_channels_in_recurrent_projection, num_channels_in_hidden_state).
        ldoi = abdc,
        /// 4D RNN bias tensor in the format (num_layers, num_directions,
        /// num_gates, output_channels).
        ///
        ///  - For LSTM cells, the gates order is input, forget, candidate
        ///    and output gate.
        ///  - For GRU cells, the gates order is update, reset and output gate.
        ldgo = abcd,
    } cpu_memory_format_t;

using CPUSpecificParams =  std::tuple<
    std::vector<cpu_memory_format_t>, // input memomry format
    std::vector<cpu_memory_format_t>, // output memory format
    std::vector<std::string>,         // priority
    std::string                       // selected primitive type
>;

std::vector<CPUSpecificParams> filterCPUInfo(const std::vector<CPUSpecificParams>& CPUParams);
std::vector<CPUSpecificParams> filterCPUInfoForArch(const std::vector<CPUSpecificParams>& CPUParams);
std::vector<CPUSpecificParams> filterCPUInfoForDevice(const std::vector<CPUSpecificParams>& CPUParams);
} // namespace CPUTestUtils
