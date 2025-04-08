/*******************************************************************************
* Copyright (c) 2022-2023 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/// @file
/// C++ API

#pragma once

#ifdef _WIN32
#include "../../../common/core/cm/base_ops.hpp"
#include "../../../common/core/cm/base_types.hpp"
#include "../../../common/core/cm/common.hpp"
#else
#include "common/core/cm/base_ops.hpp"
#include "common/core/cm/base_types.hpp"
#include "common/core/cm/common.hpp"
#endif

namespace gpu::xetla {

/// @addtogroup xetla_core_raw_send
/// @{

/// @brief Raw send with one source operand and one destination operand.
///
/// @tparam T1 is the data type of the msgDst.
/// @tparam n1 is the data length of the msgDst.
/// @tparam T2 is the data type of the msgSrc0.
/// @tparam n2 is the data length of the msgSrc0.
/// @tparam execSize is the execution size.
/// @tparam sfid is the shared function ID.
/// @tparam numSrc0 is the number of GRFs for source-0.
/// @tparam numDst is the number of GRFs for destination.
/// @tparam isEOT is the flag that indicates whether this is an EOT message (optional - default to 0).
/// @tparam isSendc is the flag that indicates whether sendc should be used (optional - default to 0).
/// @tparam N is the SIMD channel num of the mask (optional - default to 16).
/// @param msgDst  [in|out] is the destination operand of the send message.
/// @param msgSrc0 [in] is the first source operand of the send message.
/// @param exDesc  [in] is the extended message descriptor.
/// @param msgDesc [in] is the message descriptor.
/// @param mask    [in] is the predicate to specify enabled channels (optional - default to on).
///
template <typename T1, uint32_t n1, typename T2, uint32_t n2, uint8_t execSize,
        uint8_t sfid, uint8_t numSrc0, uint8_t numDst, uint8_t isEOT = 0,
        uint8_t isSendc = 0, int N = 16>
__XETLA_API void xetla_raw_send(xetla_vector_ref<T1, n1> __REF__ msgDst,
        xetla_vector<T2, n2> msgSrc0, uint32_t exDesc, uint32_t msgDesc,
        xetla_mask<N> mask = 1) {
    cm_raw_send(msgDst, msgSrc0, 0, exDesc, msgDesc, execSize, sfid, numSrc0, 0,
            numDst, isEOT, isSendc, mask);
}

/// @brief Raw send with two source operands and one destination operand.
///
/// @tparam T1 is the data type of the msgDst.
/// @tparam n1 is the data length of the msgDst.
/// @tparam T2 is the data type of the msgSrc0.
/// @tparam n2 is the data length of the msgSrc0.
/// @tparam T3 is the data type of the msgSrc1.
/// @tparam n3 is the data length of the msgSrc1.
/// @tparam execSize is the execution size.
/// @tparam sfid is the shared function ID.
/// @tparam numSrc0 is the number of GRFs for source-0.
/// @tparam numSrc1 is the number of GRFs for source-1.
/// @tparam numDst is the number of GRFs for destination.
/// @tparam isEOT is the flag that indicates whether this is an EOT message (optional - default to 0).
/// @tparam isSendc is the flag that indicates whether sendc should be used (optional - default to 0).
/// @tparam N is the SIMD channel num of the mask (optional - default to 16).
/// @param msgDst  [in|out] is the destination operand of the send message.
/// @param msgSrc0 [in] is the first source operand of the send message.
/// @param msgSrc1 [in] is the second source operand of the send message.
/// @param exDesc  [in] is the extended message descriptor.
/// @param msgDesc [in] is the message descriptor.
/// @param mask    [in] is the predicate to specify enabled channels (optional - default to on).
///
template <typename T1, uint32_t n1, typename T2, uint32_t n2, typename T3,
        uint32_t n3, uint8_t execSize, uint8_t sfid, uint8_t numSrc0,
        uint8_t numSrc1, uint8_t numDst, uint8_t isEOT = 0, uint8_t isSendc = 0,
        int N = 16>
__XETLA_API void xetla_raw_send(xetla_vector_ref<T1, n1> __REF__ msgDst,
        xetla_vector<T2, n2> msgSrc0, xetla_vector<T3, n3> msgSrc1,
        uint32_t exDesc, uint32_t msgDesc, xetla_mask<N> mask = 1) {
    cm_raw_send(msgDst, msgSrc0, msgSrc1, exDesc, msgDesc, execSize, sfid,
            numSrc0, numSrc1, numDst, isEOT, isSendc, mask);
}

/// @brief Raw send with one source operand and no return.
///
/// @tparam T1 is the data type of the msgSrc0.
/// @tparam n1 is the data length of the msgSrc0.
/// @tparam execSize is the execution size.
/// @tparam sfid is the shared function ID.
/// @tparam numSrc0 is the number of GRFs for source-0.
/// @tparam isEOT is the flag that indicates whether this is an EOT message (optional - default to 0).
/// @tparam isSendc is the flag that indicates whether sendc should be used (optional - default to 0).
/// @tparam N is the SIMD channel num of the mask (optional - default to 16).
/// @param msgSrc0 [in] is the first source operand of the send message.
/// @param exDesc  [in] is the extended message descriptor.
/// @param msgDesc [in] is the message descriptor.
/// @param mask    [in] is the predicate to specify enabled channels (optional - default to on).
///
template <typename T1, uint32_t n1, uint8_t execSize, uint8_t sfid,
        uint8_t numSrc0, uint8_t isEOT = 0, uint8_t isSendc = 0, int N = 16>
__XETLA_API void xetla_raw_send(xetla_vector<T1, n1> msgSrc0, uint32_t exDesc,
        uint32_t msgDesc, xetla_mask<N> mask = 1) {
    cm_raw_send(0, msgSrc0, 0, exDesc, msgDesc, execSize, sfid, numSrc0, 0, 0,
            isEOT, isSendc, mask);
}

/// @brief Raw send with two source operands and no return.
///
/// @tparam T1 is the data type of the msgSrc0.
/// @tparam n1 is the data length of the msgSrc0.
/// @tparam T2 is the data type of the msgSrc1.
/// @tparam n2 is the data length of the msgSrc1.
/// @tparam execSize is the execution size.
/// @tparam sfid is the shared function ID.
/// @tparam numSrc0 is the number of GRFs for source-0.
/// @tparam numSrc1 is the number of GRFs for source-1.
/// @tparam isEOT is the flag that indicates whether this is an EOT message (optional - default to 0).
/// @tparam isSendc is the flag that indicates whether sendc should be used (optional - default to 0).
/// @tparam N is the SIMD channel num of the mask (optional - default to 16).
/// @param msgSrc0 [in] is the first source operand of the send message.
/// @param msgSrc1 [in] is the second source operand of the send message.
/// @param exDesc  [in] is the extended message descriptor.
/// @param msgDesc [in] is the message descriptor.
/// @param mask    [in] is the predicate to specify enabled channels (optional - default to on).
///
template <typename T1, uint32_t n1, typename T2, uint32_t n2, uint8_t execSize,
        uint8_t sfid, uint8_t numSrc0, uint8_t numSrc1, uint8_t isEOT = 0,
        uint8_t isSendc = 0, int N = 16>
__XETLA_API void xetla_raw_send(xetla_vector<T1, n1> msgSrc0,
        xetla_vector<T2, n2> msgSrc1, uint32_t exDesc, uint32_t msgDesc,
        xetla_mask<N> mask = 1) {
    cm_raw_send(0, msgSrc0, msgSrc1, exDesc, msgDesc, execSize, sfid, numSrc0,
            numSrc1, 0, isEOT, isSendc, mask);
}

/// @} xetla_core_raw_send

} // namespace gpu::xetla
