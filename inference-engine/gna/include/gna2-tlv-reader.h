/*
 @copyright (C) 2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing,
 software distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions
 and limitations under the License.

 SPDX-License-Identifier: Apache-2.0
*/

/**************************************************************************//**
 @file gna2-tlv-reader.h
@brief Intel (R) GNA TLV model export reader header.

To be included when reading from TLV.

Supported languages: C/C++.

 @nosubgrouping

 ******************************************************************************

 @addtogroup GNA2_TLV_READER

 This module can be used to read GNA model is TLV format.

 @{
 *****************************************************************************/

#ifndef __GNA2_TLV_READER_H
#define __GNA2_TLV_READER_H

#include "gna2-tlv.h"

#include <stddef.h>
#include <stdint.h>

/**
 @private
 Internal function. Should not be used directely.
 */
GNA2_TLV_LINKAGE Gna2TlvStatus Gna2TlvCheckValid(const Gna2TlvRecord * tlvRecord, const char* tlvBlobEnd)
{
    GNA2_TLV_EXPECT_NOT_NULL(tlvRecord);

    if (tlvBlobEnd < tlvRecord->value)
    {
        return Gna2TlvStatusOutOfBuffer;
    }
    if((size_t)(tlvBlobEnd - tlvRecord->value) < tlvRecord->length)
    {
        return Gna2TlvStatusOutOfBuffer;
    }
    return Gna2TlvStatusSuccess;
}

/**
 Looks for the first occurrence of the TLV record of the selected tlvTypeToFind.

 Can be used for reading from TLV blob in memory.
 Can be used in loop to eventually find every occurrences of the record with the specified type
 if there is more than one within the blob (tlvArrayBegin and tlvArraySize have to be modified accordingly).

 @param [in] tlvArrayBegin Address of the TLV record to start the seach. TLV formatted GNA model address.
 @param [in] tlvArraySize Byte size of all the TLV records list in the memory which should be searched.
 @param [in] tlvTypeToFind TLV type of the record to find.
 @param [out] outValueLength TLV length of the record found.
 @param [out] outValue TLV value address of the record found.
 @return Status of the operation.
 @retval Gna2TlvStatusSuccess
 @retval Gna2TlvStatusNotFound
 @retval Gna2TlvStatusTlvReadError
 */
GNA2_TLV_LINKAGE Gna2TlvStatus Gna2TlvFindInArray(
    const char* tlvArrayBegin,
    uint32_t tlvArraySize,
    const Gna2TlvType tlvTypeToFind,
    uint32_t *outValueLength,
    void **outValue
)
{
    GNA2_TLV_EXPECT_NOT_NULL(tlvArrayBegin);
    GNA2_TLV_EXPECT_NOT_NULL(outValueLength);
    GNA2_TLV_EXPECT_NOT_NULL(outValue);

    const char* const tlvArrayEnd = tlvArrayBegin + tlvArraySize;
    while (tlvArrayBegin < tlvArrayEnd)
    {
        const Gna2TlvRecord* const currentRecord = (const Gna2TlvRecord*)tlvArrayBegin;
        if (Gna2TlvStatusSuccess != Gna2TlvCheckValid(currentRecord, tlvArrayEnd))
        {
            return Gna2TlvStatusTlvReadError;
        }
        if (tlvTypeToFind == currentRecord->type)
        {
            *outValue = (void *)(currentRecord->value);
            *outValueLength = currentRecord->length;
            return Gna2TlvStatusSuccess;
        }
        tlvArrayBegin = currentRecord->value + currentRecord->length;
    }
    *outValue = NULL;
    *outValueLength = 0;
    return Gna2TlvStatusNotFound;
}

/**
 Checks the correctness of the overall TLV structure of the binary blob.

 This function verifies
  - the TLV version against the TLV reader compatibility.
  - the provided memory region - [tlvArrayBegin, tlvArrayBegin+tlvArraySize) consists of a list of complete TLV records.
    In particular, no record is truncated with the end of the region.

 Can be called before reading from TLV blob in memory.

 @param [in] tlvArrayBegin Address of the TLV record to start the verification. e.g., TLV formatted GNA model address.
 @param [in] tlvArraySize Byte size of all the TLV records list in the memory which should be verified.
 @return Status of the operation.
 @retval Gna2TlvStatusSuccess Successful verification of version and completness.
 @retval Gna2TlvStatusVersionNotFound  The TLV version was not found.
 @retval Gna2TlvStatusVersionNotSupported The TLV version is not supported.
 @retval Gna2TlvStatusSecondaryVersionFound The blob contains more than one record with TLV version.
 @retval Gna2TlvStatusTlvReadError Record is truncated with the end of the region.
 */
GNA2_TLV_LINKAGE Gna2TlvStatus Gna2TlvVerifyVersionAndCohesion(
    const char* tlvArrayBegin,
    uint32_t tlvArraySize
)
{
    void * version = NULL;
    uint32_t length = 0;
    Gna2TlvStatus status = Gna2TlvFindInArray(tlvArrayBegin, tlvArraySize, Gna2TlvTypeTlvVersion, &length, &version);
    if (status != Gna2TlvStatusSuccess)
    {
        return status == Gna2TlvStatusNotFound ? Gna2TlvStatusVersionNotFound : status;
    }
    if(length != GNA2_TLV_VERSION_VALUE_LENGTH || (*(uint32_t*)version) != GNA2_TLV_VERSION)
    {
        return Gna2TlvStatusVersionNotSupported;
    }
    tlvArrayBegin += GNA2_TLV_VERSION_RECORD_SIZE;
    tlvArraySize -= GNA2_TLV_VERSION_RECORD_SIZE;
    status = Gna2TlvFindInArray(tlvArrayBegin, tlvArraySize, Gna2TlvTypeTlvVersion, &length, &version);
    if(status == Gna2TlvStatusNotFound)
    {
        return Gna2TlvStatusSuccess;
    }
    if(status == Gna2TlvStatusSuccess)
    {
        return Gna2TlvStatusSecondaryVersionFound;
    }
    return status;
}
#endif // __GNA2_TLV_READER_H

/**
 @}
 @}
 */
