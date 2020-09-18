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
 @file gna2-tlv-writer.h
 @brief Intel (R) GNA TLV model export writer header.

 To be included for building TLV model image (e.g., saving model model to file for use on embedded devices).

 Language supported: C++.

 @nosubgrouping

 ******************************************************************************

 @addtogroup GNA2_TLV_WRITER

 The following description presents a complete flow (in order) of GNA model exporting in TLV format for GNA embedded devices.

 This flow is dependent on GNA library usage.

 Consider to override GNA library default alignment (default is 64) if your target device supports it.

 If needed make sure to call Gna2ModelOverrideAlignment(newAlignment) function before Gna2ModelCreate(...).

 Allocate any number of logically separable memory regions with Gna2MemoryAlloc(...) and tag them with Gna2MemorySetTag(address, tagValue).

    Tag name                        | Tag value
    -------------------------------:|:----------
    MemoryTagReadOnly               | 3
    MemoryTagExternalBufferInput    | 4
    MemoryTagExternalBufferOutput   | 5
    MemoryTagScratch                | 6
    MemoryTagState                  | 7

 When using "External Buffer" feature, the following rules applie:
  - Separately allocate memory regions for external input and/or external output buffers.
  - Tag them appropriately with Gna2MemorySetTag().
  - Set selected (input/bias/output) Gna2Tensor tensors' Mode fields of the selected GNA layers with Gna2TensorModeExternalBuffer
     Gna2TensorMode enum type) value.

 Set up GNA model (Gna2Model structure) and pass this structure to Gna2ModelCreate(...) function.

 Create model export configuration with Gna2ModelExportConfigCreate(...).

 Select model for exporting with Gna2ModelExportConfigSetSource(...).

 Select target device version with Gna2ModelExportConfigSetTarget(...)

  Target device type             | Description      | Designation
  -------------------------------|------------------|------------
  Gna2DeviceVersionEmbedded3_1   | Autonomous GNA   | GNAA35

 Export selected components of model into memory with Gna2ModelExport()

   Component name                                   | Description  | Applicability
   -------------------------------------------------|--------------|---------------------
   Gna2ModelExportComponentLayerDescriptors         | LDT          | GNAA35
   Gna2ModelExportComponentReadOnlyDump             | RO           | GNAA35
   Gna2ModelExportComponentScratchDump              | SCRA         | GNAA35
   Gna2ModelExportComponentStateDump                | STATE        | GNAA35
   Gna2ModelExportComponentExternalBufferInputDump  | EXIN         | GNAA35
   Gna2ModelExportComponentExternalBufferOutputDump | EXOUT        | GNAA35

 In order to create TLV blob pass components to ::Gna2ExportTlvGNAA35(...) helper function.


 @{
 *****************************************************************************/

#ifndef __GNA2_TLV_WRITER_H
#define __GNA2_TLV_WRITER_H

#include "gna2-tlv.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

/**
 @private
 Internal function. Should not be used directely.
 */
GNA2_TLV_LINKAGE Gna2TlvStatus Gna2TlvImplComputePadSize(
    uint32_t currentOffset,
    uint32_t alignment,
    uint32_t minPadSize,
    uint32_t * outPadSize)
{
    GNA2_TLV_EXPECT_NOT_NULL(outPadSize);

    const uint32_t const4k = 4096;
    if (alignment == 0 || alignment > const4k || minPadSize > const4k)
    {
        return Gna2TlvStatusNotSupported;
    }
    uint32_t paddingNeeded = (alignment - (((currentOffset % alignment) + minPadSize) % alignment)) % alignment;
    if (paddingNeeded > 0 && paddingNeeded < minPadSize)
    {
        paddingNeeded += alignment;
    }
    *outPadSize = paddingNeeded;
    return Gna2TlvStatusSuccess;
}

/**
 @private
 Internal function. Should not be used directely.
 */
GNA2_TLV_LINKAGE Gna2TlvStatus Gna2TlvImplPad(char* buf, uint32_t paddingTlvLength)
{
    GNA2_TLV_EXPECT_NOT_NULL(buf);

    const auto elem = reinterpret_cast<Gna2TlvRecord*>(buf);
    elem->type = Gna2TlvTypePadding;
    elem->length = paddingTlvLength;
    buf += GNA2_TLV_EMPTY_RECORD_SIZE;

    for (uint32_t i = 0; i < paddingTlvLength; i++)
    {
        buf[0] = '*';
        buf++;
    }
    return Gna2TlvStatusSuccess;
}

/**
 @private
 Internal function. Should not be used directely.
 */
GNA2_TLV_LINKAGE void Gna2TlvImplCopy(char*& outBuf, const char * src, uint32_t srcLength)
{
    while(srcLength--)
    {
        *outBuf++ = *src++;
    }
}

/**
 @private
 Internal function. Should not be used directely.
 */
GNA2_TLV_LINKAGE void Gna2TlvImplWriteTypeLength(char*&buffer, Gna2TlvType type, Gna2TlvLength length)
{
    const auto element = reinterpret_cast<Gna2TlvRecord*>(buffer);
    element->type = type;
    element->length = length;
    buffer += GNA2_TLV_EMPTY_RECORD_SIZE;
}

/**
 @private
 Internal function. Should not be used directely.
 */
GNA2_TLV_LINKAGE void Gna2TlvImplWrite4BSize(char*&buffer, Gna2TlvType type, uint32_t sizeAsValue)
{
    Gna2TlvImplWriteTypeLength(buffer, type, sizeof(uint32_t));
    *reinterpret_cast<uint32_t*>(buffer) = sizeAsValue;
    buffer += sizeof(uint32_t);
}

/**
 @private
 Internal type. Should not be used directely.
 */
struct Gna2TlvInternalRecord
{
    Gna2TlvType tlvType;
    uint32_t tlvLength;
    const char* tlvValue;
};

/**
 @private
 Internal type. Should not be used directely.
 */
struct Gna2TlvInternalRecordOf4Length
{
    Gna2TlvType tlvType;
    uint32_t tlvValue;
};

/**
 @private
 Internal function. Should not be used directely.
 */
GNA2_TLV_LINKAGE uint32_t Gna2TlvGetCStringByteSizeSat(const char* s)
{
    if (s == NULL)
    {
        return 0;
    }
    const auto sLength = strlen(s);
    return sLength >= UINT32_MAX ? UINT32_MAX : sLength + 1;
}

/**
 Exports TLV formated GNA model into newly allocated memory region.

 This function is for GNA 3.5 devices with autonomous extension present.

 @param [in] userAllocatorIn Address od function to be used for required memory region allocation.
             May be called only once. If called outTlv and outTlvSize are also written.
 @param [out] outTlv Address of memory region allocated with userAllocatorIn where the TLV is serialized.
 @param [out] outTlvSize Size of serialized TLV.
 @param [in] lda [required not null] layer descriptor component, as exported with Gna2ModelExport() and Gna2ModelExportComponentLayerDescriptors.
 @param [in] ldaSize layer descriptor size in bytes, as exported with Gna2ModelExport() and Gna2ModelExportComponentLayerDescriptors.
 @param [in] ro read only model fragment.
 @param [in] roSize read only model part size, must be 0 if ro is NULL.
 @param [in] state state model fragment. Can be NULL.
 @param [in] stateSize state model fragment size, must be 0 if state is NULL.
 @param [in] scratchSize scratch model fragment size.
 @param [in] externalInputSize external input buffer size of model.
 @param [in] externalOutputSize external output buffer size of model.
 @param [in] gnaLibraryVersion GNA library's version c-string obtained with Gna2GetLibraryVersion(), if the model is exported using GNA library, can be NULL otherwise.
 @param [in] userData User data. Can be NULL.
 @param [in] userDataSize User data size, must be 0 if userData is NULL.
 @return Status of the operation.
 */
GNA2_TLV_LINKAGE Gna2TlvStatus Gna2ExportTlvGNAA35(
    Gna2TlvAllocator userAllocatorIn,
    char ** outTlv,
    uint32_t * outTlvSize,
    const char* lda,
    uint32_t ldaSize,
    const char* ro,
    uint32_t roSize,
    const char* state,
    uint32_t stateSize,
    uint32_t scratchSize,
    uint32_t externalInputSize,
    uint32_t externalOutputSize,
    const char* gnaLibraryVersion,
    const char* userData,
    uint32_t userDataSize
)
{
    const uint32_t gnaLibraryVersionLength = Gna2TlvGetCStringByteSizeSat(gnaLibraryVersion);
    if(gnaLibraryVersionLength > GNA2_TLV_MAX_GNA_VERSION_CSTRING_SIZE)
    {
        return Gna2TlvStatusLengthTooBig;
    }

    GNA2_TLV_EXPECT_NOT_NULL(lda);
    GNA2_TLV_EXPECT_NOT_NULL(userAllocatorIn);
    GNA2_TLV_EXPECT_NOT_NULL(outTlv);
    GNA2_TLV_EXPECT_NOT_NULL(outTlvSize);

    GNA2_TLV_EXPECT_NOT_NULL_IF_SIZE_NZ(stateSize, state);
    GNA2_TLV_EXPECT_NOT_NULL_IF_SIZE_NZ(roSize, ro);
    GNA2_TLV_EXPECT_NOT_NULL_IF_SIZE_NZ(userDataSize, userData);

    GNA2_TLV_EXPECT_LENGTH_UP_TO_256MB(ldaSize);
    GNA2_TLV_EXPECT_LENGTH_UP_TO_256MB(roSize);
    GNA2_TLV_EXPECT_LENGTH_UP_TO_256MB(stateSize);
    GNA2_TLV_EXPECT_LENGTH_UP_TO_256MB(scratchSize);
    GNA2_TLV_EXPECT_LENGTH_UP_TO_256MB(externalInputSize);
    GNA2_TLV_EXPECT_LENGTH_UP_TO_256MB(userDataSize);

    if (ldaSize % GNA2_TLV_GNAA35_LD_SIZE != 0)
    {
        return Gna2TlvStatusInvalidLdtSize;
    }
    const uint32_t layerNumber = ldaSize / GNA2_TLV_GNAA35_LD_SIZE;

    const struct Gna2TlvInternalRecordOf4Length recordsBeforeLda[] =
    {
        { Gna2TlvTypeTlvVersion, GNA2_TLV_VERSION },
        { Gna2TlvTypeLayerNumber, layerNumber },
        { Gna2TlvTypeLayerDescriptorArraySize, ldaSize },
        { Gna2TlvTypeScratchSize, scratchSize },
        { Gna2TlvTypeExternalInputBufferSize, externalInputSize },
        { Gna2TlvTypeExternalOutputBufferSize, externalOutputSize },
    };

    const uint32_t numberOfRecordsBeforeLda = sizeof(recordsBeforeLda) / sizeof(recordsBeforeLda[0]);
    const uint32_t sizeOfRecordsBeforeLda = numberOfRecordsBeforeLda * (GNA2_TLV_EMPTY_RECORD_SIZE + sizeof(uint32_t));
    uint32_t outPadSizeLda = 0;
    Gna2TlvStatus status = Gna2TlvImplComputePadSize(sizeOfRecordsBeforeLda, GNA2_TLV_GNAA35_REQUIRED_ALIGNEMENT, GNA2_TLV_EMPTY_RECORD_SIZE, &outPadSizeLda);
    if (status != Gna2TlvStatusSuccess)
    {
        return Gna2TlvStatusOutOfBuffer;
    }
    const uint32_t sizeOfLdaRecord = ldaSize + roSize + GNA2_TLV_EMPTY_RECORD_SIZE;
    const uint32_t sizeOfRecordsBeforeState = sizeOfLdaRecord + outPadSizeLda + sizeOfRecordsBeforeLda;
    uint32_t outPadSizeState = 0;
    status = Gna2TlvImplComputePadSize(sizeOfRecordsBeforeState, GNA2_TLV_GNAA35_REQUIRED_ALIGNEMENT, GNA2_TLV_EMPTY_RECORD_SIZE, &outPadSizeState);
    if (status != Gna2TlvStatusSuccess)
    {
        return Gna2TlvStatusOutOfBuffer;
    }

    const struct Gna2TlvInternalRecord recordsFromState[] =
    {
        { Gna2TlvTypeStateData, stateSize, state },
        { Gna2TlvTypeGnaLibraryVersionString, gnaLibraryVersionLength, gnaLibraryVersion },
        { Gna2TlvTypeUserData, userDataSize, userData },
    };
    uint32_t sizeFromStates = 0;
    for(const auto& record: recordsFromState)
    {
        sizeFromStates += GNA2_TLV_EMPTY_RECORD_SIZE + record.tlvLength;
    }

    const uint32_t totalSizeRequired = sizeOfRecordsBeforeState + outPadSizeState + sizeFromStates;

    *outTlv = (char*)userAllocatorIn(totalSizeRequired);
    *outTlvSize = totalSizeRequired;

    if (*outTlv == NULL)
    {
        return Gna2TlvStatusUserAllocatorError;
    }
    char* curOutBuffer = *outTlv;

    for (const auto& record : recordsBeforeLda)
    {
        Gna2TlvImplWrite4BSize(curOutBuffer, record.tlvType, record.tlvValue);
    }

    if (outPadSizeLda != 0)
    {
        const Gna2TlvStatus status = Gna2TlvImplPad(curOutBuffer, outPadSizeLda - GNA2_TLV_EMPTY_RECORD_SIZE);
        if (status != Gna2TlvStatusSuccess)
        {
            return Gna2TlvStatusOutOfBuffer;
        }
        curOutBuffer += outPadSizeLda;
    }

    Gna2TlvImplWriteTypeLength(curOutBuffer, Gna2TlvTypeLayerDescriptorAndRoArrayData, ldaSize + roSize);
    Gna2TlvImplCopy(curOutBuffer, lda, ldaSize);
    Gna2TlvImplCopy(curOutBuffer, ro, roSize);

    if (outPadSizeState != 0)
    {
        Gna2TlvStatus status = Gna2TlvImplPad(curOutBuffer, outPadSizeState - GNA2_TLV_EMPTY_RECORD_SIZE);
        if (status != Gna2TlvStatusSuccess)
        {
            return Gna2TlvStatusOutOfBuffer;
        }
        curOutBuffer += outPadSizeState;
    }

    for (const auto& record : recordsFromState)
    {
        Gna2TlvImplWriteTypeLength(curOutBuffer, record.tlvType, record.tlvLength);
        Gna2TlvImplCopy(curOutBuffer, record.tlvValue, record.tlvLength);
    }

    if (curOutBuffer != *outTlv + totalSizeRequired)
    {
        return Gna2TlvStatusUnknownError;
    }

    return Gna2TlvStatusSuccess;
}

#endif // __GNA2_TLV_WRITER_H

/**
 @}
 @}
 */
