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

/**
* @defgroup GNA2_TLV_COMMON GNA Model TLV Common.
* @defgroup GNA2_TLV_WRITER GNA Model TLV Writer.
* @defgroup GNA2_TLV_READER GNA Model TLV Reader.
*/

/**
 @file gna2-tlv.h
 @brief Intel (R) GNA TLV model export common header.

 Common header defineing TLV format basics, TLV types and function statuses.

 Supported languages: C/C++

  @addtogroup GNA2_TLV_COMMON

 @{

 GNA model exporting for GNA embedded devices in TLV format.

 The content of this documentation was created and preliminarily tested using GNA library version: 02.01.00.0955,
 and is meant as an engineering preview. Some details may change in future releases!

 @nosubgrouping
 */

#ifndef __GNA2_TLV_H
#define __GNA2_TLV_H

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>

/**
 The TLV type is represented by 4 bytes long field.

 Currentely only ASCII bytes (0-127) are used to represent types.

 Examples of generic TLV types:
    TLV Type name                               | ASCII representation
    ------------------------------------------- |---------------------
    Gna2TlvTypeLayerDescriptorArraySize         | "LDAS"
    Gna2TlvTypeLayerNumber                      | "LCNT"
    Gna2TlvTypePadding                          | "PAD\0"
    Gna2TlvTypeUserData                         | "USRD"
    Gna2TlvTypeGnaLibraryVersionString          | "GNAV"
    Gna2TlvTypeTlvVersion                       | "TLVV"

 */
typedef uint32_t Gna2TlvType;

/**
 @private
 Helper macro.
 */
#define GNA2_TLV_IMPL_CHAR_TO_TYPE(CSTR) (*((const Gna2TlvType*)CSTR))

/**
 TLV record of this type contains the byte size of the whole Layers Descriptor Array of the model.

 Equal to "number of layers" x "single layer descriptor byte size".
 */
#define Gna2TlvTypeLayerDescriptorArraySize GNA2_TLV_IMPL_CHAR_TO_TYPE("LDAS")

/**
 TLV record of this type contains the number of layers of the model.
 */
#define Gna2TlvTypeLayerNumber GNA2_TLV_IMPL_CHAR_TO_TYPE("LCNT")

/**
 TLV record of this type contains the Layers Descriptor Array data combined with read only data into single memory region.

 This combination is needed as GNAA35 devices assume the LDA + RO are placed on the same base address register (BAR).
 */
#define Gna2TlvTypeLayerDescriptorAndRoArrayData GNA2_TLV_IMPL_CHAR_TO_TYPE("L&RD")

/**
 TLV record of this type contains the "state" data.

 States data comes as separate TLV record as GNAA35 devices assume the "states" are placed on separate base address register (BAR).
 */
#define Gna2TlvTypeStateData GNA2_TLV_IMPL_CHAR_TO_TYPE("STTD")

/**
 TLV record of this type contains the needed "scratch" region size.

 Scratch size is separate record as GNAA35 devices assume the "scratch" are placed on separate base address register (BAR).
 The firmware must reserve declared amount of memory and map BAR to it.
 */
#define Gna2TlvTypeScratchSize GNA2_TLV_IMPL_CHAR_TO_TYPE("SCRS")

/**
 TLV record of this type is used to enforce that the following record's value is aproppriatelly aligned (offset from the TLV blob beginning).

 For GNAA35 Such TLV record is used before TLV records which contins LDA or states data.
 */
#define Gna2TlvTypePadding GNA2_TLV_IMPL_CHAR_TO_TYPE("PAD\0")

/**
 TLV record of this type contains External Input Buffer size.
 */
#define Gna2TlvTypeExternalInputBufferSize GNA2_TLV_IMPL_CHAR_TO_TYPE("ExIS")

/**
 TLV record of this type contains External Output Buffer size.
 */
#define Gna2TlvTypeExternalOutputBufferSize GNA2_TLV_IMPL_CHAR_TO_TYPE("ExOS")

/**
 TLV record of this type contains user data.
 */
#define Gna2TlvTypeUserData GNA2_TLV_IMPL_CHAR_TO_TYPE("USRD")

/**
 TLV record of this type contains GNA library version which was used to generate the model components.
 */
#define Gna2TlvTypeGnaLibraryVersionString GNA2_TLV_IMPL_CHAR_TO_TYPE("GNAV")

/**
 TLV record of this type contains TLV format version.
 */
#define Gna2TlvTypeTlvVersion GNA2_TLV_IMPL_CHAR_TO_TYPE("TLVV")

/**
 The TLV length is represented by 4 bytes long unsigned integer number.

 The number is little-endian.
 */
typedef uint32_t Gna2TlvLength;

/**
 Helper structure describing single TLV record and usefull for accessing particular TLV recods parts.

 The typical TLV blob contains a list of such records.

   1 |2 |3 |4 |5 |6 |7 | 8| ...
   --|--|--|--|--|--|--|--|------
   T1|T2|T3|T4|L1|L2|L3|L4| Value

 The minimal size of TLV record is 8 bytes (with zero length).

 The maximal size of TLV record is 8 + MAX_UINT32.

 */
typedef struct
{
    Gna2TlvType type;
    Gna2TlvLength length;
    char value[];
} Gna2TlvRecord;

static_assert(sizeof(Gna2TlvRecord) == 8, "Wrong size of Gna2TlvRecord");

/**
 The TLV status is represented by 4 bytes long number.

 The known statuses are as follows

    TLV Status name                             | Description
    ------------------------------------------- |-------------
    Gna2TlvStatusSuccess                        | 0
    Gna2TlvStatusUnknownError                   | 0x1
    Gna2TlvStatusInvalidLdtSize                 | 0x2
    Gna2TlvStatusNotFound                       | 0x10
    Gna2TlvStatusNullNotAllowed                 | 0x20
    Gna2TlvStatusOutOfBuffer                    | 0x40
    Gna2TlvStatusNotSupported                   | 0x80
    Gna2TlvStatusLengthTooBig                   | 0x100
    Gna2TlvStatusLengthOver256MB                | 0x101
    Gna2TlvStatusUserAllocatorError             | 0x200
    Gna2TlvStatusTlvReadError                   | 0x300
    Gna2TlvStatusSecondaryVersionFound          | 0x301
    Gna2TlvStatusVersionNotFound                | 0x302
    Gna2TlvStatusVersionNotSupported            | 0x303

 */
typedef int32_t Gna2TlvStatus;

#define Gna2TlvStatusSuccess 0
#define Gna2TlvStatusUnknownError 0x1
#define Gna2TlvStatusInvalidLdtSize 0x2
#define Gna2TlvStatusNotFound 0x10
#define Gna2TlvStatusNullNotAllowed 0x20
#define Gna2TlvStatusOutOfBuffer 0x40
#define Gna2TlvStatusNotSupported 0x80
#define Gna2TlvStatusLengthTooBig 0x100
#define Gna2TlvStatusLengthOver256MB 0x101
#define Gna2TlvStatusUserAllocatorError 0x200
#define Gna2TlvStatusTlvReadError 0x300
#define Gna2TlvStatusSecondaryVersionFound 0x301
#define Gna2TlvStatusVersionNotFound 0x302
#define Gna2TlvStatusVersionNotSupported 0x303

/**
 * The function pointer which can be used as user allocator when exporting.
 *
 * Such function should allocate memory region of requested number bytes and return its address.
 */
typedef void* Gna2TlvAllocator(uint32_t);

#define GNA2_TLV_GNAA35_REQUIRED_ALIGNEMENT 64
#define GNA2_TLV_GNAA35_LD_SIZE 128
#define GNA2_TLV_LENGTH_SIZE sizeof(Gna2TlvLength)
#define GNA2_TLV_EMPTY_RECORD_SIZE (GNA2_TLV_LENGTH_SIZE + sizeof(Gna2TlvType))
#define GNA2_TLV_VERSION 1
#define GNA2_TLV_VERSION_VALUE_LENGTH sizeof(uint32_t)
#define GNA2_TLV_VERSION_RECORD_SIZE (GNA2_TLV_EMPTY_RECORD_SIZE + GNA2_TLV_VERSION_VALUE_LENGTH)
#define GNA2_TLV_MAX_GNA_VERSION_CSTRING_SIZE 1024
/** @brief The linkage of private GNA TLV API functions. */
#define GNA2_TLV_LINKAGE static

#define GNA2_TLV_EXPECT_NOT_NULL(ADDRESS) {if((ADDRESS) == NULL) return Gna2TlvStatusNullNotAllowed;}
#define GNA2_TLV_EXPECT_NOT_NULL_IF_SIZE_NZ(DATASIZE, ADDRESS) {if((DATASIZE) != 0) GNA2_TLV_EXPECT_NOT_NULL(ADDRESS)}
#define GNA2_TLV_EXPECT_LENGTH_UP_TO_256MB(LENGTH) {if((LENGTH) > (1 << 28)) return Gna2TlvStatusLengthOver256MB;}

#endif // __GNA2_TLV_H

/**
 @}
 @}
 */
