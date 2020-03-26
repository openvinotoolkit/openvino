/*
 INTEL CONFIDENTIAL
 Copyright 2019 Intel Corporation.

 The source code contained or described herein and all documents related
 to the source code ("Material") are owned by Intel Corporation or its suppliers
 or licensors. Title to the Material remains with Intel Corporation or its suppliers
 and licensors. The Material may contain trade secrets and proprietary
 and confidential information of Intel Corporation and its suppliers and licensors,
 and is protected by worldwide copyright and trade secret laws and treaty provisions.
 No part of the Material may be used, copied, reproduced, modified, published,
 uploaded, posted, transmitted, distributed, or disclosed in any way without Intel's
 prior express written permission.

 No license under any patent, copyright, trade secret or other intellectual
 property right is granted to or conferred upon you by disclosure or delivery
 of the Materials, either expressly, by implication, inducement, estoppel
 or otherwise. Any license under such intellectual property rights must
 be express and approved by Intel in writing.

 Unless otherwise agreed by Intel in writing, you may not remove or alter this notice
 or any other notice embedded in Materials by Intel or Intel's suppliers or licensors
 in any way.
*/

#pragma once

/******************************************************************************
 *
 * Windows Specific Driver Debug interface
 *
 *****************************************************************************/

#include "GnaDrvApi.h"

#if defined(_DEBUG) || !defined(DRIVER)
#define DRV_DEBUG_INTERFACE
#endif

#ifdef  DRV_DEBUG_INTERFACE

# pragma pack (1) // set structure packaging to 1 to ensure alignment and size

/**
 * READ_REG IOCTL - input data
 * Size:    8 B
 */
typedef struct _PGNA_READREG_IN
{
    UINT32              mbarIndex;  // Index of MBAR
    UINT32              regOffset;  // Register offset

} GNA_READREG_IN, *PGNA_READREG_IN;// READ_REG IOCTL - input data

static_assert(8 == sizeof(GNA_READREG_IN), "Invalid size of GNA_READREG_IN");

/**
 * READ_REG IOCTL - output data
 * Size:    8 B
 */
typedef struct _GNA_READREG_OUT
{
    UINT32              regValue;   // Register value
    UINT32              __res;      // 4 B padding to multiple 8 B size

} GNA_READREG_OUT, *PGNA_READREG_OUT;//READ_REG IOCTL - output data

static_assert(8 == sizeof(GNA_READREG_OUT), "Invalid size of GNA_READREG_OUT");

/**
 * WRITE_REG ioctl - input data
 * Size:    16 B
 */
typedef struct _GNA_WRITEREG_IN
{
    UINT32              mbarIndex;  // Index of MBAR
    UINT32              regOffset;  // Register offset
    UINT32              regValue;   // Register value
    UINT32              __res;      // 4 B padding to multiple 8 B size

} GNA_WRITEREG_IN, *PGNA_WRITEREG_IN;// WRITE_REG ioctl - input data

static_assert(16 == sizeof(GNA_WRITEREG_IN), "Invalid size of GNA_WRITEREG_IN");

/******************************************************************************
 *
 * Driver IOCTL interface
 *
 *****************************************************************************/

#define GNA_IOCTL_READ_REG   CTL_CODE(FILE_DEVICE_PCI_GNA, 0x903, METHOD_BUFFERED, FILE_ANY_ACCESS)
#define GNA_IOCTL_WRITE_REG  CTL_CODE(FILE_DEVICE_PCI_GNA, 0x904, METHOD_BUFFERED, FILE_ANY_ACCESS)

#pragma pack ()

#endif // DRV_DEBUG_INTERFACE
