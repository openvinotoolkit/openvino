/*
    Copyright 2018 Intel Corporation.
    This software and the related documents are Intel copyrighted materials,
    and your use of them is governed by the express license under which they
    were provided to you (Intel OBL Software License Agreement (OEM/IHV/ISV
    Distribution & Single User) (v. 11.2.2017) ). Unless the License provides
    otherwise, you may not use, modify, copy, publish, distribute, disclose or
    transmit this software or the related documents without Intel's prior
    written permission.

    This software and the related documents are provided as is, with no
    express or implied warranties, other than those that are expressly
    stated in the License.
*/

/**************************************************************************//**
 @file gna2-validation-api.h
 @brief Gaussian and Neural Accelerator (GNA) 2.0 Validation API.
 @nosubgrouping

 ******************************************************************************

 @addtogroup GNA2_VALIDATION_API Gaussian and Neural Accelerator (GNA) 2.0 Validation API

 API for validating GNA library and devices.

 @{
 *****************************************************************************/

#ifndef __GNA2_VALIDATION_API_H
#define __GNA2_VALIDATION_API_H

#include "gna2-api.h"

#include <stdint.h>


/******************  GNA Debug API ******************/
/* This API is for internal GNA hardware testing only*/

typedef enum _dbg_action_type
{
    Gna2DumpMmio,
    Gna2ReservedAction,
    Gna2ZeroMemory,
    Gna2DumpMemory,
    Gna2DumpXnnDescriptor,
    Gna2DumpGmmDescriptor,
    Gna2SetXnnDescriptor,
    Gna2SetGmmDescriptor,
    Gna2ReadRegister,
    Gna2WriteRegister,
    Gna2LogMessage,
    Gna2Sleep,

    NUM_ACTION_TYPES
} Gna2DebugActionType;

typedef enum _register_operation
{
    Equal,
    And,
    Or
} Gna2DebugRegisterOperation;

typedef enum _gna_register
{
    GNA2_STS          = 0x80,
    GNA2_CTRL         = 0x84,
    GNA2_MCTL         = 0x88,
    GNA2_PTC          = 0x8C,
    GNA2_SC           = 0x90,
    GNA2_ISI          = 0x94,
    GNA2_ISV_LOW      = 0x98,
    GNA2_ISV_HIGH     = 0x9C,
    GNA2_BP_LOW       = 0xA0,
    GNA2_BP_HIGH      = 0xA4,
    GNA2_D0i3C        = 0xA8,
    GNA2_DESBASE      = 0xB0,
    GNA2_IBUFFS       = 0xB4,
    GNA2_SAI1_LOW     = 0x100,
    GNA2_SAI1_HIGH    = 0x104,
    GNA2_SAI2_LOW     = 0x108,
    GNA2_SAI2_HIGH    = 0x10C,
    GNA2_SAIV         = 0x110

} Gna2RegisterType;

typedef enum _gna_set_size
{
    GNA2_SET_BYTE =   0,
    GNA2_SET_WORD =   1,
    GNA2_SET_DWORD =  2,
    GNA2_SET_QWORD =  3,
    GNA2_SET_XNNLYR = 4,
} Gna2SetSize;

#if defined(_WIN32)
#pragma warning(disable : 201)
#endif
typedef struct _dbg_action
{
    Gna2DebugActionType action_type;
    Gna2RegisterType gna_register;
    union
    {
        uint32_t timeout;
        const char *log_message;
        const char *filename;
        uint64_t xnn_value;
        uint32_t reg_value;
        void *outputs;
    };
    union
    {
        Gna2DebugRegisterOperation reg_operation;
        struct
        {
            uint32_t xnn_offset : 29;
            Gna2SetSize xnn_value_size : 3;
        };
        uint32_t outputs_size;
    };
    uint32_t layer_number;
} Gna2DebugAction;

/**
 * Adds a custom debug scenario to the model
 * Actions will be performed sequentially in order
 *
 * @param modelId
 * @param nActions
 * @param pActions
 */
GNA2_API enum Gna2Status Gna2ModelSetPrescoreScenario(
    uint32_t modelId,
    uint32_t nActions,
    Gna2DebugAction *pActions);

GNA2_API enum Gna2Status Gna2ModelSetAfterscoreScenario(
    uint32_t modelId,
    uint32_t nActions,
    Gna2DebugAction *pActions);

#endif // __GNA2_VALIDATION_API_H

/**
 @}
 */
