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
/******************************************************************************
 *
 * Gaussian Mixture Models and Neural Network Accelerator Module
 * API Definition
 *
 *****************************************************************************/

#ifndef __GNA_DEBUG_API_H
#define __GNA_DEBUG_API_H

#include <stdint.h>

#include "gna-api.h"
#include "gna-api-status.h"
#include "gna-api-types-gmm.h"
#include "gna-api-types-xnn.h"

#ifdef __cplusplus
extern "C" {
#endif

/******************  GNA Debug API ******************/
/* This API is for internal GNA hardware testing only*/

typedef enum _dbg_action_type
{
    GnaDumpMmio,
    GnaReservedAction,
    GnaZeroMemory,
    GnaDumpMemory,
    GnaDumpXnnDescriptor,
    GnaDumpGmmDescriptor,
    GnaSetXnnDescriptor,
    GnaSetGmmDescriptor,
    GnaReadRegister,
    GnaWriteRegister,
    GnaLogMessage,
    GnaSleep,

    NUM_ACTION_TYPES
} dbg_action_type;

typedef enum _register_operation
{
    Equal,
    And,
    Or
} register_op;

typedef enum _gna_register
{
    GNA_STS          = 0x80,
    GNA_CTRL         = 0x84,
    GNA_MCTL         = 0x88,
    GNA_PTC          = 0x8C,
    GNA_SC           = 0x90,
    GNA_ISI          = 0x94,
    GNA_ISV_LOW      = 0x98,
    GNA_ISV_HIGH     = 0x9C,
    GNA_BP_LOW       = 0xA0,
    GNA_BP_HIGH      = 0xA4,
    GNA_D0i3C        = 0xA8,
    GNA_DESBASE      = 0xB0,
    GNA_BLD          = 0xB4,
    GNA_SAI1_LOW     = 0x100,
    GNA_SAI1_HIGH    = 0x104,
    GNA_SAI2_LOW     = 0x108,
    GNA_SAI2_HIGH    = 0x10C,
    GNA_SAIV         = 0x110

} gna_reg;

typedef enum _gna_set_size
{
    GNA_SET_BYTE =   0,
    GNA_SET_WORD =   1,
    GNA_SET_DWORD =  2,
    GNA_SET_QWORD =  3,
    GNA_SET_XNNLYR = 4,
} gna_set_size;

#if defined(_WIN32)
#pragma warning(disable : 201)
#endif
typedef struct _dbg_action
{
    dbg_action_type action_type;
    const char *filename;
    const char *log_message;
    gna_timeout timeout;

    struct _gna_register_params
    {
        gna_reg gna_register;
        uint32_t reg_value;
        register_op reg_operation;
    } reg_params;

    struct _dbg_output_params
    {
        void *outputs;
        uint32_t outputs_size;
    } output_params;

    struct _debug_xnn_params
    {
        uint64_t xnn_value;
        uint32_t xnn_offset;
        gna_set_size xnn_value_size;
        uint32_t layer_number;
    } xnn_params;

} dbg_action;

/**
 * Adds a custom debug scenario to the model
 * Actions will be performed sequentially in order
 *
 * @param modelId
 * @param nActions
 * @param pActions
 */
GNAAPI intel_gna_status_t GnaModelSetPrescoreScenario(
    gna_model_id modelId,
    uint32_t nActions,
    dbg_action *pActions);

GNAAPI intel_gna_status_t GnaModelSetAfterscoreScenario(
    gna_model_id modelId,
    uint32_t nActions,
    dbg_action *pActions);

#ifdef __cplusplus
}
#endif

#endif // __GNA_DEBUG_API_H
