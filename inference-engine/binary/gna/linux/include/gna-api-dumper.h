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
 * GNA 2.0 API
 *
 * Gaussian Mixture Models and Neural Network Accelerator Module
 * Model export API functions definitions
 *
*****************************************************************************/


#ifndef __GNA_API_DUMPER_H
#define __GNA_API_DUMPER_H

#if !defined(_WIN32)
#include <assert.h>
#include <stddef.h>
#endif

#include "gna-api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Header describing parameters of dumped model.
 * Structure is partially filled by GnaModelDump with parameters necessary for SueCreek,
 * other fields are populated by user as necessary.
 */
typedef struct _intel_gna_model_header
{
    uint32_t layer_descriptor_base; // Offset in bytes of first layer descriptor in network.
    uint32_t model_size;            // Total size of model in bytes determined by GnaModelDump  including hw descriptors, model data and input/output buffers.
    uint32_t gna_mode;              // Mode of GNA operation, 1 = XNN mode (default), 0 = GMM mode.
    uint32_t layer_count;           // Number of layers in model.

    uint32_t bytes_per_input;       // Network Input resolution in bytes.
    uint32_t bytes_per_output;      // Network Output resolution in bytes.
    uint32_t input_nodes_count;     // Number of network input nodes.
    uint32_t output_nodes_count;    // Number of network output nodes.

    uint32_t input_descriptor_offset;// Offset in bytes of input pointer descriptor field that need to be set for processing.
    uint32_t output_descriptor_offset;// Offset in bytes of output pointer descriptor field that need to be set for processing.

    uint32_t rw_region_size;        // Size in bytes of read-write region of statically linked GNA model.
    float    input_scaling_factor;  // Scaling factor used for quantization of input values.
    float    output_scaling_factor; // Scaling factor used for quantization of output values.

    uint8_t  reserved[12];          // Padding to 64B.
} intel_gna_model_header;

static_assert(64 == sizeof(intel_gna_model_header), "Invalid size of intel_gna_model_header");

/**
 * Definition of callback that is used to allocate memory for exported model data by GnaModelDump.
 * Allocator takes memory size (size) calculated by GNA library as parameter,
 * allocates memory as needed and returns pointer to this memory.
 * In case of allocation error NULL pointer return value is expected
 */
typedef void* (*intel_gna_alloc_cb)(uint32_t size);

/**
 * Dumps the hardware-consumable model to the memory allocated by customAlloc
 * Model should be created through standard API GnaModelCreate function
 * Model will be validated against device kind provided as function argument
 *
 * @param modelId       Id of model created previously with call to GnaModelCreate function.
 * @param deviceVersion    Device on which model will be used
 * @param modelHeader   (out) Header describing parameters of model being dumped.
 * @param status        (out) Status of conversion and dumping.
 * @param customAlloc   Pointer to a function with custom memory allocation. Total model size needs to be passed as parameter.
 * @return Pointer to memory allocated by customAlloc with binary dumped model
 */
GNAAPI void* GnaModelDump(
    gna_model_id modelId,
    gna_device_generation deviceGeneration,
    intel_gna_model_header* modelHeader,
    intel_gna_status_t* status,
    intel_gna_alloc_cb customAlloc);

#ifdef __cplusplus
}
#endif

#endif // __GNA_API_DUMPER_H
