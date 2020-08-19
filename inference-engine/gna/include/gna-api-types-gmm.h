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
 * GMM Scoring and Neural Network Accelerator Module
 * API Gaussian Mixture Model types definition
 *
 *****************************************************************************/

#ifndef __GNA_API_TYPES_GMM_H
#define __GNA_API_TYPES_GMM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum _gmm_read_elimination
{
    GMM_NORMAL_OPERATION,
    GMM_READ_ELIMINATION_ENABLED
} gmm_read_elimination;

typedef enum _gmm_calculation_mode
{
    GMM_L2_DISTANCE,
    GMM_L1_DISTANCE,
    GMM_LINF_DISTANCE
} gmm_calculation_mode;

/** GMM Calculation modes */
typedef enum _gmm_mode
{
    GNA_MAXMIX8,                    // MaxMix mode with 1B Inverse Covariances, use with inverseCovariancesForMaxMix8.
    GNA_MAXMIX16,                   // MaxMix mode with 2B Inverse Covariances, use with inverseCovariancesForMaxMix16.
    GNA_GMM_MODES_COUNT             // Number of modes.

} gna_gmm_mode;

/** GMM Data layouts */
typedef enum _gmm_layout
{
    GMM_LAYOUT_FLAT,                // Each data component is grouped by type. gna_gmm_data buffers can be separate.
    GMM_LAYOUT_INTERLEAVED,         // Each data component is grouped by state. gna_gmm_data buffers use single memory buffer.

} gna_gmm_layout;

/** GMM detailed configuration */
typedef struct _gmm_config
{
    gna_gmm_mode    mode;           // Calculation mode.
    gna_gmm_layout  layout;         // Data layout.
    uint32_t        mixtureComponentCount;// Number of mixture components.
    uint32_t        stateCount;     // Number of states.
    uint32_t        maximumScore;   // Maximum Score value above which scores are saturated.

} gna_gmm_config;

/** GMM Data buffers */
typedef union _gmm_covariances
{
    uint8_t* inverseCovariancesForMaxMix8;  // Inverse Covariances buffer, use with GNA_MAXMIX8 gna_gmm_mode.
    uint16_t* inverseCovariancesForMaxMix16;// Inverse Covariances buffer, use with GNA_MAXMIX16 gna_gmm_mode.
} gna_gmm_covariances;

typedef struct _gmm_data
{
    uint8_t* meanValues;                    // Mean values buffer.
    gna_gmm_covariances inverseCovariances; // Inverse Covariances buffer
    uint32_t* gaussianConstants;            // Gaussian constants buffer.

} gna_gmm_data;

/** GMM Layer detailed configuration */
typedef struct _gmm_layer
{
    gna_gmm_config config;          // GMM configuration.
    gna_gmm_data data;              // GMM data buffers.

} gna_gmm_layer;

/** Maximum number of mixture components per GMM State */
const uint32_t GMM_MIXTURE_COMP_COUNT_MAX = 4096;

/** Maximum number of GMM states, active list elements and  */
const uint32_t GMM_STATES_COUNT_MAX = 262144;

/** Size of memory alignment for mean, variance vectors and Gaussian Constants */
const uint32_t GMM_MEM_ALIGNMENT = 8;

/** Mean vector width in bytes */
const uint32_t GMM_MEAN_VALUE_SIZE = 1;

/** Minimum variance vector width in bytes */
const uint32_t GMM_COVARIANCE_SIZE_MIN = 1;

/** Maximum variance vector width in bytes */
const uint32_t GMM_COVARIANCE_SIZE_MAX = 2;

/** Gaussian Constants width in bytes */
const uint32_t GMM_CONSTANTS_SIZE = 4;

/** Score width in bytes */
const uint32_t GMM_SCORE_SIZE = 4;

/** Size of memory alignment for feature vectors */
const uint32_t GMM_FV_MEM_ALIGN = 64;

/** Feature vector width in bytes */
const uint32_t GMM_FV_ELEMENT_SIZE = 1;

/** Maximum number of feature vectors */
const uint32_t GMM_FV_COUNT_MAX = 8;

/** Minimum length of a vector */
const uint32_t GMM_FV_ELEMENT_COUNT_MIN = 24;

/** Maximum length of a vector */
const uint32_t GMM_FV_ELEMENT_COUNT_MAX = 96;

/** The allowed alignment of vector lengths */
const uint32_t GMM_FV_ELEMENT_COUNT_MULTIPLE_OF = 8;

#ifdef __cplusplus
}
#endif

#endif  // ifndef __GNA_API_TYPES_GMM_H
