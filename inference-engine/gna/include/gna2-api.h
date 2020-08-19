/*
 @copyright

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
 @file gna2-api.h
 @brief Gaussian and Neural Accelerator (GNA) 2.0 API.
 @nosubgrouping

 ******************************************************************************

 @mainpage GNA 2.0 Introduction

 // TODO:3:API: provide description
    GNA-3.0 introduces acceleration for both Gaussian-Mixture-Model (GMM)
    and Neural-Networks (xNN) groups of algorithms used by different speech
    recognition operations as well as sensing. The GNA supports both GMM
    and xNN operations. GNA can be activated to perform a sequence of basic
    operations which can be any of the GMM or xNN operations and/or additional
    helper functions. These operations are organized in layers which define
    the operation and its properties.

    The GNA-3.0 IP module scalable and configurable, providing and option
    to tuned GNA HW for various algorithms and use-cases. GNA can be tuned
    to optimize Large-Vocabulary Speech-Recognition algorithms which require
    relative large compute power, or be tuned for low-power always-on sensing
    algorithms. GNA-3.0 extends its support for use-cases beyond speech
    such as low-power always-on sensing, therefore it is not limited to these,
    and may be used by other algorithms.

 ******************************************************************************

 @addtogroup GNA2_API Gaussian and Neural Accelerator (GNA) 2.0 API.

 Gaussian mixture models and Neural network Accelerator.

 @note
 API functions are assumed NOT thread-safe until stated otherwise.

 @{
 *****************************************************************************/

#ifndef __GNA2_API_H
#define __GNA2_API_H

#include "gna2-common-api.h"
#include "gna2-device-api.h"
#include "gna2-inference-api.h"
#include "gna2-instrumentation-api.h"
#include "gna2-memory-api.h"
#include "gna2-model-api.h"

#include <stdint.h>

#endif // __GNA2_API_H

/**
 @}
 */
