/*******************************************************************************
* Copyright 2018 Intel Corporation
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

/* DO NOT EDIT, AUTO-GENERATED */

#ifndef MKLDNN_DEBUG_H
#define MKLDNN_DEBUG_H

#include "mkldnn_types.h"

#ifdef __cplusplus
extern "C" {
#endif

const char *mkldnn_status2str(mkldnn_status_t v);
const char *mkldnn_dt2str(mkldnn_data_type_t v);
const char *mkldnn_rmode2str(mkldnn_round_mode_t v);
const char *mkldnn_fmt2str(mkldnn_memory_format_t v);
const char *mkldnn_prop_kind2str(mkldnn_prop_kind_t v);
const char *mkldnn_prim_kind2str(mkldnn_primitive_kind_t v);
const char *mkldnn_alg_kind2str(mkldnn_alg_kind_t v);

#ifdef __cplusplus
}
#endif

#endif
