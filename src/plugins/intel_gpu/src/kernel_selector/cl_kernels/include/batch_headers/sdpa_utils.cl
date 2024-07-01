/*******************************************************************************
 * Copyright 2024 Intel Corporation
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

#ifndef GPU_OCL_SDPA_UTILS_H
#define GPU_OCL_SDPA_UTILS_H

#define _4D_OFF(tag, x0, x1, x2, x3) \
    (((x0) % tag##_B0) * tag##_SB0 + ((x0) / tag##_B0) * tag##_S0 \
            + ((x1) % tag##_B1) * tag##_SB1 + ((x1) / tag##_B1) * tag##_S1 \
            + ((x2) % tag##_B2) * tag##_SB2 + ((x2) / tag##_B2) * tag##_S2 \
            + ((x3) % tag##_B3) * tag##_SB3 + ((x3) / tag##_B3) * tag##_S3)

#define QRY_OFF(x0, x1, x2, x3) _4D_OFF(QRY, x0, x1, x2, x3)
#define KEY_OFF(x0, x1, x2, x3) _4D_OFF(KEY, x0, x1, x2, x3)
#define VAL_OFF(x0, x1, x2, x3) _4D_OFF(VAL, x0, x1, x2, x3)
#define MSK_OFF(x0, x1, x2, x3) _4D_OFF(MSK, x0, x1, x2, x3)

#define DST_OFF(x0, x1, d, h, w) \
(((x0) % DST_B0) * DST_SB0 + ((x0) / DST_B0) * DST_S0 \
+ ((x1) % DST_B1) * DST_SB1 + ((x1) / DST_B1) * DST_S1)

#endif
