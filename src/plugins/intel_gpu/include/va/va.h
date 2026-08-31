// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef OV_GPU_USE_SYSTEM_LIBVA_HEADERS
# include <va.h>
#else
using VASurfaceID = cl_uint;
using VADisplay = void *;
using VAImageFormat = void *;
#endif
