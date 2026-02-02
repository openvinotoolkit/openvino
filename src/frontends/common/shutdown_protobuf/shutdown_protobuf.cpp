// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <google/protobuf/stubs/common.h>

#include "shutdown.hpp"

static void shutdown_frontend_resources() {
    google::protobuf::ShutdownProtobufLibrary();
}

DECLARE_OV_SHUTDOWN_FUNC(shutdown_frontend_resources)
