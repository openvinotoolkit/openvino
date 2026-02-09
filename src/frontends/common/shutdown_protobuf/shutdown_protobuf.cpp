// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <google/protobuf/stubs/common.h>

#include "openvino/shutdown.hpp"

static void shutdown_frontend_resources() {
    google::protobuf::ShutdownProtobufLibrary();
}

OV_REGISTER_SHUTDOWN_CALLBACK(shutdown_frontend_resources)
