// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "unit_test_utils/mocks/mock_allocator.hpp"
#include "unit_test_utils/mocks/mock_icnn_network.hpp"
#include "unit_test_utils/mocks/mock_ie_ivariable_state.hpp"
#include "unit_test_utils/mocks/mock_iexecutable_network.hpp"
#include "unit_test_utils/mocks/mock_iinfer_request.hpp"
#include "unit_test_utils/mocks/mock_not_empty_icnn_network.hpp"

#include "unit_test_utils/mocks/cpp_interfaces/mock_task_executor.hpp"

#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_async_infer_request_default.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_async_infer_request_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_async_infer_request_thread_safe_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_executable_network_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_executable_thread_safe_async_only.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_executable_thread_safe_default.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_infer_request_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_inference_plugin_internal.hpp"

#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iasync_infer_request_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iexecutable_network_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iinfer_request_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_ivariable_state_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iinference_plugin.hpp"
