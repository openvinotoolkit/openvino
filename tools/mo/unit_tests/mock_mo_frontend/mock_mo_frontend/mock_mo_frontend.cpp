// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mock_mo_frontend.hpp"
#include "openvino/frontend/manager.hpp"

using namespace ov::frontend;

FeStat FrontEndMockPy::m_stat = {};
ModelStat InputModelMockPy::m_stat = {};
PlaceStat PlaceMockPy::m_stat = {};

std::string MockSetup::m_equal_data_node1 = {};
std::string MockSetup::m_equal_data_node2 = {};
int MockSetup::m_max_input_port_index = 0;
int MockSetup::m_max_output_port_index = 0;

ov::PartialShape InputModelMockPy::m_returnShape = {};

extern "C" MOCK_API FrontEndVersion get_api_version()
{
    return OV_FRONTEND_API_VERSION;
}

extern "C" MOCK_API void* get_front_end_data()
{
    FrontEndPluginInfo* res = new FrontEndPluginInfo();
    res->m_name = "openvino_mock_mo_frontend";
    res->m_creator = []() { return std::make_shared<FrontEndMockPy>(); };

    return res;
}
