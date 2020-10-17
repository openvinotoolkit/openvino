// Copyright (C) 2019-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <fstream>
#include <pugixml.hpp>

#include "test_environment.hpp"

namespace ngraph
{
    namespace test
    {
        Summary* Summary::p_instance = nullptr;
        SummaryDestroyer Summary::destroyer;

        SummaryDestroyer::~SummaryDestroyer() { delete p_instance; }
        void SummaryDestroyer::initialize(Summary* p) { p_instance = p; }
        Summary& Summary::getInstance()
        {
            if (!p_instance)
            {
                p_instance = new Summary();
                destroyer.initialize(p_instance);
            }
            return *p_instance;
        }

        void Summary::updateOPsStats(ngraph::NodeTypeInfo op, PassRate::Statuses status)
        {
            // TODO: Do we need to count skips?
            auto it = opsStats.find(op);
            if (it != opsStats.end())
            {
                auto& passrate = it->second;
                switch (status)
                {
                case PassRate::PASSED: passrate.passed += 1; break;
                case PassRate::FAILED: passrate.failed += 1; break;
                case PassRate::SKIPPED: passrate.skipped += 1; break;
                }
            }
            else
            {
                switch (status)
                {
                case PassRate::PASSED: opsStats[op] = PassRate(1, 0, 0); break;
                case PassRate::FAILED: opsStats[op] = PassRate(0, 1, 0); break;
                case PassRate::SKIPPED: opsStats[op] = PassRate(0, 0, 1); break;
                }
            }
        }

        void TestEnvironment::TearDown()
        {
            std::vector<ngraph::OpSet> opsets;
            opsets.push_back(ngraph::get_opset1());
            opsets.push_back(ngraph::get_opset2());
            opsets.push_back(ngraph::get_opset3());
            opsets.push_back(ngraph::get_opset4());
            opsets.push_back(ngraph::get_opset5());
            std::set<ngraph::NodeTypeInfo> opsInfo;
            for (const auto& opset : opsets)
            {
                const auto& type_info_set = opset.get_type_info_set();
                opsInfo.insert(type_info_set.begin(), type_info_set.end());
            }

            auto& s = Summary::getInstance();
            auto stats = s.getOPsStats();

            pugi::xml_document doc;

            std::ifstream file;
            file.open(reportFileName);

            time_t rawtime;
            struct tm* timeinfo;
            char timeNow[80];

            time(&rawtime);
            // cpplint require to use localtime_r instead which is not available in C++14
            timeinfo = localtime(&rawtime); // NOLINT

            strftime(timeNow, sizeof(timeNow), "%d-%m-%Y %H:%M:%S", timeinfo);

            pugi::xml_node root;
            if (file)
            {
                doc.load_file(reportFileName.c_str());
                root = doc.child("report");
                // Ugly but shorter than to write predicate for find_atrribute() to update existing
                // one
                root.remove_attribute("timestamp");
                root.append_attribute("timestamp").set_value(timeNow);

                root.remove_child("ops_list");
                root.child("results").remove_child(s.deviceName.c_str());
            }
            else
            {
                root = doc.append_child("report");
                root.append_attribute("timestamp").set_value(timeNow);
                root.append_child("results");
            }

            pugi::xml_node opsNode = root.append_child("ops_list");
            for (const auto& op : opsInfo)
            {
                std::string name = std::string(op.name) + "-" + std::to_string(op.version);
                pugi::xml_node entry = opsNode.append_child(name.c_str());
            }

            pugi::xml_node resultsNode = root.child("results");
            pugi::xml_node currentDeviceNode = resultsNode.append_child(s.deviceName.c_str());
            for (const auto& it : stats)
            {
                std::string name =
                    std::string(it.first.name) + "-" + std::to_string(it.first.version);
                pugi::xml_node entry = currentDeviceNode.append_child(name.c_str());
                entry.append_attribute("passed").set_value(
                    std::to_string(it.second.passed).c_str());
                entry.append_attribute("failed").set_value(
                    std::to_string(it.second.failed).c_str());
                entry.append_attribute("skipped").set_value(
                    std::to_string(it.second.skipped).c_str());
                entry.append_attribute("passrate")
                    .set_value(std::to_string(it.second.getPassrate()).c_str());
            }
            bool result = doc.save_file(reportFileName.c_str());
            if (!result)
            {
                std::cout << "Failed to write report to " << reportFileName << "!" << std::endl;
            }
        }
    }
}
