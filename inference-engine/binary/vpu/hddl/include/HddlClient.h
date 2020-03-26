//
// Copyright © 2017-2018 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version October 2018). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//


#ifndef HDDL_API_HDDLCLIENT_H
#define HDDL_API_HDDLCLIENT_H
#include <HddlInferData.h>
#include <HddlQuery.h>
#include <HddlGraph.h>
#include <HddlCommon.h>

#include <list>
#include <map>
#include <string>
#include <memory>
#include <atomic>
#include <cstring>


namespace hddl {

class HddlClientImpl;

class HDDL_EXPORT_API HddlClient {
public:

    /**
     * @brief Connect to hddldaemon, work as a proxy for hddl service.
     * @param clientName Client name.
     * @param config A map of configuratoins. Ignored for now.
     */
    HddlClient(const char* clientName, std::map<std::string, std::string> config={});
    HddlClient(std::string clientName, std::map<std::string, std::string> config={});

    ~HddlClient();

    /**
     * @brief check client if init success.
     * @return HDDL_ERROR_NONE if successed. //
     */
    HddlStatusCode checkInitStatus();

    /**
     * @brief Query the status of service.
     * @param type The info type to query. Please refer to HddlQuery.
     * @param query The object that contains the info data.
     * @return Status code of the operation. HDDL_ERROR_NONE if succeeded.
     */
    HddlStatusCode query(QueryType type, HddlQuery* query);

    /**
     * @brief Query the status of service.
     * @param type The info type to query. Please refer to HddlQuery.
     * @return The object that contains the info data..
     */
    HddlQuery query(QueryType type);

    /**
     * @brief Load a graph to hddldaemon from a file and let the daemon to schedule on MYX.
     * @param graph       Reference to an HddlGraph::Ptr returned from this api.
     *                    NOTE: If config "UPDATE_RUNTIME_PRIORITY" is set, this api only updates the "RUNTIME_PRIORITY"
     *                          of existing graph and the graph pointer will not be set, and the application should not use it.
     * @param graphName   Graph name defined by user.
     * @param graphPath   Path to graph file.
     * @param config      A map of configurations. Configurations includes:
     *                    "SUBCLASS":                Integer representing a firmware subclass, default: "0";
     *                    "GRAPH_TAG":               Arbitrary non-empty tag, if set to empty (""), equals to not set, default: ""; If set, the graph will be scheduled by Tag Scheduler.
     *                    "STREAM_ID":               Arbitrary non-empty id, if set to empty (""), equals to not set, default: ""; If set, the graph will be scheduled by Stream Scheduler.
     *                    "DEVICE_TAG":              Arbitrary non-empty tag, if set to empty (""), equals to not set, default: ""; If set, the graph will be scheduled by Bypass Scheduler.
     *                    "BIND_DEVICE":             "True"/"Yes" or "False"/"No", default: "False"; **Note: ONLY available when "DEVICE_TAG" is set.
     *                                               - If "True"/"Yes", inference through the HddlGraph::Ptr returned, will only be run on one device
     *                                                 with the specified "DEVICE_TAG";
     *                                               - If "False"/"No", inference through the HddlGraph::Ptr returned, will be run on all devices
     *                                                 (whose "BIND_DEVICE" is also set to "False" when they were loaded) with the same graph content.
     *                    "RUNTIME_PRIORITY":        Arbitrary integer, default: "0"; **Note: ONLY available when "DEVICE_TAG" is set and "BIND_DEVICE" is "False".
     *                                               When there are multiple devices running a certain graph (a same graph running on
     *                                               multiple devices in Bypass Scheduler), the device with a larger number has a higher priority,
     *                                               and more inference tasks will be fed to it with priority.
     *                    "UPDATE_RUNTIME_PRIORITY": "True"/"Yes" or "False"/"No", default: "False"; **Note: ONLY available when "DEVICE_TAG" is set.
     *                                               This config should be used only when the graph has been loaded already with the same graph content, the same
     *                                               "DEVICE_TAG" as used this time and "BIND_DEVICE" of the loaded graph has been set to "False".
     *                                               This config is only used to update the "RUNTIME_PRIORITY" of previous loaded graph, and the application should keep useing
     *                                               the previous HddlGraph::Ptr (instead of the one returned in this call) to do inference.
     *                                               - If "True"/"Yes": the "RUNTIME_PRIORITY" must be specified with a integer, and it will be set as the new
     *                                                 runtime priority for that graph on that device.
     *                                               - If "False"/"No": load graph to deivce.
     *                                               **Note: If "BIND_DEVICE" of the previously loaded graph was "True/Yes", the behavior of "update runtime priority" is undefined.
     *
     * @return Status code of the operation. HDDL_ERROR_NONE if succeeded.
     */
    HddlStatusCode loadGraph(HddlGraph::Ptr& graph,
                  std::string graphName,
                  std::string graphPath,
                  std::map<std::string, std::string> config={});

    /**
     * @brief Load a graph to hddldaemon from a memory buffer and let the daemon to schedule on MYX.
     * @param graph       Reference to an HddlGraph::Ptr returned from this api.
     *                    NOTE: If config "UPDATE_RUNTIME_PRIORITY" is set, this api only updates the "RUNTIME_PRIORITY"
     *                          of existing graph and the graph pointer will not be set, and the application should not use it.
     * @param graphName   Graph name defined by user.
     * @param graphData   Graph data in memory.
     * @param graphLen    Graph data length.
     * @param config      A map of configurations.
     *                    Refer to:
     *                             HddlStatusCode loadGraph(HddlGraph::Ptr& graph,
     *                                                   std::string graphName,
     *                                                   std::string graphPath,
     *                                                   std::map<std::string, std::string> config={});
     *
     * @return Status code of the operation. HDDL_ERROR_NONE if succeeded.
     */
    HddlStatusCode loadGraph(HddlGraph::Ptr& graph,
                  std::string graphName,
                  const void* graphData,
                  size_t graphLen,
                  std::map<std::string, std::string> config={});

    /**
     * @brief Ask hddldaemon to unload a graph. Daemon will unload and release the graph when no more
     *        client using this graph.
     * @param graph HddlGraph::Ptr point to an graph returned from loadGraph.
     * @return Status code of the operation. HDDL_ERROR_NONE if succeeded.
     */
    HddlStatusCode unloadGraph(HddlGraph::Ptr graph, std::map<std::string, std::string> config = {});

    /**
     * @brief Issue an inference task to hddldaemon and not return until it's finished.
     * @param graph The graph this inference is to be done on.
     * @param inputBlob input data for this inference.
     * @param outputBlob output data for this inference.
     * @param auxBlob aux data for this inference.
     * @param userData userData pass to hddl service side
     * @param output: task Id for this inference.
     * @return Status code of the operation. HDDL_ERROR_NONE if succeeded.
     */
    HddlStatusCode inferTaskSync (
            HddlGraph::Ptr      graph,
            HddlInferData::Ptr  inferData,
            HddlTaskHandle*     taskId = nullptr
    );


    /**
     * @brief Issue an inference task to hddldaemon and return immediately.
     * @param graph The graph this inference is to be done on.
     * @param inputBlob input data for this inference.
     * @param outputBlob output data for this inference.
     * @param auxBlob aux data for this inference.
     * @param userData userData pass to hddl service side
     * @param callback callback function called after receive task done.
     * @param output: task Id for this inference.
     * @return Status code of the operation. HDDL_ERROR_NONE if succeeded.
     */
    HddlStatusCode inferTaskAsync (
            HddlGraph::Ptr                            graph,
            HddlInferData::Ptr                        inferData,
            HddlTaskHandle*                           taskId
    );


    /**
     * @brief Wait inference to be finished.
     * @param taskId The inference to wait.
     * @param timeOut control the behavior of waitInfer.
     *        timeOut = -1 - wait infinitely
     *        timeOut =  0 - check inference status
     *        timeOut >  0 - wait for <timeOut> ms.
     * @return
     */
    HddlStatusCode waitTask (HddlTaskHandle taskId, int64_t timeOut);

    HddlStatusCode cancelTask (HddlTaskHandle taskId);  // to remove this api. no one use this api.

    HddlStatusCode resetDevice(uint32_t deviceId);
    HddlStatusCode resetAllDevices();

private:
    HddlClient(const HddlClient&);
    HddlClient& operator=(const HddlClient&);

private:
    HddlClientImpl *m_impl;
};

} //namespace hddl

#endif //HDDL_API_HDDLCLIENT_H
