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

#ifndef _HDDL_API_HDDL_QUERY_H_
#define _HDDL_API_HDDL_QUERY_H_

#include <vector>
#include <string>
#include <ctime>
#include <memory>

#include "HddlCommon.h"

namespace hddl {

class HddlQueryImpl;

class HddlGraphLoadStatusImpl;

class HDDL_EXPORT_API HddlGraphLoadStatus
{
public:
    HddlGraphLoadStatus();

    /**
     * @brief Get the name of the graph.
     */
    std::string                getName() const;

    /**
     * @brief Get the fps of the graph running on this device.
     */
    float                      getFps() const;

    /**
     * @brief Get the number of inference against this graph that has been processed on this device.
     */
    uint32_t                   getInferenceNum() const;

    /**
     * @brief Get the runtime priority of the graph on this device.
     */
    int                        getRunTimePriority() const;

    /**
     * @brief Get the time stamp when the graph is loaded onto this device. In milliseconds since 1970-01-01 00:00:00, with timezone info.
     */
    uint64_t                   getLoadTime() const;

    /**
     * @brief Get the time stamp when the graph is loaded onto this device. In string format, e.g. 2018/12/30 17:32:59.
     */
    std::string                getLoadTimeStr(std::string format = "%Y/%m/%d %H:%M:%S") const;

    /**
     * @brief Get the length of time this graph has been running on this device. In milliseconds.
     */
    uint64_t                   getRunTime() const;

    /**
     * @brief Get the length of time this graph has been running on this device. In string format, e.g. 01:12:13.
     */
    std::string                getRunTimeStr(std::string format = "%H:%M:%S") const;

private:
    friend class HddlClientImpl;
    std::shared_ptr<HddlGraphLoadStatusImpl> m_impl;
};

class HDDL_EXPORT_API HddlQuery
{
public:
    HddlQuery();

    /**
     * @brief Check whether this is a valid query.
     *        A query is valid only after it goes through HddlClient::query() API successfully.
     *        A user can only get valid information from a valid query.
     *        Check a query's validity before get information from it.
     */
    bool                       isValid() const;

    /**
     * @brief Get the type of this query.
     */
    QueryType                  getType() const;

    /**
     * @brief Get the total inference fps of HDDL Service.
     */
    float                      getServiceFps() const;

    /**
     * @brief Get the number of device that used in HDDL Service.
     */
    int                        getDeviceNum() const;

    /**
     * @brief Get all devices' name (busNum.portNum, e.g. 10.3) and saved in a vector.
     *        The same index in this vector can be used to find other properties of this device in other related vectors.
     */
    std::vector<std::string>   getDeviceName() const;

    /**
     * @brief Get all devices' total fps and saved in a vector.
     *        The same index in this vector can be used to find other properties of this device in other related vectors.
     */
    std::vector<float>         getDeviceFps() const;

    /**
     * @brief Get all graph(s) that loaded on each device and saved in a vector. Each graph is described in a HddlGraphLoadStatus object.
     *        The same index in this vector can be used to find other properties of this device in other related vectors.
     */
    std::vector<std::vector<HddlGraphLoadStatus>> getGraphLoadStatusOnDevice() const;

    /**
     * @brief Get all devices' thremal info and saved in a vector.
     *        The same index in this vector can be used to find other properties of this device in other related vectors.
     */
    std::vector<float>         getDeviceThermal() const;

    /**
     * @brief Get all devices' id and saved in a vector.
     *        The same index in this vector can be used to find other properties of this device in other related vectors.
     */
    std::vector<uint32_t>      getDeviceId() const;

    /**
     * @brief Get all devices' subclass and saved in a vector.
     *        The same index in this vector can be used to find other properties of this device in other related vectors.
     */
    std::vector<int>           getDeviceSubclass() const;

    /**
     * @brief Get all devices' device_tag and saved in a vector. If one device is not assinged a tag, an empty string is used as a placeholder.
     *        The same index in this vector can be used to find other properties of this device in other related vectors.
     */
    std::vector<std::string>   getDeviceTag() const;

    /**
     * @brief Get all devices' graph_tag and saved in a vector. If one device is not assinged a tag, an empty string is used as a placeholder.
     *        The same index in this vector can be used to find other properties of this device in other related vectors.
     */
    std::vector<std::string>   getGraphTag() const;

    /**
     * @brief Get all devices' stream id and saved in a vector. If one device is not assinged a stream id, an empty string is used as a placeholder.
     *        The same index in this vector can be used to find other properties of this device in other related vectors.
     */
    std::vector<std::string>   getStreamId() const;

    /**
     * @brief Get all devices' used memory in Meta Byte (MB) and saved in a vector.
     *        The same index in this vector can be used to find other properties of this device in other related vectors.
     */
    std::vector<uint32_t>      getDeviceMemoryUsed() const;

    /**
     * @brief Get all devices' total memory in Meta Byte (MB) and saved in a vector.
     *        The same index in this vector can be used to find other properties of this device in other related vectors.
     */
    std::vector<uint32_t>      getDeviceMemoryTotal() const;

    /**
     * @brief Get all devices' rough utilization (%) and saved in a vector.
     *        The same index in this vector can be used to find other properties of this device in other related vectors.
     */
    std::vector<float>         getDeviceUtilization() const;


    /**
     * @brief Get the number of graph that has been loaded to HDDL Service.
     */
    int                        getGraphNum() const;

    /**
     * @brief Get all graphs' name and saved in a vector.
     *        The same index in this vector can be used to find other properties of this graph in other related vectors.
     */
    std::vector<std::string>   getGraphName() const;

    /**
     * @brief Get all graphs' input fps and saved in a vector. The returned fps is the sum of fps from all clients that issued to this graph.
     *        The same index in this vector can be used to find other properties of this graph in other related vectors.
     */
    std::vector<float>         getGraphFpsIn() const;

    /**
     * @brief Get all graphs' output fps and saved in a vector. This returned fps is the sum of fps of devices that runs this graph.
     *        The same index in this vector can be used to find other properties of this graph in other related vectors.
     */
    std::vector<float>         getGraphFpsOut() const;

    /**
    * @brief Get all graphs' waiting requests and saved in a vector.
    *        The same index in this vector can be used to find other properties of this graph in other related vectors.
    */
    std::vector<uint32_t>      getGraphWaitingReq() const;

    /**
     * @brief  Get the number of client that has connected to HDDL Service.
     */
    int                        getClientNum() const;

    /**
     * @brief Get all clients' name and saved in a vector.
     *        The same index in this vector can be used to find other properties of this client in other related vectors.
     */
    std::vector<std::string>   getClientName() const;

    /**
     * @brief Get all clients' fps and saved in a vector. The return fpsed is a mix value (sum) of the fps this client issued to all graphs.
     *        The same index in this vector can be used to find other properties of this client in other related vectors.
     */
    std::vector<float>         getClientFps() const;

    /**
     * @brief Clear the query result saved in this object.
     */
    void                       clear();

    /**
     * @brief Print the information of this query.
     */
    void                       dump(QueryType queryType = QUERY_TYPE_ALL) const;

private:
    friend class HddlClientImpl;
    std::shared_ptr<HddlQueryImpl> m_impl;
};

} // namespace hddl

#endif
