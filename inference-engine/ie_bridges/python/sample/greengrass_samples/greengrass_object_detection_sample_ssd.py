"""
BSD 3-clause "New" or "Revised" license

Copyright (C) 2018 Intel Coporation.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import sys
import os
import cv2
import numpy as np
import greengrasssdk
import boto3
import timeit
import datetime
import json
from collections import OrderedDict 

from openvino.inference_engine import IENetwork, IEPlugin

# Specify the delta in seconds between each report
reporting_interval = 1.0

# Parameters for IoT Cloud
enable_iot_cloud_output = True

# Parameters for Kinesis
enable_kinesis_output = False
kinesis_stream_name = ""
kinesis_partition_key = ""
kinesis_region = ""

# Parameters for S3
enable_s3_jpeg_output = False
s3_bucket_name = "ssd_test"

# Parameters for jpeg output on local disk
enable_local_jpeg_output = False

# Create a Greengrass Core SDK client for publishing messages to AWS Cloud
client = greengrasssdk.client("iot-data")

# Create an S3 client for uploading files to S3
if enable_s3_jpeg_output:
    s3_client = boto3.client("s3")

# Create a Kinesis client for putting records to streams
if enable_kinesis_output:
    kinesis_client = boto3.client("kinesis", "us-west-2")

# Read environment variables set by Lambda function configuration
PARAM_MODEL_XML = os.environ.get("PARAM_MODEL_XML")
PARAM_INPUT_SOURCE = os.environ.get("PARAM_INPUT_SOURCE")
PARAM_DEVICE = os.environ.get("PARAM_DEVICE")
PARAM_OUTPUT_DIRECTORY = os.environ.get("PARAM_OUTPUT_DIRECTORY")
PARAM_CPU_EXTENSION_PATH = os.environ.get("PARAM_CPU_EXTENSION_PATH")
PARAM_LABELMAP_FILE = os.environ.get("PARAM_LABELMAP_FILE")
PARAM_TOPIC_NAME = os.environ.get("PARAM_TOPIC_NAME", "intel/faas/ssd")

def report(res_json, frame):
    now = datetime.datetime.now()
    date_prefix = str(now).replace(" ", "_")
    if enable_iot_cloud_output:
        data = json.dumps(res_json)
        client.publish(topic=PARAM_TOPIC_NAME, payload=data)
    if enable_kinesis_output:
        kinesis_client.put_record(StreamName=kinesis_stream_name, Data=json.dumps(res_json), PartitionKey=kinesis_partition_key)
    if enable_s3_jpeg_output:
        temp_image = os.path.join(PARAM_OUTPUT_DIRECTORY, "inference_result.jpeg")
        cv2.imwrite(temp_image, frame)
        with open(temp_image) as file:
            image_contents = file.read()
            s3_client.put_object(Body=image_contents, Bucket=s3_bucket_name, Key=date_prefix + ".jpeg") 
    if enable_local_jpeg_output:
        cv2.imwrite(os.path.join(PARAM_OUTPUT_DIRECTORY, date_prefix + ".jpeg"), frame)

    
def greengrass_object_detection_sample_ssd_run():
    client.publish(topic=PARAM_TOPIC_NAME, payload="OpenVINO: Initializing...")
    model_bin = os.path.splitext(PARAM_MODEL_XML)[0] + ".bin"

    # Plugin initialization for specified device and load extensions library if specified
    plugin = IEPlugin(device=PARAM_DEVICE, plugin_dirs="")
    if "CPU" in PARAM_DEVICE:
        plugin.add_cpu_extension(PARAM_CPU_EXTENSION_PATH)
    # Read IR
    net = IENetwork.from_ir(model=PARAM_MODEL_XML, weights=model_bin)
    assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"
    assert len(net.outputs) == 1, "Sample supports only single output topologies"
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    # Read and pre-process input image
    n, c, h, w = net.inputs[input_blob]
    cap = cv2.VideoCapture(PARAM_INPUT_SOURCE)
    exec_net = plugin.load(network=net)
    del net
    client.publish(topic=PARAM_TOPIC_NAME, payload="Starting inference on %s" % PARAM_INPUT_SOURCE)
    start_time = timeit.default_timer()
    inf_seconds = 0.0
    frame_count = 0
    labeldata = None
    if PARAM_LABELMAP_FILE is not None:
       with open(PARAM_LABELMAP_FILE) as labelmap_file:
            labeldata = json.load(labelmap_file)
    
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        frameid = cap.get(cv2.CAP_PROP_POS_FRAMES)
        initial_w = cap.get(3)
        initial_h = cap.get(4)
        in_frame = cv2.resize(frame, (w, h))
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        in_frame = in_frame.reshape((n, c, h, w))
        # Start synchronous inference
        inf_start_time = timeit.default_timer()
        res = exec_net.infer(inputs={input_blob: in_frame})
        inf_seconds += timeit.default_timer() - inf_start_time
        # Parse detection results of the current request
        res_json = OrderedDict() 
        frame_timestamp = datetime.datetime.now()    
        object_id = 0
        for obj in res[out_blob][0][0]:
             if obj[2] > 0.5:
                 xmin = int(obj[3] * initial_w)
                 ymin = int(obj[4] * initial_h)
                 xmax = int(obj[5] * initial_w)
                 ymax = int(obj[6] * initial_h)
                 cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 165, 20), 4)
                 obj_id = "Object" + str(object_id)
                 classlabel = labeldata[str(int(obj[1]))] if labeldata else ""
                 res_json[obj_id] = {"label": int(obj[1]), "class": classlabel, "confidence": round(obj[2], 2), "xmin": round(obj[3], 2), "ymin": round(obj[4], 2), "xmax": round(obj[5], 2), "ymax": round(obj[6], 2)}
                 object_id += 1
        frame_count += 1
        # Measure elapsed seconds since the last report
        seconds_elapsed = timeit.default_timer() - start_time
        if seconds_elapsed >= reporting_interval:
            res_json["timestamp"] = frame_timestamp.isoformat()
            res_json["frame_id"] = int(frameid)   
            res_json["inference_fps"] = frame_count / inf_seconds
            start_time = timeit.default_timer()
            report(res_json, frame)
            frame_count = 0
            inf_seconds = 0.0

    client.publish(topic=PARAM_TOPIC_NAME, payload="End of the input, exiting...")
    del exec_net
    del plugin

greengrass_object_detection_sample_ssd_run()

def function_handler(event, context):
    client.publish(topic=PARAM_TOPIC_NAME, payload='HANDLER_CALLED!')
    return
