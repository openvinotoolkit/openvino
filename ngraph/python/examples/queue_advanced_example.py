# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""
Fourth example.

User wants to run the inference on set of pictures and store the
results of the inference (e.g. in a database)
The Inference Queue allows him to run inference as parallel jobs.
Note: Utilizes API without callbacks.
"""

import numpy as np
import time

from openvino.inference_engine import Core
from openvino.inference_engine import StatusCode
from openvino.inference_engine import InferQueue

import helpers

# Generate random number of images
images = helpers.generate_random_images(num=20)

# Create database for InferQueue
engine, connection, metadata, tab = helpers.create_sqlalchemy_database('queue')
metadata.create_all(engine)

# Create databse for reference
ref_engine, ref_connection, ref_metadata, ref_tab = \
    helpers.create_sqlalchemy_database('reference')
ref_metadata.create_all(ref_engine)

# Read and Load of network
ie = Core()
ie_network = ie.read_network(
    helpers.get_example_model_path(),
    helpers.get_example_weights_path())
executable_network = ie.load_network(network=ie_network,
                                     device_name='CPU',
                                     config={})

# Calculate reference outputs
ref_start_time = time.time()
for i in range(len(images)):
    res = executable_network.infer({'data': images[i]})
    pred_class = np.argmax(res['fc_out'])
    ref_query = helpers.db.insert(ref_tab).values(Id=i,
                                                  pred_class=int(pred_class))
    ref_connection.execute(ref_query)
ref_end_time = time.time()
ref_time = (ref_end_time - ref_start_time) * 1000


# Create InferQueue with specific number of jobs/InferRequests
infer_queue = InferQueue(network=executable_network, jobs=8)

print('Starting InferQueue...')
gathered_images = 0
start_queue_time = time.time()

for i in range(len(images)):
    req_info = infer_queue.get_idle_request_info()
    if not(req_info['status'] == StatusCode.OK or
           req_info['status'] == StatusCode.INFER_NOT_STARTED):
        raise Exception('Idle request', req_info['id'], 'failed!')
    else:
        if req_info['status'] == StatusCode.OK:
            res = infer_queue[req_info['id']].get_result('fc_out')
            pred_class = np.argmax(res)

            query = helpers.db.insert(tab).values(
                        Id=infer_queue.userdata[req_info['id']]['index'],
                        pred_class=int(pred_class))
            connection.execute(query)
            gathered_images += 1
    infer_queue.async_infer(inputs={'data': images[i]},
                            userdata={'index': i})
# Wait for all jobs/InferRequests to finish
statuses = infer_queue.wait_all()

end_queue_time = time.time()
queue_time = (end_queue_time - start_queue_time) * 1000
print('Finished InferQueue!')
print('InferQueue was able to gather',
      gathered_images,
      'results between infer calls.')

# TODO: Is gathering leftovers a part of a loop?
# Gather leftovers
assert(np.all(np.array(statuses) == StatusCode.OK))
for i in range(len(infer_queue)):
    res = infer_queue[i].get_result('fc_out')
    pred_class = np.argmax(res)
    query = helpers.db.insert(tab).values(
                Id=infer_queue.userdata[i]['index'],
                pred_class=int(pred_class))
    connection.execute(query)

if np.all(np.array(statuses) == StatusCode.OK):
    print('Reference execution time:', ref_time)
    print('Finished InferQueue! Execution time:', queue_time)
    results = helpers.sort_queue_results(
        connection.execute(helpers.db.select([tab])).fetchall())
    ref_results = helpers.sort_queue_results(
        ref_connection.execute(helpers.db.select([ref_tab])).fetchall())
    metadata.drop_all(engine)  # drops all the tables in the database
    ref_metadata.drop_all(ref_engine)
    assert len(results) == len(images)
    for i in range(len(results)):
        assert ref_results[i] == results[i]
else:
    metadata.drop_all(engine)  # drops all the tables in the database
    raise RuntimeError('InferQueue failed to finish!')
