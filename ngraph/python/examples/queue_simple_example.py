# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""
Third example.

User wants to run the inference on set of pictures and store the
results of the inference (e.g. in a database)
The Inference Queue allows him to run inference as parallel jobs.
Note: Utilizes minimalistic API with callbacks.
"""

import numpy as np
import time

from openvino.inference_engine import IECore
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
ie = IECore()
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


# Create callback for InferQueue jobs
def send_to_database(request, status, userdata):
    """User-defined callback function."""
    if status != StatusCode.OK:
        raise RuntimeError('Infer Request returns with ', status)

    pred_class = np.argmax(request.get_result('fc_out'))

    callback_connection = engine.connect()
    callback_query = helpers.db.insert(tab).values(Id=userdata['index'],
                                                   pred_class=int(pred_class))
    callback_connection.execute(callback_query)


# Set callbacks on each job/InferRequest
infer_queue.set_infer_callback(send_to_database)

print('Starting InferQueue...')
start_queue_time = time.time()

for i in range(len(images)):
    # Using async_infer method invokes a blocking call when all jobs are busy.
    # That simplifies feeding next images and remove
    # responibility of synchronization from a user.
    infer_queue.async_infer(inputs={'data': images[i]},
                            userdata={'index': i})
# Wait for all jobs/InferRequests to finish
statuses = infer_queue.wait_all()

end_queue_time = time.time()
queue_time = (end_queue_time - start_queue_time) * 1000
print('Finished InferQueue!')

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
