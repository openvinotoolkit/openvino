"""
Third example.

User wants to run the inference on set of pictures and store the
results of the inference (e.g. in a database)
The Inference Queue allows him to run inference as parallel jobs.
"""

import numpy as np
import pandas as pd
import time
import sqlalchemy as db

from openvino.inference_engine import IECore
# from openvino.inference_engine import TensorDesc
# from openvino.inference_engine import Blob
from openvino.inference_engine import StatusCode
from openvino.inference_engine import InferQueue

from helpers import get_images


def get_reference(executable_network, images):
    """Get reference outputs using synchronous API."""
    return [executable_network.infer({'data': img}) for img in images]


def show_table(tab):
    results = connection.execute(db.select([tab])).fetchall()
    if len(results) != 0:
        df = pd.DataFrame(results)
        df.columns = results[0].keys()
        print(df.head(6))


engine = db.create_engine('sqlite:///queue.sqlite')
connection = engine.connect()
metadata = db.MetaData()

emp = db.Table('emp', metadata,
               db.Column('Id', db.Integer()),
               db.Column('pred_class', db.Integer()))

metadata.create_all(engine)  # Creates the table

# Read images from a folder
images = get_images()

# Read and Load of network
ie = IECore()
ie_network = ie.read_network(
    '/home/jiwaszki/testdata/models/test_model/test_model_fp32.xml',
    '/home/jiwaszki/testdata/models/test_model/test_model_fp32.bin')
executable_network = ie.load_network(network=ie_network,
                                     device_name='CPU',
                                     config={})


# Create InferQueue with specific number of jobs/InferRequests
infer_queue = InferQueue(network=executable_network, jobs=3)

executable_network.infer()

ref_results = []

ref_start_time = time.time()
for i in range(len(images)):
    res = executable_network.infer({'data': images[i]})
    pred_class = np.argmax(res['fc_out'])
    ref_results += [(i, pred_class)]
ref_end_time = time.time()
ref_time = (ref_end_time - ref_start_time) * 1000


def get_results(request, userdata):
    """User-defined callback function."""
    end_time = time.time()
    print('Finished picture', userdata)
    pred_class = np.argmax(request.get_result('fc_out'))

    # tmp_res[userdata['index']] = [(userdata['index'], pred_class)]
    
    # global tmp_res
    # tmp_res += [(userdata['index'], pred_class)]

    engine = db.create_engine('sqlite:///queue.sqlite')
    connection = engine.connect()
    query = db.insert(emp).values(Id=userdata['index'],
                                  pred_class=int(pred_class))
    connection.execute(query)

    times[userdata['index']] = (end_time - userdata['start_time']) * 1000


# tmp_res = [0] * len(images)
# tmp_res = []
# Set callbacks on each job/InferRequest
times = np.zeros((len(images)))
infer_queue.set_infer_callback(get_results)

print('Starting InferQueue...')
start_queue_time = time.time()
for i in range(len(images)):
    start_request_time = time.time()
    infer_queue.async_infer(inputs={'data': images[i]},
                            userdata={'index': i,
                                      'start_time': start_request_time})
    print('Started picture ', i)

# Wait for all jobs/InferRequests to finish!
statuses = infer_queue.wait_all()
end_queue_time = time.time()
queue_time = (end_queue_time - start_queue_time) * 1000

if np.all(np.array(statuses) == StatusCode.OK):
    print('Reference execution time:', ref_time)
    print('Finished InferQueue! Execution time:', queue_time)
    print('Times for each image: ', times)
    show_table(emp)
    results = connection.execute(db.select([emp])).fetchall()
    metadata.drop_all(engine)  # drops all the tables in the database
    reference = get_reference(executable_network, images)
    assert len(results) == len(images)
    for i in range(len(results)):
        assert ref_results[i] == results[i]
else:
    metadata.drop_all(engine)  # drops all the tables in the database
    raise RuntimeError('InferQueue failed to finish!')
