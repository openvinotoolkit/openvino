OpenVINO optimizations for Knowledge graphs
===========================================

The goal of this notebook is to showcase performance optimizations for
the ConvE knowledge graph embeddings model using the Intel® Distribution
of OpenVINO™ Toolkit. The optimizations process contains the following
steps:

1. Export the trained model to a format suitable for OpenVINO optimizations and inference
2. Report the inference performance speedup obtained with the optimized OpenVINO model

The ConvE model is an implementation of the paper - “Convolutional 2D
Knowledge Graph Embeddings” (https://arxiv.org/abs/1707.01476). The
sample dataset can be downloaded from:
https://github.com/TimDettmers/ConvE/tree/master/countries/countries_S1

Windows specific settings
-------------------------

.. code:: ipython3

    # On Windows, add the directory that contains cl.exe to the PATH
    # to enable PyTorch to find the required C++ tools.
    # This code assumes that Visual Studio 2019 is installed in the default directory.
    # If you have a different C++ compiler, please add the correct path
    # to os.environ["PATH"] directly.
    # Note that the C++ Redistributable is not enough to run this notebook.
    
    # Adding the path to os.environ["LIB"] is not always required
    # - it depends on the system's configuration
    
    import sys
    
    if sys.platform == "win32":
        import distutils.command.build_ext
        import os
        from pathlib import Path
    
        VS_INSTALL_DIR = r"C:/Program Files (x86)/Microsoft Visual Studio"
        cl_paths = sorted(list(Path(VS_INSTALL_DIR).glob("**/Hostx86/x64/cl.exe")))
        if len(cl_paths) == 0:
            raise ValueError(
                "Cannot find Visual Studio. This notebook requires a C++ compiler. If you installed "
                "a C++ compiler, please add the directory that contains"
                "cl.exe to `os.environ['PATH']`."
            )
        else:
            # If multiple versions of MSVC are installed, get the most recent version
            cl_path = cl_paths[-1]
            vs_dir = str(cl_path.parent)
            os.environ["PATH"] += f"{os.pathsep}{vs_dir}"
            # Code for finding the library dirs from
            # https://stackoverflow.com/questions/47423246/get-pythons-lib-path
            d = distutils.core.Distribution()
            b = distutils.command.build_ext.build_ext(d)
            b.finalize_options()
            os.environ["LIB"] = os.pathsep.join(b.library_dirs)
            print(f"Added {vs_dir} to PATH")

Import the packages needed for successful execution
---------------------------------------------------

.. code:: ipython3

    import json
    from pathlib import Path
    import sys
    import time
    
    import numpy as np
    import torch
    from sklearn.metrics import accuracy_score
    from torch.nn import functional as F, Parameter
    from torch.nn.init import xavier_normal_
    
    from openvino.runtime import Core
    
    sys.path.append("../utils")
    from notebook_utils import download_file

Settings: Including path to the serialized model files and input data files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    
    # Path to the pretrained model checkpoint
    modelpath = Path('models/conve.pt')
    
    # Path to the file containing the entities and entity IDs
    entdatapath = Path('../data/text/countries_S1/kg_training_entids.txt')
    
    # Path to the file containing the relations and relation IDs
    reldatapath = Path('../data/text/countries_S1/kg_training_relids.txt')
    
    # Path to the test data file
    testdatapath = Path('../data/json/countries_S1/e1rel_to_e2_ranking_test.json')
    
    # Entity and relation embedding dimensions
    EMB_DIM = 300
    
    # Top K vals to consider from the predictions
    TOP_K = 2
    
    # Required for OpenVINO conversion
    output_dir = Path("models")
    base_model_name = "conve"
    
    output_dir.mkdir(exist_ok=True)
    
    # Paths where PyTorch, ONNX and OpenVINO IR models will be stored
    fp32_onnx_path = Path(output_dir / (base_model_name + "_fp32")).with_suffix(".onnx")


.. parsed-literal::

    Using cpu device


Download Model Checkpoint
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    model_url = 'https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/knowledge-graph-embeddings/conve.pt'
    
    download_file(model_url, filename=modelpath.name, directory=modelpath.parent)



.. parsed-literal::

    models/conve.pt:   0%|          | 0.00/18.8M [00:00<?, ?B/s]




.. parsed-literal::

    PosixPath('/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-448/.workspace/scm/ov-notebook/notebooks/219-knowledge-graphs-conve/models/conve.pt')



Defining the ConvE model class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # Model implementation reference: https://github.com/TimDettmers/ConvE
    class ConvE(torch.nn.Module):
        def __init__(self, num_entities, num_relations, emb_dim):
            super(ConvE, self).__init__()
            # Embedding tables for entity and relations with num_uniq_ent in y-dim, emb_dim in x-dim
            self.emb_e = torch.nn.Embedding(num_entities, emb_dim, padding_idx=0)
            self.ent_weights_matrix = torch.ones([num_entities, emb_dim], dtype=torch.float64)
            self.emb_rel = torch.nn.Embedding(num_relations, emb_dim, padding_idx=0)
            self.ne = num_entities
            self.nr = num_relations
            self.inp_drop = torch.nn.Dropout(0.2)
            self.hidden_drop = torch.nn.Dropout(0.3)
            self.feature_map_drop = torch.nn.Dropout2d(0.2)
            self.loss = torch.nn.BCELoss()
            self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=True)
            self.bn0 = torch.nn.BatchNorm2d(1)
            self.bn1 = torch.nn.BatchNorm2d(32)
            self.ln0 = torch.nn.LayerNorm(emb_dim)
            self.register_parameter('b', Parameter(torch.zeros(num_entities)))
            self.fc = torch.nn.Linear(16128, emb_dim)
    
        def init(self):
            """ Initializes the model """
            # Xavier initialization
            xavier_normal_(self.emb_e.weight.data)
            xavier_normal_(self.emb_rel.weight.data)
    
        def forward(self, e1, rel):
            """ Forward pass on the model.
            :param e1: source entity
            :param rel: relation between the source and target entities
            Returns the model predictions for the target entities
            """
            e1_embedded = self.emb_e(e1).view(-1, 1, 10, 30)
            rel_embedded = self.emb_rel(rel).view(-1, 1, 10, 30)
            stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)
            stacked_inputs = self.bn0(stacked_inputs)
            x = self.inp_drop(stacked_inputs)
            x = self.conv1(x)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.feature_map_drop(x)
            x = x.view(1, -1)
            x = self.fc(x)
            x = self.hidden_drop(x)
            x = self.ln0(x)
            x = F.relu(x)
            x = torch.mm(x, self.emb_e.weight.transpose(1, 0))
            x = self.hidden_drop(x)
            x += self.b.expand_as(x)
            pred = torch.nn.functional.softmax(x, dim=1)
            return pred

Defining the dataloader
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    class DataLoader():
        def __init__(self):
            super(DataLoader, self).__init__()
    
            self.ent_path = entdatapath
            self.rel_path = reldatapath
            self.test_file = testdatapath
            self.entity_ids, self.ids2entities = self.load_data(data_path=self.ent_path)
            self.rel_ids, self.ids2rel = self.load_data(data_path=self.rel_path)
            self.test_triples_list = self.convert_triples(data_path=self.test_file)
    
        def load_data(self, data_path):
            """ Creates a dictionary of data items with corresponding ids """
            item_dict, ids_dict = {}, {}
            fp = open(data_path, "r")
            lines = fp.readlines()
            for line in lines:
                name, id = line.strip().split('\t')
                item_dict[name] = int(id)
                ids_dict[int(id)] = name
            fp.close()
            return item_dict, ids_dict
    
        def convert_triples(self, data_path):
            """ Creates a triple of source entity, relation and target entities"""
            triples_list = []
            dp = open(data_path, "r")
            lines = dp.readlines()
            for line in lines:
                item_dict = json.loads(line.strip())
                h = item_dict['e1']
                r = item_dict['rel']
                t = item_dict['e2_multi1'].split('\t')
                hrt_list = []
                hrt_list.append(self.entity_ids[h])
                hrt_list.append(self.rel_ids[r])
                t_ents = []
                for t_idx in t:
                    t_ents.append(self.entity_ids[t_idx])
                hrt_list.append(t_ents)
                triples_list.append(hrt_list)
            dp.close()
            return triples_list

Evaluate the trained ConvE model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, we will evaluate the model performance using PyTorch. The goal is
to make sure there are no accuracy differences between the original
model inference and the model converted to OpenVINO intermediate
representation inference results. Here, we use a simple accuracy metric
to evaluate the model performance on a test dataset. However, it is
typical to use metrics such as Mean Reciprocal Rank, Hits@10 etc.

.. code:: ipython3

    data = DataLoader()
    num_entities = len(data.entity_ids)
    num_relations = len(data.rel_ids)
    
    model = ConvE(num_entities=num_entities, num_relations=num_relations, emb_dim=EMB_DIM)
    model.load_state_dict(torch.load(modelpath))
    model.eval()
    
    pt_inf_times = []
    
    triples_list = data.test_triples_list
    num_test_samples = len(triples_list)
    pt_acc = 0.0
    for i in range(num_test_samples):
        test_sample = triples_list[i]
        h, r, t = test_sample
        start_time = time.time()
        logits = model.forward(e1=torch.tensor(h), rel=torch.tensor(r))
        end_time = time.time()
        pt_inf_times.append(end_time - start_time)
        score, pred = torch.topk(logits, TOP_K, 1)
    
        gt = np.array(sorted(t))
        pred = np.array(sorted(pred[0].cpu().detach()))
        pt_acc += accuracy_score(gt, pred)
    
    avg_pt_time = np.mean(pt_inf_times) * 1000
    print(f'Average time taken for inference: {avg_pt_time} ms')
    print(f'Mean accuracy of the model on the test dataset: {pt_acc/num_test_samples}')


.. parsed-literal::

    Average time taken for inference: 0.6920695304870605 ms
    Mean accuracy of the model on the test dataset: 0.875


Prediction on the Knowledge graph.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here, we perform the entity prediction on the knowledge graph, as a
sample evaluation task. We pass the source entity ‘san_marino’ and
relation ‘locatedIn’ to the knowledge graph and obtain the target entity
predictions. Expected predictions are target entities that form a
factual triple with the entity and relation passed as inputs to the
knowledge graph.

.. code:: ipython3

    entitynames_dict = data.ids2entities
    
    ent = 'san_marino'
    rel = 'locatedin'
    
    h_idx = data.entity_ids[ent]
    r_idx = data.rel_ids[rel]
    
    logits = model.forward(torch.tensor(h_idx), torch.tensor(r_idx))
    score, pred = torch.topk(logits, TOP_K, 1)
    
    for j, id in enumerate(pred[0].cpu().detach().numpy()):
        pred_entity = entitynames_dict[id]
        print(f'Source Entity: {ent}, Relation: {rel}, Target entity prediction: {pred_entity}')


.. parsed-literal::

    Source Entity: san_marino, Relation: locatedin, Target entity prediction: southern_europe
    Source Entity: san_marino, Relation: locatedin, Target entity prediction: europe


Convert the trained PyTorch model to ONNX format for OpenVINO inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To evaluate performance with OpenVINO, we can either convert the trained
PyTorch model to an intermediate representation (IR) format or to an
ONNX representation. This notebook uses the ONNX format. For more
details on model optimization, refer to:
https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html

.. code:: ipython3

    print('Converting the trained conve model to ONNX format')
    torch.onnx.export(model, (torch.tensor(1), torch.tensor(1)),
                      fp32_onnx_path, input_names=['input.1', 'input.2'], verbose=False, opset_version=11)


.. parsed-literal::

    Converting the trained conve model to ONNX format


Evaluate the model performance with OpenVINO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now, we evaluate the model performance with the OpenVINO framework. In
order to do so, make three main API calls:

1. Initialize the Inference engine with ``Core()``
2. Load the model with ``read_model()``
3. Compile the model with ``compile_model()``

Then, the model can be inferred on by using the
``create_infer_request()`` API call.

.. code:: ipython3

    ie = Core()
    ir_net = ie.read_model(model=fp32_onnx_path)
    compiled_model = ie.compile_model(model=ir_net)
    input_layer_source = compiled_model.input('input.1')
    input_layer_relation = compiled_model.input('input.2')
    output_layer = compiled_model.output(0)
    
    ov_acc = 0.0
    ov_inf_times = []
    for i in range(num_test_samples):
        test_sample = triples_list[i]
        source, relation, target = test_sample
        model_inputs = {input_layer_source: np.int64(source), input_layer_relation: np.int64(relation)}
        start_time = time.time()
        result = compiled_model(model_inputs)[output_layer]
        end_time = time.time()
        ov_inf_times.append(end_time - start_time)
        top_k_idxs = list(np.argpartition(result[0], -TOP_K)[-TOP_K:])
    
        gt = np.array(sorted(t))
        pred = np.array(sorted(top_k_idxs))
        ov_acc += accuracy_score(gt, pred)
    
    avg_ov_time = np.mean(ov_inf_times) * 1000
    print(f'Average time taken for inference: {avg_ov_time} ms')
    print(f'Mean accuracy of the model on the test dataset: {ov_acc/num_test_samples}')


.. parsed-literal::

    Average time taken for inference: 1.466284195582072 ms
    Mean accuracy of the model on the test dataset: 0.10416666666666667


Determine the platform specific speedup obtained through OpenVINO graph optimizations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    print(f'Speedup with OpenVINO optimizations: {round(float(avg_pt_time)/float(avg_ov_time),2)} X')


.. parsed-literal::

    Speedup with OpenVINO optimizations: 0.47 X


Benchmark the converted OpenVINO model using benchmark app
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The OpenVINO toolkit provides a benchmarking application to gauge the
platform specific runtime performance that can be obtained under optimal
configuration parameters for a given model. For more details refer to:
https://docs.openvino.ai/2023.0/openvino_inference_engine_tools_benchmark_tool_README.html

Here, we use the benchmark application to obtain performance estimates
under optimal configuration for the knowledge graph model inference. We
obtain the average (AVG), minimum (MIN) as well as maximum (MAX) latency
as well as the throughput performance (in samples/s) observed while
running the benchmark application. The platform specific optimal
configuration parameters determined by the benchmarking app for OpenVINO
inference can also be obtained by looking at the benchmark app results.

.. code:: ipython3

    print('Benchmark OpenVINO model using the benchmark app')
    ! benchmark_app -m "$fp32_onnx_path" -d CPU -api async -t 10 -shape "input.1[1],input.2[1]" 


.. parsed-literal::

    Benchmark OpenVINO model using the benchmark app
    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2023.0.0-10926-b4452d56304-releases/2023/0
    [ INFO ] 
    [ INFO ] Device info:
    [ INFO ] CPU
    [ INFO ] Build ................................. 2023.0.0-10926-b4452d56304-releases/2023/0
    [ INFO ] 
    [ INFO ] 
    [Step 3/11] Setting device configuration
    [ WARNING ] Performance hint was not explicitly specified in command line. Device(CPU) performance hint will be set to PerformanceMode.THROUGHPUT.
    [Step 4/11] Reading model files
    [ INFO ] Loading model files
    [ INFO ] Read model took 20.85 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     input.1 (node: input.1) : i64 / [...] / []
    [ INFO ]     input.2 (node: input.2) : i64 / [...] / []
    [ INFO ] Model outputs:
    [ INFO ]     51 (node: 51) : f32 / [...] / [1,271]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [ INFO ] Reshaping model: 'input.1': [1], 'input.2': [1]
    [ INFO ] Reshape model took 0.97 ms
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     input.1 (node: input.1) : i64 / [...] / [1]
    [ INFO ]     input.2 (node: input.2) : i64 / [...] / [1]
    [ INFO ] Model outputs:
    [ INFO ]     51 (node: 51) : f32 / [...] / [1,271]
    [Step 7/11] Loading the model to the device
    [ INFO ] Compile model took 57.00 ms
    [Step 8/11] Querying optimal runtime parameters
    [ INFO ] Model:
    [ INFO ]   NETWORK_NAME: torch_jit
    [ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 12
    [ INFO ]   NUM_STREAMS: 12
    [ INFO ]   AFFINITY: Affinity.CORE
    [ INFO ]   INFERENCE_NUM_THREADS: 24
    [ INFO ]   PERF_COUNT: False
    [ INFO ]   INFERENCE_PRECISION_HINT: <Type: 'float32'>
    [ INFO ]   PERFORMANCE_HINT: PerformanceMode.THROUGHPUT
    [ INFO ]   EXECUTION_MODE_HINT: ExecutionMode.PERFORMANCE
    [ INFO ]   PERFORMANCE_HINT_NUM_REQUESTS: 0
    [ INFO ]   ENABLE_CPU_PINNING: True
    [ INFO ]   SCHEDULING_CORE_TYPE: SchedulingCoreType.ANY_CORE
    [ INFO ]   ENABLE_HYPER_THREADING: True
    [ INFO ]   EXECUTION_DEVICES: ['CPU']
    [Step 9/11] Creating infer requests and preparing input tensors
    [ WARNING ] No input files were given for input 'input.1'!. This input will be filled with random values!
    [ WARNING ] No input files were given for input 'input.2'!. This input will be filled with random values!
    [ INFO ] Fill input 'input.1' with random values 
    [ INFO ] Fill input 'input.2' with random values 
    [Step 10/11] Measuring performance (Start inference asynchronously, 12 inference requests, limits: 10000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
    [ INFO ] First inference took 5.00 ms
    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            96480 iterations
    [ INFO ] Duration:         10001.93 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        1.01 ms
    [ INFO ]    Average:       1.04 ms
    [ INFO ]    Min:           0.60 ms
    [ INFO ]    Max:           3.45 ms
    [ INFO ] Throughput:   9646.14 FPS


Conclusions
~~~~~~~~~~~

In this notebook, we convert the trained PyTorch knowledge graph
embeddings model to the OpenVINO format. We confirm that there are no
accuracy differences post conversion. We also perform a sample
evaluation on the knowledge graph. Then, we determine the platform
specific speedup in runtime performance that can be obtained through
OpenVINO graph optimizations. To learn more about the OpenVINO
performance optimizations, refer to:
https://docs.openvino.ai/2023.0/openvino_docs_optimization_guide_dldt_optimization_guide.html

References
~~~~~~~~~~

1. Convolutional 2D Knowledge Graph Embeddings, Tim Dettmers et
   al. (https://arxiv.org/abs/1707.01476)
2. Model implementation: https://github.com/TimDettmers/ConvE

The ConvE model implementation used in this notebook is licensed under
the MIT License. The license is displayed below: MIT License

Copyright (c) 2017 Tim Dettmers

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the
“Software”), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
