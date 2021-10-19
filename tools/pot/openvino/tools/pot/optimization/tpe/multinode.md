# TPE Multiple Node Configuration Based on MongoDB Database. {#pot_compression_optimization_tpe_multinode}

The multi-node configuration purpose is to reduce the execution time of TPE algorithm.
The main execution model of multi-node configuration is to run the identical script
on many machines and share trials, loss function configuration,
evaluated parameters and search space objects between them.

Multi-node configuration can work with two modes: master and peer.
In peer mode, all machines calculate latency for evaluated models themselves.
It means that for this configuration nodes used need to be homogeneous.
In master configuration, only one machine called server calculates latency
for evaluated models. Other machines are called clients and are responsible for generating the model based on parameters
chosen by TPE, evaluating its accuracy and sending it to the server node
for latency measurement. This approach enables precise latency measurements in the environment with
machines with different hardware as long as server machine correctly represent target hardware.

The final result can be taken from any of the nodes.

## Configuration of nodes
In this configuration, we distinguish two types of nodes. One is server
which is responsible for proper "search space" and "loss function config" creation
and client nodes which read data produced by server and use it for further
best result search and evaluation.
Server and Client .json config file changes:
```json
"optimizer": {
    "name": "Tpe",
    "params": {
    "multinode": {
        "name": "node_name", ← optional
        "type": "server", ← for server node
        "type": "client", ← for client node
        "server_addr": "<server_ip_addr&gt;:<server_port_number&gt;",
        "tag": "group_name", ← optional
        "mode": "peer"← optional
    },
    "max_trials": 10,
    "trials_load_method": "cold_start",
    ...,
    }
}
```
`parameters:`
* `"name"`: Name saved in trials.csv file, mainly for debug purpose,
* `"type"`: Can be "server" or "client"
* `"server_addr"`: <server_ip_addr&gt;: IP address of the machine where MongoDB database
is configured. It can be different from any Node IP used for TPE execution.
<server_port_number&gt;: Port number of MongoDB database, by default it's 27017
* `"tag"`: Name for a group of systems working together. Without this tag,
systems will be grouped by the model they are working on.
If more than one group of systems is working on the same model using
the same MongoDB database, their results will collide.
* `"mode"`: "peer" or "master" mode selection.

## How to run TPE in multi node configuration
For every node, you need to have an environment prepared in the same way as for the regular run.
Models and Datasets should be prepared in configuration files.
When you add "multinode" parameter to the configuration file and MongoDB is active and running
you need to run a tool on every node you want to be part of the searching group of machines.

The server should be run first because it needs to prepare data for other nodes.
When the server node starts its 2nd Trial clients will start their search.

Steps needed to run multi-node configuration:

1. Add to your base configuration file 'multinode' parameters with one server and client
type for rest of nodes,
2. Create 'pot' database in MongoDB instance,
3. Run server node as first (*server will perform cleanup on database*),
4. Run client nodes,

## How to select mode
When a mode parameter is set to peer all nodes (clients and server) will search for the best
result and evaluate models (for accuracy and latency). In this mode nodes should be homogeneous,
so that result of the benchmark on the same model would be similar on every node.
This allows calculating correct loss and achieves better and faster convergence.

The Master mode is more reliable for latency calculation. Only the server node calculates
latency for the rest of the nodes, but it is not doing accuracy evaluation
and does not take part in searching for the best result. That's why this mode is slightly
slower than the previous one.

Select master mode when:
* machines with different hardware configuration are used (memory, CPU),
* layer option is set in configuration file, (latency sensitive),
* latency is main factor to be improved,
* for number of nodes 4+.

Select peer mode when:
* machines used are homogeneous,
* range estimator option is set in configuration file (accuracy sensitive),
* accuracy is main factor to be improved,
* for limited number of nodes 1-3.

## How it works
All synchronization is done by the MongoDB database. There is no direct communication between servers and clients. 
The client needs to wait for information about loss function configuration, fp32 metrics, or search space until
the server push this data to the database.

## Results
Time in minutes for TPE execution for 200 trials on ssd-mobilenetv1 and COCO dataset.

No. of nodes | master mode | peer mode
----------------- | ------ | -----
1| n/a | 865
2| n/a | 488
3| 510 | 329
4| 325 | 250
5| 260 | 200
6| 208 | 171
7| 188 | 153