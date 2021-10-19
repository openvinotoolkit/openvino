# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import time
from tempfile import gettempdir

# pylint: disable=import-error
from hyperopt import trials_from_docs

from openvino.tools.pot.benchmark.benchmark import benchmark_embedded
from openvino.tools.pot.graph import save_model
from openvino.tools.pot.utils.logger import get_logger
from openvino.tools.pot.utils.object_dump import object_dumps, object_loads
from openvino.tools.pot.utils.utils import create_tmp_dir

try:
    from pymongo import MongoClient
    from pymongo.errors import ServerSelectionTimeoutError
    from bson.objectid import ObjectId
    import gridfs
except ImportError:
    raise ImportError('Pymongo is not installed. Please install it before using multinode configuration.')

REMOTE_DATA_INFO_FREQ_S = 10
RESTORATION_TIME_LIMIT_S = 60 * 60
TRIALS_RESTORATION_TIME_LIMIT_S = 10
UNLOCK_TIME_LIMIT_S = 60

logger = get_logger(__name__)


class Multinode:
    def __init__(self, system_config, config_name):
        self.type = None
        self.server_addr = None
        self.tag = config_name
        self.name = 'no_name'
        self.time_limit = RESTORATION_TIME_LIMIT_S
        self.unlock_time_limit_s = UNLOCK_TIME_LIMIT_S
        self.config = system_config.multinode
        self.client = None
        self.trials = None
        self.model = None
        self.search_space = None
        self.evaluated_params = None
        self.params = None
        self.remote_fs = None
        self.mode = 'peer'
        self.id = None
        self.wait_for_client = True
        self._set_config()

    def _set_config(self):
        if all(key in self.config for key in ['type', 'server_addr']):
            self.type = self.config['type']
            if self.type not in ['server', 'client']:
                raise Exception('Illegal value for type in multinode config!')
            self.server_addr = self.config["server_addr"]
        else:
            raise Exception('Missing "server_addr" or "type" in multinode config')
        if 'mode' in self.config:
            if self.config['mode'] in ['master', 'peer']:
                self.mode = self.config['mode']
        if 'tag' in self.config:
            self.tag = self.config['tag']
        if 'name' in self.config:
            self.name = self.config['name']
        if 'time_limit' in self.config:
            self.time_limit = self.config['time_limit']
        self.client = MongoClient(self.server_addr)
        database = self.client['pot']
        self.trials = database[self.tag + '.trials']
        self.model = database[self.tag + '.model']
        self.search_space = database[self.tag + '.search_space']
        self.evaluated_params = database[self.tag + '.evaluated_params']
        self.params = database[self.tag + '.params']
        self.fp32 = database[self.tag + '.fp32']
        self.clients = database[self.tag + '.clients']
        self._init_gridfs(database)
        self._clear_remote_data(database)

    def _init_gridfs(self, database):
        """ Initialize gridfs for mongodb."""
        try:
            self.client.server_info()
        except ServerSelectionTimeoutError:
            raise Exception('WARNING: Could not connect to MongoDB!!!')
        # GridFS
        self.remote_fs = gridfs.GridFSBucket(database, bucket_name=self.tag)

    def _clear_remote_data(self, database):
        """ Removes remote data from remote database """
        chunks = database[self.tag + '.chunks']
        files = database[self.tag + '.files']
        if self.type == 'server':
            if self.trials.count_documents({}):
                self.trials.drop()
                logger.info('Remote trials found and removed.')
            if self.search_space.count_documents({}):
                self.search_space.drop()
                logger.info('Remote search space found and removed.')
            if self.evaluated_params.count_documents({}):
                self.evaluated_params.drop()
                logger.info('Remote evaluated parameters set found and removed.')
            if self.params.count_documents({}):
                self.params.drop()
                logger.info('Remote config found and removed.')
            # GridFS data cleanup
            if chunks.count_documents({}) or files.count_documents({}):
                chunks.drop()
                files.drop()
                logger.info('Remote trials file found and removed.')
            if self.model.count_documents({}):
                self.model.drop()
                logger.info('Remote model params found and removed.')
            if self.clients.count_documents({}):
                self.clients.drop()
                logger.info('Remote clients found and removed.')
            if self.fp32.count_documents({}):
                self.fp32.drop()
                logger.info('Remote fp32 data found and removed.')

    def update_or_restore_config(self, _config, valid=True):
        """ Update or restore remote loss function config."""
        if self.type == 'client':
            _config = self._restore_remote_config()
            logger.info('Remote config restored')
        else:
            self._update_remote_config(_config, valid)
            logger.info('Remote config updated')
        return _config

    def update_or_restore_fp32(self, _fp32_acc, _fp32_lat):
        """ Update or restore fp32 data (lat only)."""
        if self.mode == 'master':
            if self.type == 'server':
                self._update_remote_fp32(_fp32_acc, _fp32_lat)
            if self.type == 'client':
                _, lat = self._restore_fp32()
                return lat
        return _fp32_lat

    def _update_remote_fp32(self, _fp32_acc, _fp32_lat):
        """ Update remote fp32 function config."""
        if self.type == 'server':
            self.fp32.insert_one({
                'fp32_lat': object_dumps(_fp32_lat),
                'fp32_acc': object_dumps(_fp32_acc)})
            logger.info('Remote fp32 data updated under name: {}'.format(self.tag))

    def _restore_fp32(self):
        """ Restore fp32 function config from remote database."""
        if self.type == 'client':
            time_left = self.time_limit
            while time_left:
                if self.fp32.count_documents({}):
                    fp32_object = self.fp32.find({})
                    _fp32_acc = object_loads(fp32_object[0]['fp32_acc'])
                    _fp32_lat = object_loads(fp32_object[0]['fp32_lat'])
                    logger.info('Remote fp32 data restored')
                    return _fp32_acc, _fp32_lat
                if not time_left % REMOTE_DATA_INFO_FREQ_S:
                    logger.info('Waiting for remote data (fp32): {}s'.format(time_left))
                time.sleep(1)
                time_left -= 1
            raise Exception('WARNING: Time limit for Remote reached!!! config name: {}'.format(
                self.tag))
        return None, None

    def restore_remote_trials(self):
        """ Restore trials from remote database."""
        time_left = TRIALS_RESTORATION_TIME_LIMIT_S
        while time_left:
            if self.trials.count_documents({}):
                trials_object = self.trials.find({})
                trials_object_gfs = self.remote_fs.open_download_stream(trials_object[0]['file_id'])
                hpopt_trials = object_loads(trials_object_gfs.read())
                logger.info('Remote trials restored: {}'.format(len(hpopt_trials.trials)))
                return hpopt_trials
            if not time_left % REMOTE_DATA_INFO_FREQ_S:
                logger.info('Waiting for remote data (trials): {}s'.format(time_left))
            time.sleep(1)
            time_left -= 1
        raise Exception('WARNING: Time limit for Remote reached! config name: {}'.format(
            self.tag))

    def _restore_remote_config(self):
        """ Restore loss function config from remote database."""
        time_left = self.time_limit
        while time_left:
            if self.params.count_documents({}):
                params_object = self.params.find({})
                config_valid = params_object[0]['valid']
                if config_valid:
                    config = params_object[0]['config']
                    logger.info('\nRemote params to be restored:\n\
                        max_trials: {}\n\
                        max_minutes: {}\n\
                        accuracy_loss: {}\n\
                        latency_reduce: {}\n\
                        expected_quantization_ratio: {}\n\
                        accuracy_weight: {}\n\
                        latency_weight: {}\n\
                        quantization_ratio_weight: {}\n\
                        eval_subset_size: {}'.format(config.get('max_trials', None),
                                                     config.get('max_minutes', None),
                                                     config.get('accuracy_loss', 1),
                                                     config.get('latency_reduce', 1),
                                                     config.get('expected_quantization_ratio', 0.5),
                                                     config.get('accuracy_weight', 1.0),
                                                     config.get('latency_weight', 1.0),
                                                     config.get('quantization_ratio_weight', 1.0),
                                                     config.get('eval_subset_size', None)))
                    return config
                if not time_left % REMOTE_DATA_INFO_FREQ_S:
                    logger.info('Found old config. Waiting for config to be updated by server.')
            if not time_left % REMOTE_DATA_INFO_FREQ_S:
                logger.info('Waiting for remote config: {}s'.format(time_left))
            time.sleep(1)
            time_left -= 1
        raise Exception('WARNING: Time limit for Remote reached!!! config name: {}'.format(
            self.tag))

    def restore_remote_search_space(self):
        """ Restore search_space from remote database."""
        time_left = self.time_limit
        while time_left:
            if self.search_space.count_documents({}):
                search_space_object = self.search_space.find({})
                search_space = object_loads(search_space_object[0]['data'])
                logger.info('Remote search_space restored')
                return search_space
            if not time_left % REMOTE_DATA_INFO_FREQ_S:
                logger.info('Waiting for remote data (search_space): {}s'.format(time_left))
            time.sleep(1)
            time_left -= 1
        raise Exception('WARNING: Time limit for Remote reached!!! config name: {}'.format(
            self.tag))

    def _update_remote_config(self, _config=None, valid=True):
        """ Update remote loss function config."""
        if self.params.count_documents({}):
            self.params.update_one({}, {'$set': {'valid': valid}})
        else:
            self.params.insert_one({'config': _config, 'valid': valid})
        logger.info('Remote config updated under name: {}'.format(self.tag))

    def update_remote_search_space(self, _search_space):
        """ Update remote search space."""
        self.search_space.insert_one({'data': object_dumps(_search_space)})
        logger.info('Remote search space updated under name: {}'.format(self.tag))

    def update_remote_trials(self, _hpopt_trials):
        """ Upload local trials to remote database.
            - if some trials already in database, update only last trial and merge it,
            - if not, upload all local trials to remote database (for warm start mode),
        """
        remote_data_updated = False
        retry_time_left = self.unlock_time_limit_s
        while not remote_data_updated and retry_time_left:
            retry_time_left -= 1
            # Update remote data if exist, if not create new remote config
            if self.trials.count_documents({'app': 'tpe'}):
                trials_object_remote_s = self.trials.find({'app': 'tpe'})
                if trials_object_remote_s[0]['Lock']:
                    logger.info('Remote collection locked. Waiting for unlock')
                    if not retry_time_left:
                        raise Exception('WARNING: Retry limit for Trial remote write reached!!!')
                    time.sleep(1)
                else:
                    self.trials.update_one({'app': 'tpe'}, {'$set': {'Lock': 1}})
                    # GridFS get remote trials file
                    current_file = trials_object_remote_s[0]['file_id']
                    if not ObjectId.is_valid(current_file):
                        raise Exception('Remote file corrupted!')
                    trials_object_remote_s_gfs = self.remote_fs.open_download_stream(current_file)
                    trials_object_remote_gfs = object_loads(trials_object_remote_s_gfs.read())
                    # Merge last local trial with remote data
                    _hpopt_trials = trials_from_docs(list(trials_object_remote_gfs) + [list(_hpopt_trials)[-1]])
                    # Upload to database
                    # GridFS upload new trials file
                    new_file = self.remote_fs.upload_from_stream(
                        self.tag,
                        object_dumps(_hpopt_trials), metadata={'trials_count': len(_hpopt_trials.trials)})
                    # Update trials info with new trials file_id
                    self.trials.update_one({'app': 'tpe'}, {'$set': {'Lock': 0, 'file_id': new_file}})
                    # Remove old remote trials file
                    self.remote_fs.delete(current_file)
                    logger.info('Remote trials updated. Total: {} (tag: {})'.format(len(_hpopt_trials.trials),
                                                                                    self.tag))
                    remote_data_updated = True
                    return _hpopt_trials
            else:
                # GridFS write trials file
                new_file = self.remote_fs.upload_from_stream(self.tag, object_dumps(_hpopt_trials),
                                                             metadata={'trials_count': len(_hpopt_trials.trials)})
                # unlock trials to be available for other nodes with current trials id
                self.trials.insert_one({'app': 'tpe', 'Lock': 0, 'file_id': new_file})
                logger.info('No remote trials. First write for config {}'.format(self.tag))
                remote_data_updated = True
                return _hpopt_trials

    def update_remote_evaluated_params(self, _evaluated_params):
        """ Upload local evaluated params to remote database.
            - if some params already in database, update only last params and merge it,
            - if not, upload all local params to remote database (for warm start mode),
        """
        remote_data_updated = False
        retry_time_left = self.unlock_time_limit_s
        while not remote_data_updated and retry_time_left:
            retry_time_left -= 1
            # Update remote data if exist, if not create new remote config
            if self.evaluated_params.count_documents({'app': 'tpe'}):
                params_object_remote_s = self.evaluated_params.find({'app': 'tpe'})
                if params_object_remote_s[0]['Lock']:
                    logger.info('Remote collection locked. Waiting for unlock')
                    if not retry_time_left:
                        raise Exception('WARNING: Retry limit for evaluated parameters remote write reached!!!')
                    time.sleep(1)
                else:
                    self.evaluated_params.update_one({'app': 'tpe'}, {'$set': {'Lock': 1}})
                    remote_params = object_loads(params_object_remote_s[0]['data'])
                    if not isinstance(remote_params, list):
                        raise Exception('Received remote parameters object is not a list!!!')
                    _evaluated_params = list(remote_params) + [list(_evaluated_params)[-1]]
                    self.evaluated_params.update_one(
                        {'app': 'tpe'}, {'$set': {'Lock': 0, 'data': object_dumps(_evaluated_params)}})
                    remote_data_updated = True
                    logger.info('Remote evaluated parameters set updated under name: {}'.format(self.tag))
                    return _evaluated_params
            else:
                self.evaluated_params.insert_one({'app': 'tpe', 'Lock': 0, 'data': object_dumps(_evaluated_params)})
                remote_data_updated = True
                logger.info('Remote evaluated parameters set updated under name: {}'.format(self.tag))
                return _evaluated_params

    def restore_remote_evaluated_params(self):
        """ Restore evaluated_params from remote database."""
        time_left = self.time_limit
        while time_left:
            if self.evaluated_params.count_documents({}):
                evaluated_params_object = self.evaluated_params.find({})
                evaluated_params = object_loads(evaluated_params_object[0]['data'])
                logger.info('Remote evaluated_params restored')
                return evaluated_params
            if not time_left % REMOTE_DATA_INFO_FREQ_S:
                logger.info('Waiting for remote data (evaluated_params): {}s'.format(time_left))
            time.sleep(1)
            time_left -= 1
        raise Exception('WARNING: Time limit for Remote reached!!! config name: {}'.format(
            self.tag))

    def request_remote_benchmark(self, model, iteration):
        """ For Clients only.
            Upload request with params for server to create model and run benchmark.
            Wait for response.
        """
        if self.type == 'client' and self.mode == 'master':
            self._set_client_status(active=True)
            model_file_id, weights_file_id = self.upload_model(model, iteration)
            remote_id = self.model.insert_one({
                'name': self.name,
                'iter': iteration,
                'lat': 0,
                'file_id': model_file_id,
                'bin_file_id': weights_file_id})
            logger.info('Model queued for remote evaluation id: {}'.format(remote_id.inserted_id))
            time_left = 0
            remote_lat = 0
            while not remote_lat and time_left < self.time_limit:
                remote_params_obj = self.model.find({'_id':remote_id.inserted_id})[0]
                remote_lat = remote_params_obj['lat']
                time_left += 1
                if not time_left % REMOTE_DATA_INFO_FREQ_S:
                    logger.info('Waiting for remote benchmark: {}s'.format(time_left))
                time.sleep(1)
            if remote_lat:
                logger.info('remote_lat_client: {}'.format(remote_lat))
                self.model.delete_one({'_id' : remote_id.inserted_id})
                self.remote_fs.delete(model_file_id)
                self.remote_fs.delete(weights_file_id)
                return remote_lat
        return None

    def upload_model(self, model, trial_no):
        tmp_dir, path_to_model, path_to_weights = self._create_temp_dir()
        save_model(model, tmp_dir, 'tmp_model')
        with open(path_to_model, 'rb') as file:
            model_file_id = self.remote_fs.upload_from_stream(
                self.tag,
                file, metadata={'iter': trial_no, 'type': 'model'})
        with open(path_to_weights, 'rb') as file:
            weights_file_id = self.remote_fs.upload_from_stream(
                self.tag,
                file, metadata={'iter': trial_no, 'type': 'weights'})
        return model_file_id, weights_file_id

    def calculate_remote_requests(self):
        """ For Server only.
            Check in loop for requests from clients to run benchmark.
            Finish when all clients clear activation flag.
        """
        if self.type == 'server' and self.mode == 'master':
            waiting_time = 1
            while self._check_clients_status(not waiting_time % REMOTE_DATA_INFO_FREQ_S):
                waiting_time += 1
                model_count = self.model.count_documents({'lat' : {'$eq': 0}})
                if  not waiting_time % REMOTE_DATA_INFO_FREQ_S:
                    logger.info('Waiting for requests: {}'.format(waiting_time))
                    if model_count:
                        logger.info('Models in queue: {}'.format(model_count))
                if model_count:
                    remote_params_obj = self.model.find({'lat' : {'$eq': 0}})[0]
                    remote_lat = remote_params_obj['lat']
                    remote_iter = remote_params_obj['iter']
                    remote_name = remote_params_obj['name']
                    remote_file_id = remote_params_obj['file_id']
                    remote_file_id_bin = remote_params_obj['bin_file_id']
                    logger.info('Starting for: {} iter: {}'.format(remote_name, remote_iter))
                    if remote_lat == 0:
                        lat = self._run_remote_benchmark(remote_file_id, remote_file_id_bin)
                        logger.info('name: {} remote_lat_res: {} for iter: {}'.format(remote_name, lat, remote_iter))
                        self.model.update_one({'_id': remote_params_obj['_id']}, {'$set': {'lat': lat}})
                time.sleep(1)

    def _create_temp_dir(self):
        __MODEL_PATH__ = create_tmp_dir(gettempdir())
        model_name = 'tmp_model'
        path_to_model = __MODEL_PATH__.name + '/' + model_name + '.xml'
        path_to_weights = __MODEL_PATH__.name + '/' + model_name + '.bin'
        return __MODEL_PATH__.name, path_to_model, path_to_weights

    def _run_remote_benchmark(self, file_id, remote_file_id_bin):
        __MODEL_PATH__ = create_tmp_dir(gettempdir())
        model_name = 'tmp_model'
        path_to_model = __MODEL_PATH__.name + '/' + model_name + '.xml'
        path_to_weights = __MODEL_PATH__.name + '/' + model_name + '.bin'
        new_file = self.remote_fs.open_download_stream(file_id)
        new_file_bin = self.remote_fs.open_download_stream(remote_file_id_bin)
        with open(path_to_model, 'wb') as file:
            file.write(new_file.read())
        with open(path_to_weights, 'wb') as file:
            file.write(new_file_bin.read())
        lat = benchmark_embedded(mf=path_to_model)
        return lat

    def _check_clients_status(self, log=False):
        """ Check if are any clients in 'active' state."""
        clients_number = self.clients.count_documents({})
        clients_active = self.clients.count_documents({'active' : {'$eq': 1}})
        if log:
            logger.info('Total/Active clients:{}/{}'.format(clients_number, clients_active))
        if clients_active:
            self.wait_for_client = False
        return 1 if self.wait_for_client else clients_active

    def _set_client_status(self, active=False):
        """ Set or clear 'active' flag for clients."""
        if self.type == 'client':
            if self.id is None:
                self.id = self.clients.insert_one({'active': 1 if active else 0})
                logger.info('Setting client status to True')
            else:
                self.clients.update_one({'_id': self.id.inserted_id}, {'$set': {'active': 1 if active else 0}})
                if not active:
                    logger.info('Setting client status to False')

    def cleanup(self):
        """ Ending cleanup """
        if self.type == 'client':
            self._set_client_status(active=False)
        if self.type == 'server':
            self._update_remote_config(valid=False)
