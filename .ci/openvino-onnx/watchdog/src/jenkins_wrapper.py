#!/usr/bin/python3

# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import requests
import jenkins
import logging
from retrying import retry

# Logging
logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

_RETRY_LIMIT = 3
_RETRY_COOLDOWN_MS = 5000


class JenkinsWrapper:
    """Class wrapping Python-Jenkins API.

    The purpose of this class is to wrap methods from Python-Jenkins API used in Watchdog, for less error-prone and
    more convenient use. Docs for used API, including wrapped methods can be found at:
    https://python-jenkins.readthedocs.io/en/latest/

        :param jenkins_token:       Token used for Jenkins
        :param jenkins_user:        Username used to connect to Jenkins
        :param jenkins_server:      Jenkins server address
        :type jenkins_token:        String
        :type jenkins_user:         String
        :type jenkins_server:       String
    """

    def __init__(self, jenkins_token, jenkins_user, jenkins_server):
        self.jenkins_server = jenkins_server
        self.jenkins = jenkins.Jenkins(jenkins_server, username=jenkins_user,
                                       password=jenkins_token)

    @retry(stop_max_attempt_number=_RETRY_LIMIT, wait_fixed=_RETRY_COOLDOWN_MS)
    def get_build_console_output(self, job_name, build_number):
        return self.jenkins.get_build_console_output(job_name, build_number)

    @retry(stop_max_attempt_number=_RETRY_LIMIT, wait_fixed=_RETRY_COOLDOWN_MS)
    def get_job_info(self, job_name):
        return self.jenkins.get_job_info(job_name)

    @retry(stop_max_attempt_number=_RETRY_LIMIT, wait_fixed=_RETRY_COOLDOWN_MS)
    def get_build_info(self, job_name, build_number):
        return self.jenkins.get_build_info(job_name, build_number)

    @retry(stop_max_attempt_number=_RETRY_LIMIT, wait_fixed=_RETRY_COOLDOWN_MS)
    def get_queue_item(self, queue_id):
        """Attempt to retrieve Jenkins job queue item.

        Exception communicating queue doesn't exist is expected,
        in that case method returns empty dict.

            :param queue_id:            Jenkins job queue ID number
            :type queue_id:             int
            :return:                    Dictionary representing Jenkins job queue item
            :rtype:                     dict
        """
        try:
            return self.jenkins.get_queue_item(queue_id)
        except Exception as e:
            # Exception 'queue does not exist' is expected behaviour when job is running
            if 'queue' in str(e) and 'does not exist' in str(e):
                return {}
            else:
                raise

    @retry(stop_max_attempt_number=_RETRY_LIMIT, wait_fixed=_RETRY_COOLDOWN_MS)
    def get_idle_ci_hosts(self):
        """Query Jenkins for idle servers.

        Send GET request to Jenkins server, querying for idle servers labeled
        for OpenVino-ONNX CI job.

            :return:     Number of idle hosts delegated to OpenVino-ONNX CI
            :rtype:      int
        """
        jenkins_request_url = self.jenkins_server + 'label/ci&&onnx/api/json?pretty=true'
        try:
            log.info('Sending request to Jenkins: %s', jenkins_request_url)
            r = requests.Request(method='GET', url=jenkins_request_url, verify=False)
            response = self.jenkins.jenkins_request(r).json()
            return int(response['totalExecutors']) - int(response['busyExecutors'])
        except Exception as e:
            log.exception('Failed to send request to Jenkins!\nException message: %s', str(e))
            raise
