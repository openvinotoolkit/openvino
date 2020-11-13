#!/usr/bin/python3

# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import timeout_decorator
from datetime import datetime
from retrying import retry
from github import Github, GithubException

# Logging
logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

_RETRY_LIMIT = 3
_RETRY_COOLDOWN_MS = 2000
_REQUEST_TIMEOUT_S = 10


class GitWrapper:
    """Class wrapping PyGithub API.

    The purpose of this class is to wrap methods from PyGithub API used in Watchdog, for less error-prone and
    more convenient use. Docs for used API, including wrapped methods can be found at:
    https://pygithub.readthedocs.io/en/latest/introduction.html

    :param github_credentials:       Credentials used for GitHub
    :param repository:            GitHub repository name
    :param project:               GitHub project name
    :type github_credentials:        String
    :type repository:             String
    :type project:                String
    """

    def __init__(self, github_credentials, repository, project):
        self.git = Github(*github_credentials)
        self.repository = repository
        self.project = project
        self.github_credentials = github_credentials

    @retry(stop_max_attempt_number=_RETRY_LIMIT, wait_fixed=_RETRY_COOLDOWN_MS)
    def get_git_time(self):
        """Retrieve time from GitHub.

        Used to reliably determine time during Watchdog run.

        :return:                    Datetime object describing current time
        :rtype:                     datetime
        """
        try:
            datetime_object = self._get_git_time()
        except ValueError as e:
            raise GitWrapperError(str(e))
        except GithubException as e:
            message = 'GitHub Exception during API status retrieval. Exception: {}'.format(str(e))
            raise GitWrapperError(message)
        except timeout_decorator.TimeoutError:
            message = 'GitHub Exception during API status retrieval. Timeout during API request.'
            raise GitWrapperError(message)
        return datetime_object

    @retry(stop_max_attempt_number=_RETRY_LIMIT, wait_fixed=_RETRY_COOLDOWN_MS)
    def get_pull_requests(self):
        """Retrieve paginated list of pull requests from GitHub.

        :return:                    Paginated list of Pull Requests in GitHub repo
        :rtype:                     github.PaginatedList.PaginatedList of github.PullRequest.PullRequest
        """
        try:
            prs = self._get_pull_requests()
        except GithubException as e:
            message = 'GitHub Exception during API status retrieval. Exception: {}'.format(str(e))
            raise GitWrapperError(message)
        return prs

    @timeout_decorator.timeout(_REQUEST_TIMEOUT_S)
    def _get_git_time(self):
        """Private method retrieving time from GitHub.

        :return:                    Datetime object describing current time
        :rtype:                     datetime
        """
        datetime_string = self.git.get_api_status().raw_headers.get('date', '')
        datetime_format = '%a, %d %b %Y %H:%M:%S %Z'
        datetime_object = datetime.strptime(datetime_string, datetime_format)
        return datetime_object

    @timeout_decorator.timeout(_REQUEST_TIMEOUT_S)
    def _get_pull_requests(self):
        """Private method retrieving pull requests from GitHub.

        :return:                    Paginated list of Pull Requests in GitHub repo
        :rtype:                     github.PaginatedList.PaginatedList of github.PullRequest.PullRequest
        """
        return self.git.get_organization(self.repository).get_repo(self.project).get_pulls()


class GitWrapperError(Exception):
    """Base class for exceptions raised in GitWrapper.

    :param message                   Explanation of the error
    """

    def __init__(self, message):
        self.message = message
        log.exception(message)
