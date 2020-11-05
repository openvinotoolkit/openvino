#!/usr/bin/python3

# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
from watchdog import Watchdog

DEFAULT_MSTEAMS_URL_FILE = '/home/lab_nerval/tokens/msteams_url'
DEFAULT_GITHUB_ORGANIZATION = 'openvinotoolkit'
DEFAULT_GITHUB_PROJECT = 'openvino'
DEFAULT_JENKINS_TOKEN_FILE = '/home/lab_nerval/tokens/crackerjack'
DEFAULT_JENKINS_SERVER = 'https://crackerjack.intel.com/'
DEFAULT_JENKINS_USER = 'lab_nerval'
DEFAULT_CI_JOB_NAME = 'onnx/OpenVino_CI'
DEFAULT_WATCHDOG_JOB_NAME = 'onnx/ci_watchdog'


def main(args):
    """
    Read args passed to script, load tokens and run watchdog.

    Keyword arguments:
    :param args:    arguments parsed by argparse ArgumentParser

    :return:        returns status code 0 on successful completion

    """
    jenkins_server = args.jenkins_server.strip()
    jenkins_user = args.jenkins_user.strip()
    jenkins_token = open(args.jenkins_token).read().replace('\n', '').strip()
    msteams_url = open(args.msteams_url).read().replace('\n', '').strip()
    github_credentials = args.github_credentials
    github_org = args.github_org
    github_project = args.github_project
    ci_job = args.ci_job.strip()
    watchdog_job = args.watchdog_job.strip()
    quiet = args.quiet

    wd = Watchdog(jenkins_token=jenkins_token,
                  jenkins_server=jenkins_server,
                  jenkins_user=jenkins_user,
                  github_credentials=github_credentials,
                  git_org=github_org,
                  git_project=github_project,
                  msteams_url=msteams_url,
                  ci_job_name=ci_job,
                  watchdog_job_name=watchdog_job)
    wd.run(quiet=quiet)

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--msteams-url', help='Path to MS Teams channel url to communicate messages.',
                        default=DEFAULT_MSTEAMS_URL_FILE, action='store', required=False)

    parser.add_argument('--github-credentials', help='GitHub user credentials to access repo.',
                        nargs="+", required=True)

    parser.add_argument('--github-org', help='Name of organization on GitHub.',
                        default=DEFAULT_GITHUB_ORGANIZATION, action='store', required=False)

    parser.add_argument('--github-project', help='Name of project on GitHub.',
                        default=DEFAULT_GITHUB_PROJECT, action='store', required=False)

    parser.add_argument('--jenkins-token', help='Path to Jenkins user token to access build info.',
                        default=DEFAULT_JENKINS_TOKEN_FILE, action='store', required=False)

    parser.add_argument('--jenkins-server', help='Jenkins server address.',
                        default=DEFAULT_JENKINS_SERVER, action='store', required=False)

    parser.add_argument('--jenkins-user', help='Jenkins user used to log in.',
                        default=DEFAULT_JENKINS_USER, action='store', required=False)

    parser.add_argument('--ci-job', help='Jenkins CI job name.',
                        default=DEFAULT_CI_JOB_NAME, action='store', required=False)

    parser.add_argument('--watchdog-job', help='Jenkins CI Watchdog job name.',
                        default=DEFAULT_WATCHDOG_JOB_NAME, action='store', required=False)

    parser.add_argument('--quiet', help="Quiet mode - doesn\'t send message to communicator.",
                        action='store_true', required=False)

    args = parser.parse_args()
    sys.exit(main(args))
