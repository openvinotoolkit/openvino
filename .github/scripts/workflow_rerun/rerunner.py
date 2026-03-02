# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import tempfile
import psycopg2
from pathlib import Path
from psycopg2 import sql

import requests
from github import Github, Auth
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from workflow_rerun.argument_parser import get_arguments
from workflow_rerun.constants import GITHUB_TOKEN, LOGGER
from workflow_rerun.log_analyzer import LogAnalyzer
from workflow_rerun.log_collector import collect_logs_for_run

def record_rerun_to_db(repository_full_name: str, run_id: int, ticket_number: int, rerunner_run_id: int, error_text: str):
    """Record the rerun event to the PostgreSQL database."""
    db_username = os.environ.get('PGUSER')
    db_password = os.environ.get('PGPASSWORD')
    db_host = os.environ.get('PGHOST')
    db_database = os.environ.get('PGDATABASE')
    db_port = os.environ.get('PGPORT')
    conn = psycopg2.connect(host=db_host,
                            port=db_port,
                            user=db_username,
                            password=db_password,
                            database=db_database)

    cursor = None
    try:
        cursor = conn.cursor()

        insert_query = sql.SQL("""
            INSERT INTO rerunner_stats (repository_full_name, run_id, ticket_number, rerun_at, rerunner_run_id, error_text)
            VALUES (%s, %s, %s, NOW() AT TIME ZONE 'UTC', %s, %s)
        """)

        cursor.execute(insert_query, (repository_full_name, run_id, ticket_number, rerunner_run_id, error_text))
        conn.commit()

        LOGGER.info(f'Successfully recorded rerun to database: repo={repository_full_name}, '
                    f'run_id={run_id}, ticket={ticket_number}, rerunner_run_id={rerunner_run_id}, error_text={error_text}')

    except psycopg2.Error as e:
        LOGGER.error(f'Failed to record rerun to database: {e}')
        conn.rollback()
        raise
    finally:
        if cursor:
            cursor.close()
        conn.close()

def rerun_failed_jobs(repository_name: str, run_id: int, session: requests.Session):
    # PyGitHub does not expose the "/repos/{owner}/{repo}/actions/runs/RUN_ID/rerun-failed-jobs" endpoint
    # so we have to use requests
    response = session.post(
        url=f'https://api.github.com/repos/{repository_name}/actions/runs/{run_id}/rerun-failed-jobs',
        headers={'Authorization': f'Bearer {GITHUB_TOKEN}'}
    )

    response.raise_for_status()

    LOGGER.info(f'RUN RETRIGGERED SUCCESSFULLY: {run.html_url}')

def analyze_and_rerun(run, repository_name: str, run_id: int, rerunner_run_id: int,
                      errors_file: Path, is_dry_run: bool, session: requests.Session):
    with tempfile.TemporaryDirectory() as temp_dir:
        logs_dir = Path(temp_dir)
        collect_logs_for_run(
            run=run,
            logs_dir=logs_dir,
            session=session
        )

        log_analyzer = LogAnalyzer(
            path_to_logs=logs_dir,
            path_to_errors_file=errors_file
        )
        log_analyzer.analyze()

        if log_analyzer.found_matching_error:
            LOGGER.info(f'FOUND MATCHING ERROR, RETRIGGERING {run.html_url}')
            if is_dry_run:
                LOGGER.info(f'RUNNING IN DRY RUN MODE, NOT RETRIGGERING, EXITING')
                return

            rerun_failed_jobs(repository_name, run_id, session)

            if log_analyzer.found_error_ticket and log_analyzer.matched_error_text:
                record_rerun_to_db(repository_name, run_id,
                                   log_analyzer.found_error_ticket,
                                   rerunner_run_id,
                                   log_analyzer.matched_error_text)
            else:
                LOGGER.error(f'Cannot record to database: missing ticket_number or error_text')
                raise ValueError('Missing ticket_number or error_text for database recording.')
        else:
            LOGGER.info(f'NO ERROR WAS FOUND, NOT RETRIGGERING')

if __name__ == '__main__':
    args = get_arguments()
    run_id = args.run_id
    rerunner_run_id = args.rerunner_run_id
    repository_name = args.repository_name
    errors_file = args.errors_to_look_for_file
    is_dry_run = args.dry_run
    if is_dry_run:
        LOGGER.info('RUNNING IN DRY RUN MODE. IF ERROR WILL BE FOUND, WILL NOT RETRIGGER')

    session = requests.Session()
    retry_strategy = Retry(
        total=5,
        backoff_factor=3,
        backoff_jitter=1,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    session.mount("https://github.com", HTTPAdapter(max_retries=retry_strategy))

    github = Github(auth=Auth.Token(token=GITHUB_TOKEN))
    gh_repo = github.get_repo(full_name_or_id=repository_name)
    run = gh_repo.get_workflow_run(id_=run_id)

    LOGGER.info(f'CHECKING IF RERUN IS NEEDED FOR {run.html_url} RUN IN {repository_name}.')

    # Check if the run has already been retriggered
    # we do not want to fall into a loop with retriggers
    if run.run_attempt > 1:
        LOGGER.info(f'THERE ARE {run.run_attempt} ATTEMPTS ALREADY. NOT CHECKING LOGS AND NOT RETRIGGERING. EXITING')
        sys.exit(0)

    analyze_and_rerun(run, repository_name, run_id, rerunner_run_id, errors_file, is_dry_run, session)
