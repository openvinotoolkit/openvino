# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3

from github import Github
from psycopg2 import sql
import os
import logging
import psycopg2
import dateutil
import argparse

def init_logger():
    LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
    logging.basicConfig(level=LOGLEVEL,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d-%Y %H:%M:%S')

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--repository-name', type=str, required=True,
                        help='Repository name in OWNER/REPOSITORY format')
    parser.add_argument('--run-id', type=str, required=True,
                        help='Workflow Run ID')

    return parser

def create_db_tables(conn, cur):
    cur.execute('''CREATE TABLE IF NOT EXISTS workflow_runs(
    id SERIAL PRIMARY KEY,
    run_id BIGINT,
    html_url TEXT,
    name VARCHAR(255),
    run_started_at TIMESTAMP,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    triggering_actor_login VARCHAR(255),
    conclusion VARCHAR(25),
    event VARCHAR(50),
    run_attempt INT,
    repository_full_name VARCHAR(255),
    head_repository_full_name VARCHAR(255),
    head_branch VARCHAR(255),
    status VARCHAR(25),
    display_title TEXT,
    path TEXT,
    total_duration_seconds INT
    );
    ''')
    cur.execute('''CREATE TABLE IF NOT EXISTS workflow_jobs(
    id SERIAL PRIMARY KEY,
    job_id BIGINT,
    parent_run_id BIGINT,
    html_url TEXT,
    name VARCHAR(255),
    created_at TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    queued_duration_seconds INT,
    duration_seconds INT,
    runner_name VARCHAR(255),
    status VARCHAR(25),
    conclusion VARCHAR(25),
    head_branch VARCHAR(255),
    run_attempt INT,
    workflow_name TEXT
    );
    ''')
    cur.execute('''CREATE TABLE IF NOT EXISTS workflow_steps(
    id SERIAL PRIMARY KEY,
    parent_job_id BIGINT,
    name VARCHAR(255),
    conclusion VARCHAR(25),
    number INT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    duration_seconds INT
    );
    ''')
    conn.commit()

def main():
    init_logger()
    parser = make_parser()
    args = parser.parse_args()
    logger = logging.getLogger(__name__)

    github_token = os.environ.get('GITHUB_TOKEN')
    if not github_token:
        raise ValueError('GITHUB_TOKEN environment variable is not set!')

    run_id = args.run_id
    repo_name = args.repository_name


    # this should be specified in runner's env
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

    # Create tables
    cur = conn.cursor()
    create_db_tables(conn, cur)

    # Get the data
    g = Github(github_token)
    repo = g.get_repo(repo_name)

    run = repo.get_workflow_run(int(run_id))
    logger.info('Processing run ID %s - %s, URL: %s', run_id, run.name, run.html_url)
    if run.status != 'completed':
        logger.error('Run %s is not completed! Only completed runs should be in the database', run_id)
        raise SystemExit(1)

    # We rely on the following assumptions:
    # - The workflow run is completed. When run.status != 'completed' we should not add it to the database
    #   theoretically the second attempt can be triggerred right after the completion of the first one
    #   or while the runner which executes this script is deploying
    #
    # - Job's queued duration equals "job.started_at - job.created_at" if started_at > created_at.
    #   Otherwise the job should not be added to the database
    total_duration_seconds = round(run.timing().run_duration_ms / 1000)

    workflow_data_query = sql.SQL('''INSERT INTO workflow_runs(
    run_id, html_url, name,
    run_started_at, created_at, updated_at, triggering_actor_login, conclusion,
    event, run_attempt, repository_full_name, head_repository_full_name,
    head_branch, status, display_title, path, total_duration_seconds)
    VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
    ''')

    workflow_query_args = (run_id, run.html_url, run.name,
                           run.run_started_at, run.created_at, run.updated_at,
                           run.raw_data['triggering_actor']['login'],
                           run.conclusion, run.event,
                           run.run_attempt, run.raw_data['repository']['full_name'],
                           run.raw_data['head_repository']['full_name'],
                           run.head_branch, run.status,
                           run.display_title, run.path, total_duration_seconds)

    logger.debug('Workflow run query: %s', cur.mogrify(workflow_data_query, workflow_query_args))
    cur.execute(workflow_data_query, workflow_query_args)

    for job in run.jobs():
        job_id = job.id
        logger.info('Processing job %s, URL: %s', job.name, job.html_url)
        queued_duration_seconds = 0
        duration_seconds = 0

        job_created_at_date = dateutil.parser.parse(job.raw_data['created_at'])
        if job_created_at_date > job.started_at:
            logger.warning('Skipping job %s of run %s - most likely a stub \
            job created after workflow restart', job.name, run_id)
            continue

        queued_duration_timedelta = job.started_at - job_created_at_date
        queued_duration_seconds = round(queued_duration_timedelta.total_seconds())

        duration_timedelta = job.completed_at - job.started_at
        duration_seconds = round(duration_timedelta.total_seconds())

        job_data_query = sql.SQL('''
        INSERT INTO workflow_jobs(
        job_id, parent_run_id, html_url, name,
        created_at, started_at, completed_at,
        queued_duration_seconds, duration_seconds,
        runner_name, status, conclusion, head_branch,
        run_attempt, workflow_name
        )
        VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        ''')

        job_query_args = (job_id, run_id, job.html_url, job.name,
                          job.raw_data['created_at'], job.started_at, job.completed_at,
                          queued_duration_seconds, duration_seconds,
                          job.raw_data['runner_name'], job.status, job.conclusion,
                          job.raw_data['head_branch'], job.raw_data['run_attempt'], job.raw_data['workflow_name'])

        logger.debug('Job query: %s', cur.mogrify(job_data_query, job_query_args))
        cur.execute(job_data_query, job_query_args)
        conn.commit()
        for step in job.steps:
            logger.info('Processing step %s', step.name)

            # It seems like a GitHub bug, but
            # sometimes there're steps "In progress"
            # despite the workflow being completed
            step_duration_seconds = 0

            if step.status != 'completed':
                logger.warning('Step %s isn\'t  completed. Setting duration to zero.', step.name)
                step_duration_seconds = 0
            else:
                step_duration_seconds_timedelta = step.completed_at - step.started_at
                step_duration_seconds = round(step_duration_seconds_timedelta.total_seconds())

            step_data_query = sql.SQL('''
            INSERT INTO workflow_steps(
            parent_job_id, name, conclusion,
            number, started_at, completed_at,
            duration_seconds)
            VALUES (%s, %s, %s, %s, %s, %s, %s);
            ''')
            step_query_args = (
                job_id, step.name, step.conclusion,
                step.number, step.started_at, step.completed_at,
                step_duration_seconds)

            logger.debug('Step query: %s', cur.mogrify(step_data_query, step_query_args))
            cur.execute(step_data_query, step_query_args)

    conn.commit()
    cur.close()
    conn.close()
    g.close()
if __name__ == "__main__":
    main()
