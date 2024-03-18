#!/usr/bin/env python3

from github import Github
from psycopg2 import sql
import os
import logging
import psycopg2
import dateutil

def init_logger():
    LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
    logging.basicConfig(level=LOGLEVEL,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d-%Y %H:%M:%S')

def create_db_tables(conn, cur):
    cur.execute('''CREATE TABLE IF NOT EXISTS github_workflow_runs_test(
    id SERIAL,
    run_id BIGINT PRIMARY KEY,
    html_url TEXT,
    name VARCHAR(255),
    run_started_at TIMESTAMP,
    triggering_actor_login VARCHAR(255),
    conclusion VARCHAR(25),
    run_number INT,
    event VARCHAR(50),
    run_attempt INT,
    repository_full_name VARCHAR(255),
    head_repository_full_name VARCHAR(255),
    head_branch VARCHAR(255),
    status VARCHAR(25),
    display_title TEXT,
    path TEXT
    );
    ''')
    cur.execute('''CREATE TABLE IF NOT EXISTS github_workflow_jobs_test(
    id SERIAL,
    job_id BIGINT PRIMARY KEY,
    parent_run_id BIGINT REFERENCES github_workflow_runs_test(run_id),
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
    head_branch VARCHAR(255)
    );
    ''')
    cur.execute('''CREATE TABLE IF NOT EXISTS github_workflow_steps_test(
    id SERIAL PRIMARY KEY,
    parent_job_id BIGINT REFERENCES github_workflow_jobs_test(job_id),
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

    logger = logging.getLogger(__name__)

    github_token = os.environ.get('GITHUB_TOKEN')
    if not github_token:
        raise ValueError('GITHUB_TOKEN environment variable is not set!')

    run_id = os.environ.get('RUN_ID')
    if not run_id:
        raise ValueError('RUN_ID environment variable is not set!')

    repo_name = os.environ.get('GITHUB_REPOSITORY')
    if not repo_name:
        raise ValueError('GITHUB_REPOSITORY environment variable is not set!')


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

    workflow_data_query = f'''INSERT INTO github_workflow_runs_test(
    run_id, html_url, name,
    run_started_at, triggering_actor_login, conclusion,
    run_number, event, run_attempt, repository_full_name,
    head_branch, display_title, path)
    VALUES(
    '{run_id}', '{run.html_url}', '{run.name}', '{run.run_started_at}',
    '{run.raw_data['triggering_actor']['login']}',
    '{run.conclusion}', '{run.run_number}', '{run.event}',
    '{run.run_attempt}', '{run.raw_data['repository']['full_name']}',
    '{run.head_branch}', '{run.display_title}', '{run.path}'
    );
    '''

    logger.debug('Workflow run query: %s', workflow_data_query)
    cur.execute(workflow_data_query)

    for job in run.jobs():
        job_id = job.id
        queued_duration_seconds = 0
        duration_seconds = 0

        job_created_at_date = dateutil.parser.parse(job.raw_data['created_at'])

        queued_duration_timedelta = job.started_at - job_created_at_date
        queued_duration_seconds = round(queued_duration_timedelta.total_seconds())

        duration_timedelta = job.completed_at - job.started_at
        duration_seconds = round(duration_timedelta.total_seconds())

        job_data_query = f'''
        INSERT INTO github_workflow_jobs_test(
        job_id, parent_run_id, html_url, name,
        created_at, started_at, completed_at,
        queued_duration_seconds, duration_seconds,
        runner_name, status, conclusion, head_branch)
        VALUES(
        '{job_id}', '{run_id}', '{job.html_url}', '{job.name}',
        '{job.raw_data['created_at']}', '{job.started_at}', '{job.completed_at}',
        '{queued_duration_seconds}', '{duration_seconds}',
        '{job.raw_data['runner_name']}', '{job.status}', '{job.conclusion}',
        '{job.raw_data['head_branch']}'
        );
        '''
        logger.debug('Job query: %s', job_data_query)
        cur.execute(job_data_query)
        for step in job.steps:
            duration_seconds_timedelta = step.completed_at - step.started_at
            duration_seconds = round(duration_seconds_timedelta.total_seconds())

            step_data_query = f'''
            INSERT INTO github_workflow_steps_test(
            parent_job_id, name, conclusion,
            number, started_at, completed_at,
            duration_seconds)
            VALUES(
            '{job_id}', '{step.name}','{step.conclusion}',
            '{step.number}', '{step.started_at}', '{step.completed_at}',
            '{duration_seconds}'
            );
            '''
            logger.debug('Step query: %s', step_data_query)
            cur.execute(step_data_query)

    conn.commit()
    cur.close()
    conn.close()
    g.close()
if __name__ == "__main__":
    main()
