// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

'use strict';

const PLATFORMS = Object.freeze({
    'linux_arm64.yml': ['Linux ARM64 (Ubuntu 22.04, Python 3.11)', 'ubuntu_22_04_arm64'],
    'mac_arm64.yml': ['macOS ARM64 (mac-15, Python 3.11)', 'macos_arm64'],
    'manylinux_2_28.yml': ['Manylinux 2_28', 'manylinux_2_28_x86_64'],
    'ubuntu_22.yml': ['Linux (Ubuntu 22.04, Python 3.11)', 'ubuntu_22_04_x86_64'],
    'ubuntu_22_arm64_cross_compilation.yml': [
        'Linux (Ubuntu 22.04, ARM64 cross-compilation, Python 3.11)', 'ubuntu_22_04_arm64_cross_compile',
    ],
    'ubuntu_24.yml': ['Linux (Ubuntu 24.04, Python 3.12)', 'ubuntu_24_04_x86_64'],
    'ubuntu_26.yml': ['Linux (Ubuntu 26.04, Python 3.14)', 'ubuntu_26_04_x86_64'],
    'windows_vs2022_debug.yml': ['Windows (VS 2022, Python 3.11, Debug)', 'windows_vs2022_Debug'],
    'windows_vs2022_release.yml': ['Windows (VS 2022, Python 3.11, Release)', 'windows_vs2022_Release'],
    'windows_vs2026_release.yml': ['Windows (VS 2026, Python 3.11, Release)', 'windows_vs2026_Release'],
});

const COMPARE_STEP = 'Compare BOM against GOLD';
const DIFF_TITLE = 'BOM package layout changed';
const JOB_NAMES = new Set(['BOM tests / BOM tests', 'BOM tests']);

function platformFor(run) {
    const entry = Object.entries(PLATFORMS).find(([file, [name]]) =>
        run.path === `.github/workflows/${file}` && run.name === name);
    return entry && {name: entry[1][0], configuration: entry[1][1]};
}

function isCurrentPR(pr, run, repositoryId) {
    return pr.state === 'open' && pr.base?.repo?.id === repositoryId &&
        pr.head?.repo?.id === run.head_repository?.id && pr.head?.sha === run.head_sha;
}

function newer(left, right) {
    return left.run_number > right.run_number ||
        (left.run_number === right.run_number && left.run_attempt > right.run_attempt);
}

function checkRunId(job, context) {
    const url = new URL(job.check_run_url);
    const api = new URL(context.apiUrl || 'https://api.github.com');
    const prefix = `${api.pathname.replace(/\/$/, '')}/repos/${context.repo.owner}/${context.repo.repo}/check-runs/`;
    if (url.origin !== api.origin || !url.pathname.startsWith(prefix) ||
        !/^\d+$/.test(url.pathname.slice(prefix.length)) || url.search || url.hash) {
        throw new Error('BOM job has an unexpected check-run URL');
    }
    return Number(url.pathname.slice(prefix.length));
}

async function notify({github, context, core}) {
    const skip = message => core.info(`BOM comment: ${message}`);
    const event = context.payload;
    const incoming = event.workflow_run;
    const repositoryId = event.repository?.id;
    if (event.action !== 'completed' || !repositoryId || incoming?.event !== 'pull_request' ||
        incoming.repository?.id !== repositoryId || !platformFor(incoming)) {
        return skip('not a recognized pull-request workflow');
    }

    const repo = context.repo;
    const {data: run} = await github.rest.actions.getWorkflowRun({...repo, run_id: incoming.id});
    const platform = platformFor(run);
    if (!platform || run.repository?.id !== repositoryId || run.event !== 'pull_request' ||
        run.status !== 'completed' || run.run_attempt !== incoming.run_attempt ||
        !run.head_repository?.id || !/^[a-f0-9]{40}$/i.test(run.head_sha)) {
        return skip('run no longer matches the completed event');
    }
    const {data: workflow} = await github.rest.actions.getWorkflow({...repo, workflow_id: run.workflow_id});
    if (workflow.path !== run.path || workflow.name !== run.name) {
        return skip('workflow identity mismatch');
    }

    const candidates = new Set((run.pull_requests || []).map(pr => pr.number));
    // Fork workflow_run payloads can omit pull_requests; the association API is only
    // a source of candidates, never authority for a PR's current head or base.
    if (!candidates.size) {
        const associated = await github.paginate(github.rest.repos.listPullRequestsAssociatedWithCommit, {
            ...repo, commit_sha: run.head_sha, per_page: 100,
        });
        for (const pr of associated) candidates.add(pr.number);
    }
    if (!candidates.size && run.head_repository.owner?.login && run.head_branch) {
        const byBranch = await github.paginate(github.rest.pulls.list, {
            ...repo, state: 'open',
            head: `${run.head_repository.owner.login}:${run.head_branch}`, per_page: 100,
        });
        for (const pr of byBranch) candidates.add(pr.number);
    }
    if (!candidates.size) return skip('no associated pull requests');

    const runs = await github.paginate(github.rest.actions.listWorkflowRuns, {
        ...repo, workflow_id: run.workflow_id, event: 'pull_request',
        head_sha: run.head_sha, per_page: 100,
    });
    if (runs.some(other => other.head_repository?.id === run.head_repository.id &&
        other.head_branch === run.head_branch && newer(other, run))) {
        return skip('a newer run already exists for this head');
    }

    const jobs = await github.paginate(github.rest.actions.listJobsForWorkflowRun, {
        ...repo, run_id: run.id, filter: 'latest', per_page: 100,
    });
    const latest = new Map();
    for (const job of jobs) {
        if (!latest.has(job.name) || latest.get(job.name).id < job.id) latest.set(job.name, job);
    }
    const bomJobs = [...latest.values()].filter(job => JOB_NAMES.has(job.name));
    if (bomJobs.length !== 1) return skip('no unambiguous BOM job');
    const job = bomJobs[0];
    if (!Number.isSafeInteger(job.id)) throw new Error('Invalid BOM job ID');
    const compare = job.steps?.find(step => step.name === COMPARE_STEP);
    let status;
    if (job.conclusion === 'success' && compare?.conclusion === 'success') {
        status = 'resolved';
    } else if (job.conclusion === 'failure' && compare?.conclusion === 'failure') {
        const annotations = await github.paginate(github.rest.checks.listAnnotations, {
            ...repo, check_run_id: checkRunId(job, context), per_page: 100,
        });
        if (annotations.some(annotation =>
            annotation.title === DIFF_TITLE && annotation.annotation_level === 'failure')) {
            status = 'changed';
        }
    }
    if (!status) return skip('no verified layout difference or successful comparison');

    const marker = `<!-- openvino-bom:${platform.configuration} -->`;
    const server = context.serverUrl || 'https://github.com';
    const jobUrl = `${server}/${repo.owner}/${repo.repo}/actions/runs/${run.id}/job/${job.id}`;
    const docsUrl = `${server}/${repo.owner}/${repo.repo}/blob/` +
        `${encodeURIComponent(event.repository.default_branch)}/docs/dev/ci/github_actions/bom_tests.md`;
    const metadata = `<!-- bom-run:${run.run_number}:${run.run_attempt} -->`;
    const text = status === 'changed'
        ? 'Package components or file paths differ from the GOLD baseline. If intentional, update ' +
          `the affected GOLD JSON files in \`boms/${platform.configuration}\` in this PR; otherwise fix the packaging regression.`
        : 'Resolved: the latest BOM comparison matches the GOLD baseline.';
    const body = `${marker}\n${metadata}\n### BOM: ${platform.configuration}\n\n${text}\n\n` +
        `[BOM job](${jobUrl}) · [BOM documentation and update instructions](${docsUrl})`;

    for (const number of candidates) {
        if (!Number.isSafeInteger(number) || number <= 0) continue;
        const {data: pr} = await github.rest.pulls.get({...repo, pull_number: number});
        if (!isCurrentPR(pr, run, repositoryId)) continue;
        const comments = await github.paginate(github.rest.issues.listComments, {
            ...repo, issue_number: number, per_page: 100,
        });
        const own = comments.filter(comment => comment.user?.type === 'Bot' &&
            comment.user.login === 'github-actions[bot]' && comment.body?.startsWith(`${marker}\n`));
        if (own.length > 1) throw new Error('Multiple notifier comments exist for one BOM configuration');
        const existing = own[0];
        if (!existing && status === 'resolved') continue;
        const previous = existing?.body.match(/<!-- bom-run:(\d+):(\d+) -->/);
        if (previous && newer({run_number: Number(previous[1]), run_attempt: Number(previous[2])}, run)) continue;
        if (existing?.body === body) continue;
        // Recheck immediately before writing so a PR push while reading APIs cannot
        // cause an old result to be posted against the new head.
        const {data: currentPR} = await github.rest.pulls.get({...repo, pull_number: number});
        const {data: currentRun} = await github.rest.actions.getWorkflowRun({...repo, run_id: run.id});
        if (!isCurrentPR(currentPR, run, repositoryId) ||
            currentRun.run_attempt !== run.run_attempt || currentRun.status !== 'completed') continue;
        if (existing) {
            await github.rest.issues.updateComment({...repo, comment_id: existing.id, body});
        } else {
            await github.rest.issues.createComment({...repo, issue_number: number, body});
        }
    }
}

module.exports = notify;
module.exports.PLATFORMS = PLATFORMS;
