// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

'use strict';

const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const {execFileSync} = require('node:child_process');
const {PLATFORMS} = require('./comment.js');

const MARKER = '<!-- openvino-bom:unified-gold -->';
const MAX_ARCHIVE = 32 * 1024 * 1024;
const MAX_BLOB = 2 * 1024 * 1024;
const BUILD_NAMES = new Set(['Build', 'Build / Build']);
const BOM_NAMES = new Set(['BOM tests', 'BOM tests / BOM tests']);
const OUTPUT = 'bom-gold-output';
const STATE = 'bom-gold-state.json';

function requireValue(value, message) {
    if (!value) throw new Error(message);
}

function engine(input) {
    return JSON.parse(execFileSync(process.env.BOM_PYTHON || 'python3', [path.join(__dirname, 'patch.py')], {
        input: JSON.stringify(input), encoding: 'utf8', maxBuffer: 128 * 1024 * 1024,
        timeout: 60000,
    }));
}

function identity(pr) {
    return [pr.number, pr.head.sha, pr.head.repo.id, pr.head.ref,
        pr.base.repo.id, pr.base.ref, pr.base.sha];
}

function livePR(pr, repositoryId, sourceId, branch) {
    return pr.state === 'open' && pr.base?.repo?.id === repositoryId &&
        pr.head?.repo?.id === sourceId && pr.head.ref === branch &&
        /^[a-f0-9]{40}$/i.test(pr.head.sha) && /^[a-f0-9]{40}$/i.test(pr.base.sha);
}

function hash(value) {
    return crypto.createHash('sha256').update(JSON.stringify(value)).digest('hex');
}

function latestJobs(jobs) {
    const latest = new Map();
    for (const job of jobs) {
        const old = latest.get(job.name);
        if (!old || job.run_attempt > old.run_attempt ||
            (job.run_attempt === old.run_attempt && job.id > old.id)) latest.set(job.name, job);
    }
    return [...latest.values()];
}

function oneJob(jobs, names) {
    const found = jobs.filter(job => names.has(job.name));
    requireValue(found.length <= 1, 'Ambiguous product jobs');
    return found[0];
}

function complete(job, conclusion) {
    return job?.status === 'completed' && job.conclusion === conclusion;
}

class Aggregator {
    constructor({github, context, core, runEngine = engine}) {
        Object.assign(this, {github, context, core, runEngine});
        this.repo = context.repo;
        this.repositoryId = context.payload.repository.id;
    }

    async pages(method, args) {
        const result = await this.github.paginate(method, {...args, per_page: 100});
        requireValue(result.length < 1000, 'API result limit reached; refusing an incomplete platform set');
        return result;
    }

    async pr(number) {
        return (await this.github.rest.pulls.get({...this.repo, pull_number: number})).data;
    }

    async write(pr, text) {
        const marker = `${MARKER}\n`;
        const comments = await this.pages(this.github.rest.issues.listComments, {
            ...this.repo, issue_number: pr.number,
        });
        const own = comments.filter(comment => comment.user?.type === 'Bot' &&
            comment.user.login === 'github-actions[bot]' && comment.body?.startsWith(marker));
        own.sort((a, b) => a.id - b.id);
        const server = this.context.serverUrl || 'https://github.com';
        const docs = `${server}/${this.repo.owner}/${this.repo.repo}/blob/` +
            `${encodeURIComponent(this.context.payload.repository.default_branch)}/docs/dev/ci/github_actions/bom_tests.md`;
        const body = `${marker}### Unified GOLD BOM update\n\nHead: \`${pr.head.sha}\`\n\n${text}\n\n` +
            '**Accepting GOLD is engineering approval of the public package layout, not merely clearing red CI.** ' +
            `Review every change first. [BOM documentation](${docs}) · ` +
            `[Notifier run](${server}/${this.repo.owner}/${this.repo.repo}/actions/runs/${this.context.runId})`;
        const current = await this.pr(pr.number);
        if (JSON.stringify(identity(pr)) !== JSON.stringify(identity(current)) || current.state !== 'open') return;
        if (own[0]) {
            if (own[0].body !== body) await this.github.rest.issues.updateComment({
                ...this.repo, comment_id: own[0].id, body,
            });
        } else {
            await this.github.rest.issues.createComment({...this.repo, issue_number: pr.number, body});
        }
        // Repair duplicates from old notifier versions without editing user comments.
        for (const duplicate of own.slice(1)) {
            await this.github.rest.issues.deleteComment({...this.repo, comment_id: duplicate.id});
        }
    }

    async source(pr) {
        const repo = {owner: pr.head.repo.owner.login, repo: pr.head.repo.name};
        const {data} = await this.github.rest.repos.get(repo);
        requireValue(data.id === pr.head.repo.id, 'Source repository identity changed');
        const commit = (await this.github.rest.git.getCommit({...repo, commit_sha: pr.head.sha})).data;
        requireValue(commit.sha === pr.head.sha, 'Source commit identity mismatch');
        const trees = new Map();
        const tree = async sha => {
            if (!trees.has(sha)) {
                const {data} = await this.github.rest.git.getTree({...repo, tree_sha: sha});
                requireValue(!data.truncated && data.sha === sha && data.tree.length <= 20000,
                    'Incomplete or oversized source tree');
                trees.set(sha, data.tree);
            }
            return trees.get(sha);
        };
        const directory = async parts => {
            let sha = commit.tree.sha;
            for (const part of parts) {
                const entries = (await tree(sha)).filter(entry => entry.path === part);
                requireValue(entries.length <= 1, 'Duplicate tree path');
                if (!entries.length) return null;
                requireValue(entries[0].type === 'tree' && entries[0].mode === '040000',
                    'Non-directory in GOLD/workflow path');
                sha = entries[0].sha;
            }
            return tree(sha);
        };
        let total = 0;
        const blob = async entry => {
            requireValue(entry.type === 'blob' && entry.mode === '100644' &&
                entry.size > 0 && entry.size <= MAX_BLOB, 'Invalid source blob');
            total += entry.size;
            requireValue(total <= MAX_ARCHIVE, 'Source BOM/workflow size limit exceeded');
            const {data} = await this.github.rest.git.getBlob({...repo, file_sha: entry.sha});
            requireValue(data.sha === entry.sha && data.encoding === 'base64' && data.size === entry.size,
                'Source blob identity mismatch');
            const encoded = data.content.replace(/\n/g, '');
            requireValue(/^[A-Za-z0-9+/]*={0,2}$/.test(encoded), 'Invalid blob encoding');
            const raw = Buffer.from(encoded, 'base64');
            requireValue(raw.length === entry.size &&
                crypto.createHash('sha1').update(`blob ${raw.length}\0`).update(raw).digest('hex') === entry.sha,
            'Source blob hash mismatch');
            return raw;
        };
        return {directory, blob};
    }

    async expected(pr, source) {
        const files = await this.pages(this.github.rest.pulls.listFiles, {
            ...this.repo, pull_number: pr.number,
        });
        requireValue(files.length === pr.changed_files, 'PR changed-file list is incomplete');
        const entries = await source.directory(['.github', 'workflows']);
        requireValue(entries, 'PR workflows are missing');
        const workflows = [];
        for (const [file, [name, configuration]] of Object.entries(PLATFORMS)) {
            const entry = entries.find(value => value.path === file);
            requireValue(entry, 'A product workflow is missing from the PR revision');
            workflows.push({
                file, name, configuration,
                trusted: fs.readFileSync(path.join(__dirname, '..', '..', 'workflows', file), 'utf8'),
                head: (await source.blob(entry)).toString('utf8'),
            });
        }
        return this.runEngine({
            operation: 'policy', workflows, base: pr.base.ref,
            // Renames affect both the old and new paths. For GitHub's large-diff
            // fallback, conservatively include every path-filtered workflow.
            files: files.flatMap(file => [file.filename, ...(file.previous_filename ? [file.previous_filename] : [])]),
            largeDiff: pr.commits > 1000,
        });
    }

    validRun(run, platform, pr, workflowId) {
        return run.repository?.id === this.repositoryId && run.head_repository?.id === pr.head.repo.id &&
            run.head_sha === pr.head.sha && run.head_branch === pr.head.ref &&
            run.event === 'pull_request' && run.workflow_id === workflowId &&
            run.path === `.github/workflows/${platform.file}` && run.name === platform.name;
    }

    async selection(pr) {
        // Fork workflow_run PR associations may be empty. A unique live branch
        // alone is insufficient after a PR retarget: an old run has the old base
        // in its immutable event context, even when it is rerun later.
        const result = await this.github.graphql(`
            query($owner: String!, $repo: String!, $number: Int!) {
                repository(owner: $owner, name: $repo) {
                    databaseId
                    pullRequest(number: $number) {
                        number state headRefOid baseRefName createdAt
                        timelineItems(last: 1, itemTypes: [BASE_REF_CHANGED_EVENT]) {
                            nodes { ... on BaseRefChangedEvent { createdAt } }
                        }
                    }
                }
            }`, {...this.repo, number: pr.number});
        const repository = result.repository;
        const current = repository?.pullRequest;
        requireValue(repository?.databaseId === this.repositoryId && current?.number === pr.number &&
            current.state === 'OPEN' && current.headRefOid === pr.head.sha &&
            current.baseRefName === pr.base.ref && Date.parse(current.createdAt) === Date.parse(pr.created_at),
        'Live GraphQL PR identity mismatch');
        const events = current.timelineItems?.nodes;
        requireValue(Array.isArray(events) && events.length <= 1, 'Incomplete PR base-change history');
        const timestamps = [Date.parse(current.createdAt), ...events.map(event => Date.parse(event?.createdAt))];
        requireValue(timestamps.every(Number.isFinite), 'Invalid PR base-selection timestamp');
        return Math.max(...timestamps);
    }

    async platform(pr, platform, selectedAt) {
        const {data: workflow} = await this.github.rest.actions.getWorkflow({
            ...this.repo, workflow_id: platform.file,
        });
        requireValue(workflow.path === `.github/workflows/${platform.file}` && workflow.name === platform.name,
            'Workflow identity mismatch');
        const runs = await this.pages(this.github.rest.actions.listWorkflowRuns, {
            ...this.repo, workflow_id: workflow.id, event: 'pull_request', head_sha: pr.head.sha,
        });
        const matching = runs.filter(run => run.head_repository?.id === pr.head.repo.id &&
            run.head_branch === pr.head.ref && run.head_sha === pr.head.sha);
        matching.sort((a, b) => b.run_number - a.run_number || b.run_attempt - a.run_attempt);
        if (!matching.length) return {...platform, state: 'missing: workflow has not started'};
        const {data: run} = await this.github.rest.actions.getWorkflowRun({
            ...this.repo, run_id: matching[0].id,
        });
        requireValue(this.validRun(run, platform, pr, workflow.id), 'Run identity mismatch');
        requireValue(Number.isFinite(Date.parse(run.created_at)), 'Invalid run creation timestamp');
        if (Date.parse(run.created_at) < selectedAt) {
            return {...platform, state: 'blocked: run predates this PR/base selection; start a new PR workflow run'};
        }
        if (run.pull_requests?.length) {
            requireValue(run.pull_requests.some(value => value.number === pr.number &&
                value.base?.ref === pr.base.ref && value.base?.repo?.id === this.repositoryId &&
                value.head?.sha === pr.head.sha && value.head?.repo?.id === pr.head.repo.id),
            'Run PR/base association mismatch');
        }
        const record = {...platform, run: {id: run.id, attempt: run.run_attempt, status: run.status}};
        if (run.status !== 'completed') return {...record, state: 'pending: workflow or rerun is running'};
        const jobs = latestJobs(await this.pages(this.github.rest.actions.listJobsForWorkflowRun, {
            ...this.repo, run_id: run.id, filter: 'all',
        }));
        requireValue(jobs.every(job => job.run_id === run.id &&
            Number.isSafeInteger(job.run_attempt) && job.run_attempt <= run.run_attempt), 'Job identity mismatch');
        const build = oneJob(jobs, BUILD_NAMES);
        const bom = oneJob(jobs, BOM_NAMES);
        const dependencies = platform.dependencies.map(name => oneJob(jobs, new Set([name])));
        // A skipped Build alone is never evidence of Smart CI: Docker or Smart_CI
        // might have failed. Require all prerequisites and the gated BOM to agree.
        if (complete(build, 'skipped') && complete(bom, 'skipped') &&
            dependencies.every(job => complete(job, 'success') && job.run_attempt === run.run_attempt) &&
            dependencies.find(job => job.name === 'Smart_CI')?.steps?.some(step =>
                step.name === 'Get affected components' && step.conclusion === 'success')) {
            return {...record, state: 'skipped by Smart CI'};
        }
        if (!complete(build, 'success')) return {...record, state: `blocked: Build ${build?.conclusion || 'missing/pending'}`};
        requireValue(build.steps?.filter(step => step.name === 'Upload CPack component BOM' &&
            step.conclusion === 'success').length === 1, 'Successful BOM upload step is missing');
        const artifacts = await this.pages(this.github.rest.actions.listWorkflowRunArtifacts, {
            ...this.repo, run_id: run.id,
        });
        const found = artifacts.filter(artifact => artifact.name === `bom_${platform.configuration}`);
        requireValue(found.length === 1 && !found[0].expired, 'Required BOM artifact is missing, expired, or ambiguous');
        const artifact = found[0];
        const created = Date.parse(artifact.created_at);
        requireValue(Number.isSafeInteger(artifact.id) && artifact.size_in_bytes > 0 &&
            artifact.size_in_bytes <= MAX_ARCHIVE &&
            created >= Date.parse(build.started_at) && created <= Date.parse(build.completed_at) &&
            artifact.workflow_run?.id === run.id && artifact.workflow_run?.head_sha === pr.head.sha &&
            artifact.workflow_run?.head_repository_id === pr.head.repo.id,
        'Artifact does not belong to the successful Build for this PR head');
        return {...record, state: 'ready', build: {id: build.id, attempt: build.run_attempt},
            artifact: {id: artifact.id, digest: artifact.digest || null, size: artifact.size_in_bytes}};
    }

    async snapshot(pr) {
        const branches = await this.pages(this.github.rest.pulls.list, {
            ...this.repo, state: 'open', head: `${pr.head.repo.owner.login}:${pr.head.ref}`,
        });
        const live = [];
        for (const candidate of branches) {
            const current = await this.pr(candidate.number);
            if (livePR(current, this.repositoryId, pr.head.repo.id, pr.head.ref)) live.push(current);
        }
        requireValue(live.length === 1 && live[0].number === pr.number &&
            JSON.stringify(identity(live[0])) === JSON.stringify(identity(pr)),
        'Live PR source/base association changed or became ambiguous');
        const source = await this.source(pr);
        const expected = await this.expected(pr, source);
        const selectedAt = await this.selection(pr);
        requireValue(expected.length > 0, 'No expected product PR platforms');
        const platforms = [];
        for (const platform of expected) {
            try {
                platforms.push(await this.platform(pr, platform, selectedAt));
            } catch (error) {
                error.platform = platform.configuration;
                throw error;
            }
        }
        return {source, platforms, signature: hash({pr: identity(pr), selectedAt, platforms})};
    }

    status(platforms) {
        return platforms.map(platform => `- \`${platform.configuration}\`: ${platform.state}`).join('\n');
    }

    async candidates() {
        const incoming = this.context.payload.workflow_run;
        if (!incoming || incoming.event !== 'pull_request' ||
            incoming.repository?.id !== this.repositoryId ||
            !Object.entries(PLATFORMS).some(([file, [name]]) =>
                incoming.path === `.github/workflows/${file}` && incoming.name === name)) return [];
        this.triggerError = null;
        try {
            const {data: run} = await this.github.rest.actions.getWorkflowRun({...this.repo, run_id: incoming.id});
            requireValue(run.id === incoming.id && run.workflow_id === incoming.workflow_id &&
                run.repository?.id === this.repositoryId && run.head_repository?.id === incoming.head_repository?.id &&
                run.path === incoming.path && run.name === incoming.name && run.event === 'pull_request' &&
                run.head_branch === incoming.head_branch && run.head_sha === incoming.head_sha,
            'Trigger run identity mismatch');
        } catch (error) {
            // Still retire an old proposal using independently verified live PR
            // identities if the run API is unavailable or returns invalid data.
            this.triggerError = error;
        }
        // Query by source branch even for an old-SHA event, so it can retire a
        // previously advertised patch after a push. Branch names are API data.
        const prs = await this.pages(this.github.rest.pulls.list, {
            ...this.repo, state: 'open', head: `${incoming.head_repository.owner.login}:${incoming.head_branch}`,
        });
        const current = [];
        for (const value of prs) {
            const pr = await this.pr(value.number);
            if (livePR(pr, this.repositoryId, incoming.head_repository.id, incoming.head_branch)) current.push(pr);
        }
        // Empty fork run associations cannot disambiguate multiple target bases.
        if (current.length > 1) {
            for (const pr of current) await this.write(pr,
                '**Blocked: multiple open PR bases share this source branch.** Run associations are ambiguous; no patch is offered and previous patches are superseded.');
            throw new Error('Multiple open PR bases for this source branch; cannot safely associate fork runs');
        }
        return current;
    }

    async prepare() {
        fs.rmSync(OUTPUT, {recursive: true, force: true});
        fs.rmSync(STATE, {force: true});
        const prepared = [];
        const prs = await this.candidates();
        for (const pr of prs) {
            await this.write(pr, 'Checking all expected platforms. Any previous patch is superseded; no patch is currently offered.');
            try {
                if (this.triggerError) throw this.triggerError;
                const snapshot = await this.snapshot(pr);
                const status = this.status(snapshot.platforms);
                if (snapshot.platforms.some(platform => !['ready', 'skipped by Smart CI'].includes(platform.state))) {
                    await this.write(pr, `**Waiting / blocked — no partial patch is published.**\n\n${status}`);
                    continue;
                }
                const platforms = [];
                for (const platform of snapshot.platforms.filter(value => value.state === 'ready')) {
                    const {data} = await this.github.rest.actions.downloadArtifact({
                        ...this.repo, artifact_id: platform.artifact.id, archive_format: 'zip',
                    });
                    const archive = Buffer.from(data);
                    requireValue(archive.length <= MAX_ARCHIVE, 'Downloaded artifact exceeds size limit');
                    if (platform.artifact.digest) {
                        requireValue(platform.artifact.digest ===
                            `sha256:${crypto.createHash('sha256').update(archive).digest('hex')}`, 'Artifact digest mismatch');
                    }
                    const entries = await snapshot.source.directory(['boms', platform.configuration]) || [];
                    requireValue(entries.length <= 512, 'Too many GOLD manifests');
                    const gold = [];
                    for (const entry of entries) {
                        requireValue(/^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}\.json$/.test(entry.path),
                            'Unexpected GOLD entry');
                        gold.push({name: entry.path, content: (await snapshot.source.blob(entry)).toString('base64')});
                    }
                    platforms.push({
                        configuration: platform.configuration,
                        platform: platform.configuration.startsWith('windows_') ? 'windows' :
                            platform.configuration.startsWith('macos_') ? 'macos' : 'linux',
                        archive: archive.toString('base64'), gold,
                    });
                }
                const patch = this.runEngine({operation: 'patch', platforms});
                const current = await this.pr(pr.number);
                if (JSON.stringify(identity(pr)) !== JSON.stringify(identity(current)) ||
                    (await this.snapshot(current)).signature !== snapshot.signature) {
                    await this.write(current, 'State changed while preparing. Previous patches are superseded; waiting for fresh platform events.');
                    continue;
                }
                if (!patch) {
                    const message = platforms.length ? 'All clear: generated BOMs match GOLD for all affected platforms.' :
                        'All expected platforms were skipped by Smart CI.';
                    await this.write(pr, `**${message}** No patch is needed.\n\n${status}`);
                    continue;
                }
                fs.mkdirSync(OUTPUT, {recursive: true});
                const filename = `pr-${pr.number}-gold.patch`;
                fs.writeFileSync(path.join(OUTPUT, filename), patch, 'utf8');
                prepared.push({number: pr.number, identity: identity(pr), signature: snapshot.signature, filename, status});
                await this.write(pr, `All expected Builds succeeded and BOM data validated. Uploading the combined patch; it is not available until publication completes.\n\n${status}`);
            } catch (error) {
                const platform = Object.values(PLATFORMS).some(([, configuration]) => configuration === error.platform)
                    ? ` Platform: \`${error.platform}\`.` : '';
                await this.write(pr, '**Blocked: platform, API, artifact, or baseline validation failed.**' +
                    platform + ' No patch is offered; any previous patch is superseded. See the trusted BOM PR comments workflow log.');
                throw error;
            }
        }
        fs.writeFileSync(STATE, JSON.stringify(prepared));
        this.core.setOutput('upload', prepared.length > 0 ? 'true' : 'false');
    }

    async publish(artifactId) {
        if (!fs.existsSync(STATE)) return;
        const prepared = JSON.parse(fs.readFileSync(STATE, 'utf8'));
        for (const item of prepared) {
            const pr = await this.pr(item.number);
            try {
                if (JSON.stringify(identity(pr)) !== JSON.stringify(item.identity) ||
                    (await this.snapshot(pr)).signature !== item.signature) {
                    await this.write(pr, 'Previous patches are superseded: the PR or platform runs changed before publication. Waiting for fresh results; no patch is offered.');
                    continue;
                }
                requireValue(/^[1-9]\d*$/.test(String(artifactId)), 'Combined patch upload did not succeed');
                const {data: artifact} = await this.github.rest.actions.getArtifact({
                    ...this.repo, artifact_id: Number(artifactId),
                });
                requireValue(artifact.id === Number(artifactId) && !artifact.expired &&
                    artifact.workflow_run?.id === Number(this.context.runId) &&
                    artifact.name === `gold-bom-patch-${this.context.runId}-${process.env.GITHUB_RUN_ATTEMPT || '1'}`,
                'Uploaded artifact identity mismatch');
                const server = this.context.serverUrl || 'https://github.com';
                const url = `${server}/${this.repo.owner}/${this.repo.repo}/actions/runs/${this.context.runId}/artifacts/${artifact.id}`;
                await this.write(pr, `**Combined GOLD patch ready.** [Download patch artifact](${url}) ` +
                    `(expires with workflow artifacts). It contains \`${item.filename}\` for this head only.\n\n` +
                    'After engineering review, extract the artifact and run from your clean PR checkout:\n\n' +
                    `\`\`\`sh\ngit apply --check ${item.filename}\ngit apply ${item.filename}\n\`\`\`\n\n` +
                    'Review and commit the GOLD diff in this PR, then rerun CI. Do not apply a patch from an older head or after rerunning a Build.\n\n' +
                    'Every expected non-skipped Build succeeded. BOM test / overall workflow failures do not prevent this proposal.\n\n' +
                    item.status);
            } catch (error) {
                await this.write(pr, '**Blocked: patch upload or final validation failed.** No patch is offered; any previous patch is superseded. See the trusted workflow log.');
                throw error;
            }
        }
    }
}

module.exports = {Aggregator, engine, latestJobs, livePR};
