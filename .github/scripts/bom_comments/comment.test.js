// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {test} = require('node:test');
const notify = require('./comment.js');

function fixture() {
    const run = {
        id: 100, repository: {id: 1}, head_repository: {id: 2, owner: {login: 'fork-owner'}},
        head_sha: 'a'.repeat(40), head_branch: 'feature',
        event: 'pull_request', status: 'completed', run_attempt: 1,
        run_number: 10, workflow_id: 50, pull_requests: [{number: 7}],
        path: '.github/workflows/ubuntu_22.yml',
        name: 'Linux (Ubuntu 22.04, Python 3.11)',
    };
    const pr = {number: 7, state: 'open', base: {repo: {id: 1}},
        head: {repo: {id: 2}, sha: run.head_sha}};
    const job = {
        id: 200, name: 'BOM tests / BOM tests', conclusion: 'failure',
        steps: [{name: 'Compare BOM against GOLD', conclusion: 'failure'}],
        check_run_url: 'https://api.github.com/repos/owner/repo/check-runs/300',
    };
    const state = {
        run, pr, jobs: [job], runs: [run], comments: [],
        annotations: [{title: 'BOM package layout changed', annotation_level: 'failure'}],
        associated: [pr], byBranch: [pr], writes: [], calls: [], logs: [],
    };
    const context = {
        repo: {owner: 'owner', repo: 'repo'},
        apiUrl: 'https://api.github.com', serverUrl: 'https://github.com',
        payload: {action: 'completed', repository: {id: 1, default_branch: 'master'},
            workflow_run: structuredClone(run)},
    };
    const endpoint = (name, fn) => Object.assign(async args => {
        state.calls.push({name, args});
        return {data: await fn(args)};
    }, {endpointName: name});
    const github = {rest: {
        actions: {
            getWorkflowRun: endpoint('run', () => state.run),
            getWorkflow: endpoint('workflow', () => state.workflow || {name: run.name, path: run.path}),
            listWorkflowRuns: endpoint('runs', () => state.runs),
            listJobsForWorkflowRun: endpoint('jobs', () => state.jobs),
        },
        repos: {listPullRequestsAssociatedWithCommit: endpoint('associated', () => state.associated)},
        pulls: {
            get: endpoint('pr', () => state.pr),
            list: endpoint('branch-prs', () => state.byBranch),
        },
        checks: {listAnnotations: endpoint('annotations', () => state.annotations)},
        issues: {
            listComments: endpoint('comments', () => state.comments),
            createComment: endpoint('create', args => state.writes.push({kind: 'create', ...args})),
            updateComment: endpoint('update', args => state.writes.push({kind: 'update', ...args})),
        },
    }};
    github.paginate = async (method, args) => {
        assert.equal(args.per_page, 100);
        const {data} = await method(args);
        return data;
    };
    state.context = context;
    state.execute = () => notify({github, context, core: {info: message => state.logs.push(message)}});
    return state;
}

function ownComment(body) {
    return {id: 400, user: {type: 'Bot', login: 'github-actions[bot]'}, body};
}

test('fork lookup falls back to branch when commit association is empty', async () => {
    const f = fixture();
    f.run.pull_requests = [];
    f.associated = [];
    await f.execute();
    assert.equal(f.writes.length, 1);
    const lookup = f.calls.find(call => call.name === 'branch-prs');
    assert.equal(lookup.args.head, 'fork-owner:feature');
    assert.equal(lookup.args.state, 'open');
});

test('branch fallback rejects a mismatching PR head SHA or repository', async () => {
    for (const mismatch of ['sha', 'repository']) {
        const f = fixture();
        f.run.pull_requests = [];
        f.associated = [];
        if (mismatch === 'sha') f.pr.head.sha = 'b'.repeat(40);
        else f.pr.head.repo.id = 99;
        await f.execute();
        assert.equal(f.writes.length, 0);
    }
});

test('creates a fixed, linked comment for a verified layout change, including fork PRs', async () => {
    const f = fixture();
    f.annotations[0].message = '[evil](javascript:alert(1))';
    await f.execute();
    assert.equal(f.writes.length, 1);
    const {body} = f.writes[0];
    assert.match(body, /openvino-bom:ubuntu_22_04_x86_64/);
    assert.match(body, /boms\/ubuntu_22_04_x86_64/);
    assert.match(body, /https:\/\/github.com\/owner\/repo\/actions\/runs\/100\/job\/200/);
    assert.match(body, /docs\/dev\/ci\/github_actions\/bom_tests.md/);
    assert.doesNotMatch(body, /evil|javascript/);
});

test('reruns update one own comment and identical notifications do nothing', async () => {
    const f = fixture();
    await f.execute();
    f.comments = [ownComment(f.writes[0].body)];
    f.writes = [];
    await f.execute();
    assert.equal(f.writes.length, 0);
    f.run.run_attempt = 2;
    f.context.payload.workflow_run.run_attempt = 2;
    await f.execute();
    assert.equal(f.writes.length, 1);
    assert.equal(f.writes[0].kind, 'update');
    assert.equal(f.writes[0].comment_id, 400);
});

test('green comparison resolves an existing comment but never creates a green comment', async () => {
    const f = fixture();
    await f.execute();
    f.comments = [ownComment(f.writes[0].body)];
    f.writes = [];
    f.jobs[0].conclusion = 'success';
    f.jobs[0].steps[0].conclusion = 'success';
    await f.execute();
    assert.match(f.writes[0].body, /Resolved:/);
    f.comments = [];
    f.writes = [];
    await f.execute();
    assert.equal(f.writes.length, 0);
});

for (const [label, change] of [
    ['setup failure', f => { f.jobs[0].steps = []; }],
    ['missing baseline', f => { f.annotations[0].title = 'Missing GOLD baseline'; }],
    ['generic comparator failure', f => { f.annotations = []; }],
    ['nonfailure annotation', f => { f.annotations[0].annotation_level = 'warning'; }],
    ['skipped comparator', f => { f.jobs[0].steps[0].conclusion = 'skipped'; }],
    ['cancelled job', f => { f.jobs[0].conclusion = 'cancelled'; }],
    ['upload failure after green comparator', f => { f.jobs[0].steps[0].conclusion = 'success'; }],
    ['unrecognized job', f => { f.jobs[0].name = 'Build'; }],
    ['ambiguous jobs', f => { f.jobs.push({...f.jobs[0], id: 201, name: 'BOM tests'}); }],
    ['non-PR event', f => { f.context.payload.workflow_run.event = 'push'; }],
    ['non-completed event', f => { f.context.payload.action = 'requested'; }],
    ['wrong event repository', f => { f.context.payload.workflow_run.repository.id = 99; }],
    ['wrong API repository', f => { f.run.repository.id = 99; }],
    ['wrong API event', f => { f.run.event = 'push'; }],
    ['unrecognized path', f => { f.context.payload.workflow_run.path = '.github/workflows/evil.yml'; }],
    ['wrong API path', f => { f.run.path = '.github/workflows/evil.yml'; }],
    ['wrong workflow name', f => { f.run.name = 'evil'; }],
    ['workflow ID mismatch', f => { f.workflow = {name: f.run.name, path: 'evil'}; }],
    ['stale attempt', f => { f.run.run_attempt = 2; }],
    ['in-progress rerun', f => { f.run.status = 'in_progress'; }],
    ['stale PR commit', f => { f.pr.head.sha = 'b'.repeat(40); }],
    ['wrong fork', f => { f.pr.head.repo.id = 3; }],
    ['deleted fork', f => { f.pr.head.repo = null; }],
    ['wrong PR base', f => { f.pr.base.repo.id = 3; }],
    ['closed PR', f => { f.pr.state = 'closed'; }],
    ['invalid SHA', f => { f.run.head_sha = 'bad'; }],
    ['newer workflow run', f => { f.runs.push({...f.run, id: 101, run_number: 11}); }],
    ['newer comment', f => { f.comments = [ownComment(
        '<!-- openvino-bom:ubuntu_22_04_x86_64 -->\n<!-- bom-run:11:1 -->')]; }],
]) {
    test(`ignores ${label}`, async () => {
        const f = fixture();
        change(f);
        await f.execute();
        assert.equal(f.writes.length, 0);
    });
}

test('fork fallback uses associated PR lookup and still checks live head and base', async () => {
    const f = fixture();
    f.run.pull_requests = [];
    await f.execute();
    assert.equal(f.writes.length, 1);
    assert.ok(f.calls.some(call => call.name === 'associated'));
    f.pr.head.repo.id = 99;
    f.writes = [];
    await f.execute();
    assert.equal(f.writes.length, 0);
});

test('latest job per name wins over a prior failed job and its annotations', async () => {
    const f = fixture();
    f.jobs.push({...f.jobs[0], id: 201, conclusion: 'success',
        steps: [{name: 'Compare BOM against GOLD', conclusion: 'success'}]});
    await f.execute();
    assert.equal(f.writes.length, 0);
    assert.equal(f.calls.find(call => call.name === 'jobs').args.filter, 'latest');
    assert.ok(!f.calls.some(call => call.name === 'annotations'));
});

test('does not edit user or another bot comments containing the marker', async () => {
    const f = fixture();
    const marker = '<!-- openvino-bom:ubuntu_22_04_x86_64 -->\n';
    f.comments = [
        {id: 1, user: {type: 'User', login: 'github-actions[bot]'}, body: marker},
        {id: 2, user: {type: 'Bot', login: 'another[bot]'}, body: marker},
    ];
    await f.execute();
    assert.equal(f.writes[0].kind, 'create');
});

test('PR update during API reads prevents the write', async () => {
    const f = fixture();
    let reads = 0;
    const pr = f.pr;
    Object.defineProperty(f, 'pr', {get: () => {
        reads++;
        return reads > 1 ? {...pr, head: {...pr.head, sha: 'b'.repeat(40)}} : pr;
    }});
    await f.execute();
    assert.equal(f.writes.length, 0);
});

for (const url of [
    'https://evil.example/repos/owner/repo/check-runs/300',
    'https://api.github.com/repos/evil/repo/check-runs/300',
    'https://api.github.com/repos/owner/repo/check-runs/300?evil=1',
    'https://api.github.com/repos/owner/repo/check-runs/evil',
]) {
    test(`rejects unexpected check URL: ${url}`, async () => {
        const f = fixture();
        f.jobs[0].check_run_url = url;
        await assert.rejects(f.execute(), /unexpected check-run URL/);
        assert.equal(f.writes.length, 0);
    });
}

test('API errors are propagated, not silently reported as no differences', async () => {
    const f = fixture();
    Object.defineProperty(f, 'annotations', {get: () => { throw new Error('403 API failure'); }});
    await assert.rejects(f.execute(), /403 API failure/);
});

test('consumes fully paginated annotation, job, PR-association, and comment results', async () => {
    const f = fixture();
    f.run.pull_requests = [];
    f.associated = Array.from({length: 100}, () => ({number: 0})).concat(f.pr);
    f.annotations = Array.from({length: 100}, () => ({title: 'Other'})).concat(f.annotations);
    f.jobs = Array.from({length: 100}, (_, i) => ({id: i, name: `Other ${i}`})).concat(f.jobs);
    await f.execute();
    const existing = ownComment(f.writes[0].body);
    f.comments = Array.from({length: 100}, (_, i) => ({id: i, body: 'Other'})).concat(existing);
    f.writes = [];
    f.run.run_attempt = 2;
    f.context.payload.workflow_run.run_attempt = 2;
    await f.execute();
    assert.equal(f.writes.length, 1);
    assert.equal(f.writes[0].kind, 'update');
});

for (const [file, [name, configuration]] of Object.entries(notify.PLATFORMS)) {
    test(`notifies the allowlisted ${configuration} platform`, async () => {
        const f = fixture();
        f.run.path = `.github/workflows/${file}`;
        f.run.name = name;
        f.context.payload.workflow_run = structuredClone(f.run);
        await f.execute();
        assert.equal(f.writes.length, 1);
        assert.ok(f.writes[0].body.includes(`<!-- openvino-bom:${configuration} -->`));
    });
}

test('all ten allowlisted platform mappings match callers and workflow trigger names', () => {
    const root = path.resolve(__dirname, '..', '..', '..');
    const notifier = fs.readFileSync(path.join(root, '.github', 'workflows', 'bom_pr_comment.yml'), 'utf8');
    assert.equal(Object.keys(notify.PLATFORMS).length, 10);
    for (const [file, [name, configuration]] of Object.entries(notify.PLATFORMS)) {
        const caller = fs.readFileSync(path.join(root, '.github', 'workflows', file), 'utf8');
        assert.ok(caller.startsWith(`name: ${name}\n`) || caller.startsWith(`name: ${name}\r\n`), file);
        assert.ok(caller.includes(`artifact-name: bom_${configuration}`), file);
        assert.ok(notifier.includes(`- '${name}'`), file);
    }
});
