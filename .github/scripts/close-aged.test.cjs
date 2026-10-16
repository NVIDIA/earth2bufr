// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

const assert = require('node:assert/strict');
const { test } = require('node:test');
const closeAged = require('./close-aged.cjs');

const now = Date.parse('2026-10-01T12:00:00Z');
const day = 24 * 60 * 60 * 1000;

function item(number, age, extra = {}) {
  return {
    number, state: 'open', labels: [],
    created_at: new Date(now - age * day).toISOString(),
    updated_at: new Date(now).toISOString(),
    ...extra,
  };
}

function client(items, refreshed = {}) {
  const events = [];
  let listed = false;
  const github = {
    rest: {
      issues: {
        listForRepo: async () => {},
        get: async ({ issue_number }) => ({
          data: refreshed[issue_number] || items.find(i => i.number === issue_number),
        }),
        update: async args => {
          assert.ok(listed, 'finish pagination before closing any items');
          events.push(['issue', args]);
        },
        createComment: async args => events.push(['comment', args]),
      },
      pulls: { update: async args => events.push(['pr', args]) },
    },
    paginate: async (endpoint, args) => {
      assert.equal(endpoint, github.rest.issues.listForRepo);
      assert.deepEqual(args, { owner: 'org', repo: 'repo', state: 'open', per_page: 100 });
      listed = true;
      return items;
    },
  };
  return { github, events, context: { repo: { owner: 'org', repo: 'repo' } },
    core: { info() {} }, now };
}

test('issues close strictly after 21 days, even with recent activity', async () => {
  const api = client([item(1, 21), item(2, 21.001), item(3, 20)]);
  await closeAged(api);
  assert.deepEqual(api.events.map(([kind, args]) => [kind, args.issue_number]),
    [['issue', 2], ['comment', 2]]);
  assert.equal(api.events[0][1].state, 'closed');
  assert.equal(api.events[0][1].state_reason, 'not_planned');
  assert.match(api.events[1][1].body, /21 days/);
});

test('confirmed issues are exempt, regardless of label case', async () => {
  const api = client([item(1, 100, { labels: [{ name: 'confirmed' }] }),
    item(2, 100, { labels: [{ name: 'Confirmed' }] })]);
  await closeAged(api);
  assert.deepEqual(api.events, []);
});

test('PRs close strictly after 14 days including confirmed and draft PRs', async () => {
  const pr = { pull_request: { url: 'https://api.github.com/repos/org/repo/pulls/1' } };
  const api = client([item(1, 14, pr), item(2, 14.001, pr),
    item(3, 15, { ...pr, draft: true, labels: [{ name: 'confirmed' }] })]);
  await closeAged(api);
  assert.deepEqual(api.events.filter(([kind]) => kind === 'pr').map(([, args]) => args.pull_number), [2, 3]);
  assert.ok(api.events.filter(([kind]) => kind === 'pr').every(([, args]) => args.state === 'closed'));
  assert.match(api.events.find(([kind]) => kind === 'comment')[1].body, /14 days/);
});

test('rechecks labels and state before closing candidates', async () => {
  const api = client([item(1, 30), item(2, 30), item(3, 30)], {
    1: item(1, 30, { labels: [{ name: 'confirmed' }] }),
    2: item(2, 30, { state: 'closed' }),
  });
  await closeAged(api);
  assert.deepEqual(api.events.filter(([kind]) => kind === 'issue').map(([, args]) => args.issue_number), [3]);
});

test('processes the full paginated result, including more than 100 candidates', async () => {
  const api = client(Array.from({ length: 105 }, (_, i) => item(i + 1, 30)));
  await closeAged(api);
  assert.equal(api.events.filter(([kind]) => kind === 'issue').length, 105);
});

test('API errors fail the run and do not post a false closure comment', async () => {
  const api = client([item(1, 30)]);
  api.github.rest.issues.update = async () => { throw new Error('API unavailable'); };
  await assert.rejects(closeAged(api), /API unavailable/);
  assert.deepEqual(api.events, []);
});
