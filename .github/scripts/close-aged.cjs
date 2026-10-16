// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

module.exports = async function closeAged({ github, context, core, now = Date.now() }) {
  const day = 24 * 60 * 60 * 1000;
  const eligible = item => {
    const isPr = Boolean(item.pull_request);
    const confirmed = item.labels.some(label => label.name?.toLowerCase() === 'confirmed');
    return item.state === 'open' && (isPr || !confirmed) &&
      Date.parse(item.created_at) < now - (isPr ? 14 : 21) * day;
  };

  // Collect all pages before closing anything so pagination cannot skip items.
  const items = await github.paginate(github.rest.issues.listForRepo, {
    ...context.repo, state: 'open', per_page: 100,
  });
  for (const candidate of items.filter(eligible)) {
    const args = { ...context.repo, issue_number: candidate.number };
    const { data: item } = await github.rest.issues.get(args);
    if (!eligible(item)) continue;

    const isPr = Boolean(item.pull_request);
    if (isPr) {
      await github.rest.pulls.update({
        ...context.repo, pull_number: item.number, state: 'closed',
      });
    } else {
      await github.rest.issues.update({ ...args, state: 'closed', state_reason: 'not_planned' });
    }
    await github.rest.issues.createComment({
      ...args,
      body: isPr
        ? 'Automatically closed because this pull request is more than 14 days old, per repository policy.'
        : 'Automatically closed because this issue is more than 21 days old and does not have the `confirmed` label.',
    });
    core.info(`Closed ${isPr ? 'PR' : 'issue'} #${item.number}`);
  }
};
