
async function main() {
  const res = await fetch('./papers.json');
  const data = await res.json();
  const papers = data.papers;
  const state = { q: '', category: '', year: '', platform: '', model: '', status: '', publisher: '', venue: '' };

  fillSelect('year', data.filters.years);
  fillSelect('platform', data.filters.platforms);
  fillSelect('model', data.filters.models);
  fillSelect('publisher', data.filters.publishers);
  fillSelect('venue', data.filters.venues);
  document.getElementById('total-count').textContent = String(data.count);

  const bind = (id, key) => {
    const el = document.getElementById(id);
    el.addEventListener('input', () => { state[key] = el.value.trim(); render(); });
    el.addEventListener('change', () => { state[key] = el.value.trim(); render(); });
  };
  bind('q', 'q');
  bind('year', 'year');
  bind('platform', 'platform');
  bind('model', 'model');
  bind('publisher', 'publisher');
  bind('venue', 'venue');

  document.querySelectorAll('[data-filter-chip]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const [key, val] = btn.dataset.filterChip.split(':');
      const group = key === 'status' ? 'status' : 'category';
      state[group] = state[group] === val ? '' : val;
      document.querySelectorAll(`[data-filter-chip^="${group}:"]`).forEach((b) => {
        b.classList.toggle('active', b.dataset.filterChip === `${group}:${state[group]}`);
      });
      render();
    });
  });

  document.getElementById('reset').addEventListener('click', () => {
    Object.keys(state).forEach((k) => state[k] = '');
    ['q','year','platform','model','publisher','venue'].forEach((id) => { document.getElementById(id).value = ''; });
    document.querySelectorAll('[data-filter-chip]').forEach((b) => b.classList.remove('active'));
    render();
  });

  function fillSelect(id, values) {
    const el = document.getElementById(id);
    (values || []).forEach((v) => {
      const opt = document.createElement('option');
      opt.value = v; opt.textContent = v; el.appendChild(opt);
    });
  }

  function matches(p) {
    if (state.category && p.category !== state.category) return false;
    if (state.year && String(p.year) !== state.year) return false;
    if (state.platform && p.platform !== state.platform) return false;
    if (state.model && !(p.model || []).includes(state.model)) return false;
    if (state.status && p.status !== state.status) return false;
    if (state.publisher && p.publisher !== state.publisher) return false;
    if (state.venue && p.venue_short !== state.venue) return false;
    if (state.q) {
      const hay = [
        p.title, p.authors_text, p.venue_full, p.venue_short, p.doi_text, p.doi, p.publisher,
        p.status_label, ...(p.keywords_paper || []), ...(p.keywords_meta || []), ...(p.model || [])
      ].join(' ').toLowerCase();
      if (!hay.includes(state.q.toLowerCase())) return false;
    }
    return true;
  }

  function escapeHtml(s) {
    return String(s ?? '')
      .replaceAll('&', '&amp;').replaceAll('<', '&lt;')
      .replaceAll('>', '&gt;').replaceAll('"', '&quot;');
  }

  function tags(list, cls='') {
    return (list || []).map((t) => `<span class="tag ${cls}">${escapeHtml(t)}</span>`).join('');
  }

  function splitBadge(left, right, cls='') {
    const L = String(left || '').trim();
    const R = String(right || '').trim();
    if (!L && !R) return '';
    if (!R) return `<span class="mchip">${escapeHtml(L)}</span>`;
    if (!L) return `<span class="mchip">${escapeHtml(R)}</span>`;
    return `<span class="split-badge ${cls}"><span class="sb-left">${escapeHtml(L)}</span><span class="sb-right">${escapeHtml(R)}</span></span>`;
  }

  function formatAuthors(list) {
    const names = (list || []).map((n) => String(n || '').trim()).filter(Boolean);
    if (!names.length) return '';
    if (names.length <= 6) return names.join(', ');
    return `${names.slice(0, 5).join(', ')}, et al.`;
  }

  function cardHtml(p, i) {
    const href = p.url || p.doi || '#';
    const doiHtml = p.doi
      ? `<div class="doi-row"><a href="${escapeHtml(p.doi)}" target="_blank" rel="noopener">DOI</a><span class="doi-text">${escapeHtml(p.doi_text)}</span></div>`
      : '';
    const dl = p.download
      ? `<a class="download" href="${escapeHtml(p.download)}" target="_blank" rel="noopener">Download</a>`
      : `<span class="download disabled">No PDF</span>`;
    const authorsText = formatAuthors(p.authors);
    const authorsHtml = authorsText
      ? `<p class="authors">${escapeHtml(authorsText)}</p>`
      : '';
    const shortNeeded = p.venue_short && p.venue_short.toLowerCase() !== String(p.publisher || '').toLowerCase();
    const pubBadge = splitBadge(p.publisher, p.year);
    const delay = Math.min(i, 12) * 28;
    return `<article class="card" style="animation-delay:${delay}ms">
      <h3><a href="${escapeHtml(href)}" target="_blank" rel="noopener">${escapeHtml(p.title)}</a></h3>
      ${authorsHtml}
      <p class="venue-full">${escapeHtml(p.venue_full)}</p>
      <div class="meta-line">
        <span class="mchip cat" style="--cat:${escapeHtml(p.category_color)}">${escapeHtml(p.category_label)}</span>
        ${shortNeeded ? `<span class="mchip">${escapeHtml(p.venue_short)}</span>` : ''}
        ${pubBadge}
      </div>
      ${doiHtml}
      <div class="kw">
        <div><strong>Keywords</strong><div class="kw-row">${tags(p.keywords_paper) || '<span class="tag">—</span>'}</div></div>
        <div><strong>Meta</strong><div class="kw-row">${tags(p.keywords_meta, 'meta') || '<span class="tag">—</span>'}</div></div>
      </div>
      <div class="card-foot">
        <span class="status" style="--st:${escapeHtml(p.status_color)}"><i></i>${escapeHtml(p.status_label)}</span>
        ${dl}
      </div>
    </article>`;
  }

  function render() {
    const filtered = papers.filter(matches);
    document.getElementById('visible-count').textContent = String(filtered.length);
    document.getElementById('empty').classList.toggle('hidden', filtered.length > 0);
    const byYear = new Map();
    filtered.forEach((p) => {
      const y = String(p.year || 'Unknown');
      if (!byYear.has(y)) byYear.set(y, []);
      byYear.get(y).push(p);
    });
    document.getElementById('year-groups').innerHTML = [...byYear.entries()].map(([year, list]) => `
      <section class="year-block">
        <h2>${escapeHtml(year)} <span style="font-weight:500;font-size:.85rem;opacity:.7">(${list.length})</span></h2>
        <div class="cards">${list.map((p, i) => cardHtml(p, i)).join('')}</div>
      </section>
    `).join('');
  }

  render();
}

main().catch((err) => {
  document.getElementById('year-groups').innerHTML = `<p class="empty">Failed to load papers.json: ${err}</p>`;
});
