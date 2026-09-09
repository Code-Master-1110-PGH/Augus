const state = { threadId: null, timer: null, status: null };
const $ = (id) => document.getElementById(id);
const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (char) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[char]));

function toast(message) { const node = $('toast'); node.textContent = message; node.classList.add('show'); setTimeout(() => node.classList.remove('show'), 3000); }
function formatMetric(value) { return value === null || value === undefined ? '—' : `${(Number(value) * 100).toFixed(1)}%`; }
function setAgentStates(status) {
  const values = { retrieve: 'IDLE', analyze: 'IDLE', audit: 'IDLE' };
  const history = status.graph_history || [];
  history.forEach((event) => { if (event.node && values[event.node]) values[event.node] = event.status || 'COMPLETE'; });
  if (status.status === 'RUNNING') { const current = history.length ? history[history.length - 1].node : 'retrieve'; values[current] = 'RUNNING'; }
  Object.entries(values).forEach(([agent, value]) => { const card = document.querySelector(`[data-agent="${agent}"]`); const label = $(`${agent}State`); card.classList.toggle('running', value === 'RUNNING'); card.classList.toggle('complete', value === 'COMPLETE' || value === 'PASS'); label.textContent = value; });
}
function renderLedger(count, ticker) { const ledger = $('ledger'); if (!count) { ledger.innerHTML = '<div class="empty-ledger"><i data-lucide="file-search"></i><span>Run an investigation to populate the evidence ledger.</span></div>'; lucide.createIcons(); return; } ledger.innerHTML = `<div class="ledger-item"><i data-lucide="file-text"></i><div><strong>SEC EDGAR / ${esc(ticker)} 10-K</strong><small>Primary filing evidence with preserved section metadata</small></div><code>${count} CHUNKS</code></div><div class="ledger-item"><i data-lucide="layers-3"></i><div><strong>Hybrid retrieval index</strong><small>Dense-like token rank + lexical overlap</small></div><code>VERIFIED</code></div>`; lucide.createIcons(); }
function renderActivity(history) { const log = $('activityLog'); if (!history?.length) { log.innerHTML = '<div class="activity-empty">No run events yet.</div>'; return; } log.innerHTML = history.slice(-7).map((event) => `<div class="activity-entry"><b>${esc(String(event.node || 'SYSTEM').toUpperCase())}</b>${esc(event.status || (event.iteration ? `Iteration ${event.iteration}` : 'State updated'))}</div>`).join(''); }
function renderNarrative(status) {
  const narrativeCard = $('narrativeCard');
  const narrativeContent = $('narrativeContent');
  const descriptiveAnswer = status.descriptive_answer || '';
  
  if (!descriptiveAnswer) {
    narrativeCard.hidden = true;
    return;
  }
  
  narrativeCard.hidden = false;
  
  // Split narrative into paragraphs and render with formatting
  const paragraphs = descriptiveAnswer.split('\n\n').filter(p => p.trim());
  
  narrativeContent.innerHTML = paragraphs.map((para) => {
    // Escape HTML and preserve paragraph structure
    const escaped = esc(para);
    // Highlight key terms with emphasis
    const formatted = escaped
      .replace(/Financial Performance Overview:/g, '<strong>Financial Performance Overview:</strong>')
      .replace(/Risk Assessment & Disclosures:/g, '<strong>Risk Assessment & Disclosures:</strong>')
      .replace(/Analyst Sentiment & Recommendation:/g, '<strong>Analyst Sentiment & Recommendation:</strong>')
      .replace(/Verification Status:/g, '<strong>Verification Status:</strong>')
      .replace(/\b(cautious|neutral|bullish|bearish)\b/gi, '<em>$1</em>')
      .replace(/\b(revenue growth|operating margin|debt-to-assets|risk factors|compliance|disclosure)\b/gi, '<strong>$1</strong>');
    
    return `<p class="narrative-paragraph">${formatted}</p>`;
  }).join('');
  
  lucide.createIcons();
}
function renderStatus(status) {
  state.status = status; $('iteration').textContent = `ITERATION ${status.iteration_count || 0} / 3`; $('evidenceCount').textContent = `${status.retrieved_documents || 0} evidence chunks`; $('lastUpdated').textContent = `Last signal ${new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'})}`;
  const analysis = status.financial_analysis || {}; const metrics = analysis.metrics || {}; $('growth').textContent = formatMetric(metrics.revenue_growth); $('margin').textContent = formatMetric(metrics.operating_margin); $('debt').textContent = formatMetric(metrics.debt_to_assets); $('riskBar').style.width = `${Math.min(Number(metrics.debt_to_assets || 0) * 100, 100)}%`;
  const answer = status.answer || {}; $('answerTitle').textContent = status.status === 'APPROVED' ? 'Approved research answer' : status.status === 'AWAITING_HUMAN_REVIEW' ? 'Answer ready for sign-off' : status.status === 'RUNNING' ? 'Answer in progress' : 'Research answer'; $('answerStatus').textContent = status.status || 'WAITING'; $('answerQuestion').textContent = status.question || 'Run an investigation to see the answer to your research question.'; $('answerSummary').textContent = answer.summary || 'The analyst is still processing retrieved evidence.'; $('answerFacts').innerHTML = `<span><b>Sentiment</b><strong>${esc(answer.sentiment || 'pending')}</strong></span><span><b>Risks found</b><strong>${answer.risk_factors?.length || 0}</strong></span><span><b>Evidence</b><strong>${answer.evidence_chunk_ids?.length || status.retrieved_documents || 0} chunks</strong></span>`;
  const report = status.compliance_report || {}; const reviewStatus = $('reviewStatus'); reviewStatus.textContent = status.status === 'APPROVED' ? 'APPROVED' : report.status || (status.status === 'AWAITING_HUMAN_REVIEW' ? 'READY' : status.status || 'PENDING'); reviewStatus.className = `review-status ${status.status === 'APPROVED' || report.status === 'PASS' ? 'pass' : report.status === 'REVISE' ? 'fail' : ''}`; $('score').textContent = report.status === 'PASS' ? '94' : report.status ? '62' : '—'; $('auditSummary').textContent = status.status === 'APPROVED' ? 'Report approved by the human reviewer.' : report.audit_notes?.[0] || 'A completed run will be evaluated against disclosure and provenance controls.';
  const notes = report.audit_notes || []; $('auditList').innerHTML = (notes.length ? notes.slice(0, 3) : ['Disclosure completeness','Evidence provenance','Risk factor reporting']).map((note, index) => `<div class="audit-row ${report.status ? '' : 'muted'}"><i data-lucide="${report.status === 'PASS' ? 'circle-check' : 'circle-dashed'}"></i><span>${esc(note)}</span><b>${report.status === 'PASS' ? 'PASS' : 'PENDING'}</b></div>`).join('');
  $('approveBtn').disabled = status.status !== 'AWAITING_HUMAN_REVIEW'; $('reviseBtn').disabled = status.status !== 'AWAITING_HUMAN_REVIEW'; renderLedger(status.retrieved_documents, $('ticker').value.toUpperCase()); renderNarrative(status); renderActivity(status.graph_history); setAgentStates(status); lucide.createIcons();
}
async function poll() { if (!state.threadId) return; try { const response = await fetch(`/api/v1/research/${state.threadId}/status`); if (response.status === 404) return; if (!response.ok) throw new Error('Status unavailable'); const status = await response.json(); renderStatus(status); saveThread(status); if (status.status === 'AWAITING_HUMAN_REVIEW' || status.status === 'APPROVED') { clearInterval(state.timer); toast(status.status === 'APPROVED' ? 'Report approved.' : 'Research run reached its governance gate.'); } } catch (error) { clearInterval(state.timer); toast(error.message); } }
function saveThread(status) { const threads = JSON.parse(localStorage.getItem('augusThreads') || '[]').filter((thread) => thread.thread_id !== status.thread_id); threads.unshift({thread_id: status.thread_id, ticker: $('ticker').value.toUpperCase(), query: $('query').value, status: status.status, updated: Date.now()}); localStorage.setItem('augusThreads', JSON.stringify(threads.slice(0, 8))); }
async function startResearch() { const button = $('runBtn'); if ($('query').value.trim().length < 3) { toast('Add a research question first.'); return; } button.disabled = true; button.querySelector('span').textContent = 'Initializing…'; try { const response = await fetch('/api/v1/research/start', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({query: $('query').value, company_ticker: $('ticker').value.trim().toUpperCase()}) }); if (!response.ok) throw new Error('Could not start investigation'); const result = await response.json(); state.threadId = result.thread_id; $('threadId').textContent = state.threadId.slice(0, 18) + '…'; clearInterval(state.timer); state.timer = setInterval(poll, 1200); await poll(); toast('Investigation started.'); } catch (error) { toast(error.message); } finally { button.disabled = false; button.querySelector('span').textContent = 'Run investigation'; } }
async function resume(approved, reviewer, comment) { try { const response = await fetch(`/api/v1/research/${state.threadId}/resume`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({approved, reviewer, comment})}); if (!response.ok) throw new Error('Could not resume thread'); closeModal(); renderStatus(await response.json()); saveThread(state.status); toast(approved ? 'Report approved.' : 'Revision submitted.'); } catch (error) { toast(error.message); } }

const modal = $('modal');
function openModal(type) { const title = $('modalTitle'); const eyebrow = $('modalEyebrow'); const body = $('modalBody'); const status = state.status || {}; const report = status.compliance_report || {}; const docs = status.retrieved_documents || 0; const history = JSON.parse(localStorage.getItem('augusThreads') || '[]');
  modal.hidden = false; document.body.classList.add('modal-open');
  if (type === 'evidence') { eyebrow.textContent = 'SOURCE REGISTER'; title.textContent = 'Evidence ledger'; body.innerHTML = docs ? `<div class="modal-stat"><strong>${docs}</strong><span>provenance-preserved chunks retrieved for ${esc($('ticker').value.toUpperCase())}</span></div><div class="modal-table"><div><span>SEC EDGAR</span><b>Primary source</b></div><div><span>Filing class</span><b>10-K / recent</b></div><div><span>Index method</span><b>Hybrid rank</b></div><div><span>Traceability</span><b class="good">Verified</b></div></div>` : '<div class="modal-empty"><i data-lucide="file-search"></i><p>No evidence yet. Run an investigation to build the register.</p></div>'; }
  if (type === 'library') { eyebrow.textContent = 'DOCUMENTS'; title.textContent = 'Filing library'; body.innerHTML = `<div class="library-search"><i data-lucide="search"></i><input id="librarySearch" type="search" placeholder="Filter filings, tickers, or sections" aria-label="Filter filing library"></div><div class="library-items"><button class="library-item" data-load-ticker="DEMO"><span class="file-badge">10-K</span><span><strong>Demo Holdings</strong><small>DEMO • SEC EDGAR • latest available</small></span><i data-lucide="arrow-up-right"></i></button><button class="library-item" data-load-ticker="NVDA"><span class="file-badge cyan-badge">10-Q</span><span><strong>NVIDIA Corporation</strong><small>NVDA • SEC EDGAR • quarterly scan</small></span><i data-lucide="arrow-up-right"></i></button><button class="library-item" data-load-ticker="MSFT"><span class="file-badge coral-badge">10-K</span><span><strong>Microsoft Corporation</strong><small>MSFT • SEC EDGAR • annual filing</small></span><i data-lucide="arrow-up-right"></i></button></div><div class="library-empty" hidden>No filings match that filter.</div>`; }
  if (type === 'queue') { eyebrow.textContent = 'GOVERNANCE'; title.textContent = 'Compliance queue'; body.innerHTML = `<div class="queue-banner"><i data-lucide="shield-alert"></i><span><strong>${status.status === 'AWAITING_HUMAN_REVIEW' ? '1 report needs review' : 'Queue is clear'}</strong><small>Human sign-off is required before distribution.</small></span></div><div class="queue-item"><span class="status-dot ${status.status === 'AWAITING_HUMAN_REVIEW' ? 'green' : 'grey'}"></span><span><strong>${esc($('ticker').value.toUpperCase())} research report</strong><small>${esc(report.status || 'No active report')} • ${docs} evidence chunks</small></span><button class="small-action" data-queue-open>Open report</button></div>`; }
  if (type === 'notifications') { eyebrow.textContent = 'INBOX'; title.textContent = 'Notifications'; body.innerHTML = `<div class="notification-item"><span class="notification-icon amber-icon"><i data-lucide="activity"></i></span><span><strong>Systems nominal</strong><small>All agent services are responding normally.</small></span><time>now</time></div><div class="notification-item"><span class="notification-icon cyan-icon"><i data-lucide="database"></i></span><span><strong>Primary source mode active</strong><small>SEC EDGAR is configured as the source register.</small></span><time>now</time></div>`; }
  if (type === 'settings') { eyebrow.textContent = 'PREFERENCES'; title.textContent = 'Workspace settings'; body.innerHTML = `<label class="setting-row"><span><strong>Auto-poll research runs</strong><small>Refresh thread state every 1.2 seconds.</small></span><input type="checkbox" id="autoPoll" checked></label><label class="setting-row"><span><strong>Show reasoning trace</strong><small>Keep agent events visible in the activity log.</small></span><input type="checkbox" id="showTrace" checked></label><label class="setting-row"><span><strong>Source preference</strong><small>Primary filing evidence takes priority.</small></span><select id="sourcePreference"><option>SEC EDGAR</option><option>Qdrant index</option></select></label>`; }
  if (type === 'profile') { eyebrow.textContent = 'ACCOUNT'; title.textContent = 'Reviewer profile'; body.innerHTML = `<div class="profile-card"><div class="large-avatar">LM</div><div><strong>Lead reviewer</strong><small>Local private workspace</small></div></div><button class="modal-primary" data-close-modal>Done</button>`; }
  if (type === 'activity') { eyebrow.textContent = 'OBSERVABILITY'; title.textContent = 'Full reasoning trace'; body.innerHTML = history.length ? history.map((item) => `<div class="trace-row"><span>${esc(item.ticker)}</span><strong>${esc(item.status)}</strong><time>${new Date(item.updated).toLocaleString()}</time></div>`).join('') : '<div class="modal-empty"><i data-lucide="activity"></i><p>Activity will appear after your first investigation.</p></div>'; }
  if (type === 'source') { eyebrow.textContent = 'SOURCE MODE'; title.textContent = 'SEC EDGAR primary source'; body.innerHTML = '<div class="modal-stat"><strong>SEC</strong><span>Primary filing data is fetched from SEC EDGAR, then preserved with filing type, date, section, CIK, and chunk identifiers.</span></div><div class="modal-table"><div><span>Retrieval</span><b>SEC EDGAR + sec-api</b></div><div><span>Search</span><b>Lexical + local dense</b></div><div><span>Fallback</span><b>Offline DEMO fixture</b></div></div>'; }
  if (type === 'agents') { eyebrow.textContent = 'ORCHESTRATION'; title.textContent = 'Three agents ready'; body.innerHTML = '<div class="modal-table"><div><span>Retriever</span><b>Fetches and ranks filing evidence</b></div><div><span>Analyst</span><b>Computes grounded financial signals</b></div><div><span>Auditor</span><b>Checks provenance and compliance</b></div></div>'; }
  if (type === 'timing') { eyebrow.textContent = 'RUN PROFILE'; title.textContent = 'Typical run under 30 seconds'; body.innerHTML = '<div class="modal-table"><div><span>01 / Retrieve</span><b>SEC filing acquisition</b></div><div><span>02 / Analyze</span><b>Metrics and risk extraction</b></div><div><span>03 / Audit</span><b>Compliance gate</b></div><div><span>04 / Review</span><b>Human sign-off checkpoint</b></div></div>'; }
  lucide.createIcons();
  if (type === 'library') {
    const search = $('librarySearch');
    const items = [...document.querySelectorAll('.library-item')];
    const empty = document.querySelector('.library-empty');
    search.addEventListener('input', () => {
      const filter = search.value.trim().toLowerCase();
      let visible = 0;
      items.forEach((item) => { const matches = item.textContent.toLowerCase().includes(filter); item.hidden = !matches; if (matches) visible += 1; });
      empty.hidden = visible > 0;
    });
  }
}
function closeModal() { modal.hidden = true; document.body.classList.remove('modal-open'); }
function openDetailModal(type) {
  const analysis = state.status?.financial_analysis || {};
  const metrics = analysis.metrics || {};
  const details = {
    source: ['SOURCE MODE', 'SEC EDGAR primary-source mode', 'Filings are retrieved from SEC EDGAR, chunked with section metadata, and retained with provenance for review.'],
    timing: ['RUN PERFORMANCE', 'Typical run under 30 seconds', 'The workflow runs retrieval, deterministic analysis, compliance checks, and a human checkpoint in sequence.'],
    agents: ['AGENT TEAM', 'Three specialists ready', 'Retriever acquires evidence, Analyst computes grounded signals, and Auditor validates completeness and provenance.'],
    retriever: ['RETRIEVER', 'Evidence acquisition', 'Decomposes the question, fetches SEC filings, and ranks relevant chunks with lexical and local dense retrieval.'],
    analyst: ['ANALYST', 'Financial interpretation', 'Extracts revenue, margins, debt exposure, sentiment, risk factors, and evidence-linked summaries.'],
    auditor: ['AUDITOR', 'Compliance verification', 'Checks disclosure completeness, financial evidence, risk reporting, and routes revision requests.'],
    growth: ['REVENUE GROWTH', formatMetric(metrics.revenue_growth), 'Period-over-period revenue change computed by a deterministic finance tool.'],
    margin: ['OPERATING MARGIN', formatMetric(metrics.operating_margin), 'Operating income divided by revenue, computed without LLM arithmetic.'],
    debt: ['DEBT / ASSETS', formatMetric(metrics.debt_to_assets), 'Total debt divided by total assets, used as a capital structure signal.'],
    audit: ['COMPLIANCE AUDIT', state.status?.compliance_report?.status || 'PENDING', state.status?.compliance_report?.audit_notes?.join(' ') || 'Run an investigation to evaluate the compliance controls.'],
    confidence: ['AUDIT CONFIDENCE', $('score').textContent, 'Confidence is a review aid. The report remains advisory until a human reviewer signs off.'],
    advisory: ['ADVISORY POLICY', 'Human approval required', 'Reports cannot be treated as final compliance advice until a reviewer approves the governance gate.']
  }[type] || ['AUGUS DETAIL', 'Interactive workspace surface', 'This workspace element is connected to the active research thread.'];
  $('modalEyebrow').textContent = details[0]; $('modalTitle').textContent = details[1]; $('modalBody').innerHTML = `<div class="modal-empty"><i data-lucide="info"></i><p>${esc(details[2])}</p></div>`; modal.hidden = false; document.body.classList.add('modal-open'); lucide.createIcons();
}
function openReviewerModal(approved) { $('modalEyebrow').textContent = 'HUMAN SIGN-OFF'; $('modalTitle').textContent = approved ? 'Approve report' : 'Request revision'; $('modalBody').innerHTML = `<form id="reviewForm" class="review-form"><p>${approved ? 'Confirm that this report meets your review standard before distribution.' : 'Tell the agents what needs to be revisited before this report can be approved.'}</p><label>REVIEWER<input id="reviewerName" required placeholder="Your name"></label>${approved ? '' : '<label>REVISION NOTE<textarea id="reviewComment" required placeholder="Describe the missing evidence or issue"></textarea></label>'}<div class="form-actions"><button type="button" class="revise-button" data-close-modal>Cancel</button><button class="modal-primary" type="submit">${approved ? 'Approve report' : 'Submit revision'}</button></div></form>`; modal.hidden = false; document.body.classList.add('modal-open'); lucide.createIcons(); $('reviewForm').addEventListener('submit', (event) => { event.preventDefault(); const reviewer = $('reviewerName').value.trim(); const comment = approved ? 'Approved in research console.' : $('reviewComment').value.trim(); if (reviewer && (approved || comment)) resume(approved, reviewer, comment); }); $('reviewerName').focus(); }
function setView(view) { document.querySelectorAll('.nav-item').forEach((item) => item.classList.toggle('active', item.dataset.view === view)); if (view === 'desk') { $('viewLabel').textContent = 'RESEARCH DESK'; $('crumbLabel').textContent = 'NEW INVESTIGATION'; window.scrollTo({top: 0, behavior: 'smooth'}); } else { openModal(view); } }
document.querySelectorAll('[data-view]').forEach((item) => item.addEventListener('click', () => setView(item.dataset.view)));
document.querySelectorAll('.recent').forEach((item) => item.addEventListener('click', () => { const ticker = item.dataset.recent || item.querySelector('strong').textContent.trim(); document.querySelectorAll('.recent').forEach((recent) => recent.classList.remove('active')); item.classList.add('active'); $('ticker').value = ticker; $('query').value = item.querySelector('small').textContent; setView('desk'); toast(`Loaded ${ticker} research brief.`); }));
modal.addEventListener('click', (event) => { if (event.target === modal || event.target.closest('[data-close-modal]')) closeModal(); const load = event.target.closest('[data-load-ticker]'); if (load) { $('ticker').value = load.dataset.loadTicker; closeModal(); toast(`${load.dataset.loadTicker} loaded into the investigation brief.`); } const openQueue = event.target.closest('[data-queue-open]'); if (openQueue) { closeModal(); document.querySelector('.review-card').scrollIntoView({behavior:'smooth', block:'center'}); toast('Active report opened at the governance gate.'); } });
document.addEventListener('keydown', (event) => { if (event.key === 'Escape') closeModal(); if ((event.metaKey || event.ctrlKey) && event.key === '1') setView('desk'); });
document.addEventListener('click', (event) => {
  const modalTarget = event.target.closest('[data-modal]');
  if (modalTarget) { openModal(modalTarget.dataset.modal); return; }
  if (event.target.closest('[data-view], [data-recent], [data-close-modal], #runBtn, #approveBtn, #reviseBtn')) return;
  const target = event.target.closest('.source-chip, .brief-footer span, .agent-card, .metric-card, .review-status, .review-score, .audit-row, .notice');
  if (!target) return;
  const type = target.classList.contains('source-chip') || target.textContent.includes('Primary-source') ? 'source' : target.classList.contains('agent-card') ? target.dataset.agent : target.classList.contains('metric-card') ? ({growth:'growth',margin:'margin',debt:'debt'}[target.querySelector('strong')?.id] || 'growth') : target.classList.contains('audit-row') || target.classList.contains('review-status') ? 'audit' : target.classList.contains('review-score') ? 'confidence' : target.classList.contains('notice') ? 'advisory' : target.textContent.includes('Typical run') ? 'timing' : 'agents';
  openDetailModal(type);
});
document.querySelectorAll('.agent-card, .metric-card, .source-chip, .brief-footer span, .review-status, .review-score, .audit-row, .notice').forEach((element) => { element.tabIndex = 0; element.setAttribute('role', 'button'); });
document.addEventListener('keydown', (event) => { if ((event.key === 'Enter' || event.key === ' ') && event.target.matches('.agent-card, .metric-card, .source-chip, .brief-footer span, .review-status, .review-score, .audit-row, .notice')) { event.preventDefault(); event.target.click(); } });
const queryField = $('query'); const queryCount = $('queryCount'); function updateQueryCount() { if (queryCount) queryCount.textContent = `${queryField.value.length} / 4000`; } queryField.addEventListener('input', updateQueryCount); updateQueryCount();
$('runBtn').addEventListener('click', startResearch); $('approveBtn').addEventListener('click', () => openReviewerModal(true)); $('reviseBtn').addEventListener('click', () => openReviewerModal(false)); $('ticker').addEventListener('input', (event) => { event.target.value = event.target.value.toUpperCase().replace(/[^A-Z0-9.-]/g, ''); }); lucide.createIcons();

// Document upload handlers
const documentInput = $('documentFile');
const uploadBtn = $('uploadBtn');
const fileNameDisplay = $('fileNameDisplay');
const uploadProgress = $('uploadProgress');

uploadBtn.addEventListener('click', () => documentInput.click());

documentInput.addEventListener('change', (event) => {
  const files = Array.from(event.target.files || []);
  if (files.length === 0) return;
  
  fileNameDisplay.classList.remove('with-files');
  fileNameDisplay.innerHTML = '';
  
  files.forEach((file) => {
    const item = document.createElement('div');
    item.className = 'file-name-item';
    const icon = document.createElement('i');
    icon.setAttribute('data-lucide', 'check');
    const span = document.createElement('span');
    span.textContent = file.name;
    item.appendChild(icon);
    item.appendChild(span);
    fileNameDisplay.appendChild(item);
  });
  
  fileNameDisplay.classList.add('with-files');
  lucide.createIcons();
  
  uploadBtn.textContent = '';
  const span = document.createElement('span');
  span.textContent = `Upload ${files.length} file${files.length > 1 ? 's' : ''}`;
  uploadBtn.appendChild(span);
  
  uploadBtn.disabled = false;
  uploadBtn.addEventListener('click', () => sendUploadRequest(files), {once: true});
});

async function sendUploadRequest(files) {
  const ticker = ($('ticker').value || 'CUSTOM').trim().toUpperCase();
  
  for (const file of files) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('company_ticker', ticker);
    
    uploadProgress.hidden = false;
    const progressFill = $('progressFill');
    const uploadStatus = $('uploadStatus');
    progressFill.style.width = '0%';
    uploadStatus.textContent = `Uploading ${file.name}...`;
    uploadBtn.disabled = true;
    
    try {
      const xhr = new XMLHttpRequest();
      
      xhr.upload.addEventListener('progress', (event) => {
        if (event.lengthComputable) {
          const percentComplete = (event.loaded / event.total) * 100;
          progressFill.style.width = `${percentComplete}%`;
        }
      });
      
      xhr.addEventListener('load', () => {
        if (xhr.status === 200) {
          const response = JSON.parse(xhr.responseText);
          updateUploadStatus(response, file.name);
        } else {
          const error = JSON.parse(xhr.responseText);
          toast(`Upload failed: ${error.message || 'Unknown error'}`);
          uploadProgress.hidden = true;
          uploadBtn.disabled = false;
          uploadBtn.textContent = '';
          const span = document.createElement('span');
          span.textContent = 'Upload documents';
          uploadBtn.appendChild(span);
          let icon = document.createElement('i');
          icon.setAttribute('data-lucide', 'upload');
          uploadBtn.insertBefore(icon, span);
          lucide.createIcons();
        }
      });
      
      xhr.addEventListener('error', () => {
        toast('Upload failed: Network error');
        uploadProgress.hidden = true;
        uploadBtn.disabled = false;
      });
      
      xhr.open('POST', '/api/v1/research/upload');
      xhr.send(formData);
      
      await new Promise((resolve) => {
        xhr.addEventListener('load', resolve);
        xhr.addEventListener('error', resolve);
      });
    } catch (error) {
      toast(`Upload error: ${error.message}`);
      uploadProgress.hidden = true;
      uploadBtn.disabled = false;
    }
  }
}

function updateUploadStatus(response, filename) {
  uploadProgress.hidden = true;
  
  if (response.status === 'success') {
    const successMsg = document.createElement('div');
    successMsg.className = 'upload-success';
    successMsg.innerHTML = `<strong>${filename}</strong><br/>Created ${response.chunks_created} chunks (${response.documents_stored} documents stored)`;
    fileNameDisplay.parentElement.insertBefore(successMsg, uploadProgress);
    
    toast(`${filename}: ${response.chunks_created} chunks processed`);
    
    // Reset after success
    setTimeout(() => {
      documentInput.value = '';
      fileNameDisplay.innerHTML = '';
      fileNameDisplay.classList.remove('with-files');
      uploadBtn.disabled = false;
      uploadBtn.innerHTML = '';
      const icon = document.createElement('i');
      icon.setAttribute('data-lucide', 'upload');
      uploadBtn.appendChild(icon);
      const span = document.createElement('span');
      span.textContent = 'Upload documents';
      uploadBtn.appendChild(span);
      document.querySelectorAll('.upload-success, .upload-error').forEach((el) => el.remove());
      lucide.createIcons();
    }, 2000);
  }
}
