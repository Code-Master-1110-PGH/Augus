(() => {
  const body = document.body;
  const runButton = document.getElementById('runBtn');
  const reviewStatus = document.getElementById('reviewStatus');
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  let stageTimer;
  let stage = 0;

  const setStage = () => {
    document.querySelectorAll('.agent-card').forEach((card, index) => {
      card.classList.toggle('motion-active', index === stage);
    });
    body.dataset.motionStage = String(stage);
    stage = (stage + 1) % 3;
  };

  const stopRunMotion = () => {
    body.classList.remove('investigation-live');
    delete body.dataset.motionStage;
    clearInterval(stageTimer);
    document.querySelectorAll('.agent-card').forEach((card) => card.classList.remove('motion-active'));
  };

  runButton?.addEventListener('click', () => {
    if (reducedMotion.matches) return;
    body.classList.add('investigation-live');
    stage = 0;
    setStage();
    clearInterval(stageTimer);
    stageTimer = setInterval(setStage, 850);
  }, true);

  document.addEventListener('click', (event) => {
    if (reducedMotion.matches) return;
    const button = event.target.closest('button');
    if (!button || button.disabled || button.classList.contains('icon-button')) return;
    const ripple = document.createElement('span');
    ripple.className = 'click-ripple';
    button.appendChild(ripple);
    window.setTimeout(() => ripple.remove(), 520);
  });

  const metricObserver = new MutationObserver((changes) => {
    if (reducedMotion.matches) return;
    changes.forEach((change) => {
      if (change.type === 'childList' && change.target.matches('.metric-card strong')) {
        change.target.classList.remove('metric-value-in');
        void change.target.offsetWidth;
        change.target.classList.add('metric-value-in');
      }
    });
  });
  document.querySelectorAll('.metric-card strong').forEach((metric) => metricObserver.observe(metric, {childList: true}));

  const statusObserver = new MutationObserver(() => {
    const status = reviewStatus?.textContent.trim();
    if (status === 'PASS' || status === 'READY') stopRunMotion();
  });
  if (reviewStatus) statusObserver.observe(reviewStatus, {childList: true, characterData: true, subtree: true});
})();
