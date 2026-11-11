/**
 * RAG-DocAssist UI Controller & State Management
 */
document.addEventListener('DOMContentLoaded', () => {
  // State
  const state = {
    documents: [],
    stats: {},
    config: {},
    chatHistory: [],
    activeTab: 'chat', // 'chat' or 'search'
    isGenerating: false,
    apiKey: localStorage.getItem('rag_api_key') || '',
    selectedProvider: localStorage.getItem('rag_provider') || 'gemini'
  };

  // DOM Elements
  const el = {
    // Top Bar
    docCountBadge: document.getElementById('stat-docs-count'),
    chunkCountBadge: document.getElementById('stat-chunks-count'),
    providerBadge: document.getElementById('stat-provider-badge'),
    btnLoadSamples: document.getElementById('btn-load-samples'),
    btnOpenSettings: document.getElementById('btn-open-settings'),
    
    // Sidebar
    fileInput: document.getElementById('file-input'),
    dropZone: document.getElementById('drop-zone'),
    uploadProgress: document.getElementById('upload-progress'),
    progressBar: document.getElementById('progress-bar'),
    progressStatus: document.getElementById('progress-status'),
    docList: document.getElementById('doc-list'),
    docEmptyState: document.getElementById('doc-empty-state'),
    btnResetAll: document.getElementById('btn-reset-all'),
    
    // Raw Text Modal/Accordion
    btnToggleRawText: document.getElementById('btn-toggle-raw-text'),
    rawTextContainer: document.getElementById('raw-text-container'),
    rawTextTitle: document.getElementById('raw-text-title'),
    rawTextContent: document.getElementById('raw-text-content'),
    btnSubmitRawText: document.getElementById('btn-submit-raw-text'),

    // Tabs
    tabBtnChat: document.getElementById('tab-btn-chat'),
    tabBtnSearch: document.getElementById('tab-btn-search'),
    viewChat: document.getElementById('view-chat'),
    viewSearch: document.getElementById('view-search'),

    // Chat View
    chatMessages: document.getElementById('chat-messages'),
    chatInput: document.getElementById('chat-input'),
    btnSendChat: document.getElementById('btn-send-chat'),
    btnClearChat: document.getElementById('btn-clear-chat'),
    chatTopK: document.getElementById('chat-top-k'),
    presetChips: document.querySelectorAll('.preset-chip'),

    // Search View
    searchInput: document.getElementById('search-input'),
    searchTopK: document.getElementById('search-top-k'),
    btnRunSearch: document.getElementById('btn-run-search'),
    searchResultsContainer: document.getElementById('search-results-container'),
    searchEmptyState: document.getElementById('search-empty-state'),
    searchTimingBadge: document.getElementById('search-timing-badge'),

    // Settings Modal
    settingsModal: document.getElementById('settings-modal'),
    btnCloseSettings: document.getElementById('btn-close-settings'),
    settingsProvider: document.getElementById('settings-provider'),
    settingsApiKey: document.getElementById('settings-api-key'),
    settingsChunkSize: document.getElementById('settings-chunk-size'),
    settingsChunkOverlap: document.getElementById('settings-chunk-overlap'),
    btnSaveSettings: document.getElementById('btn-save-settings'),

    // Toast
    toast: document.getElementById('toast'),
    toastMessage: document.getElementById('toast-message')
  };

  // Setup Markdown parser options
  if (window.marked) {
    marked.setOptions({
      breaks: true,
      gfm: true
    });
  }

  // Toast Helper
  function showToast(message, type = 'info') {
    if (!el.toast) return;
    el.toastMessage.textContent = message;
    el.toast.className = `fixed bottom-5 right-5 z-50 flex items-center gap-3 px-4 py-3 rounded-xl shadow-xl transition-all duration-300 transform translate-y-0 text-sm font-medium ${
      type === 'error' ? 'bg-red-600 text-white' :
      type === 'success' ? 'bg-emerald-600 text-white' :
      'bg-slate-900 text-white'
    }`;
    el.toast.classList.remove('hidden', 'translate-y-10', 'opacity-0');

    setTimeout(() => {
      el.toast.classList.add('translate-y-10', 'opacity-0');
      setTimeout(() => el.toast.classList.add('hidden'), 300);
    }, 3500);
  }

  // Fetch initial stats and configuration
  async function refreshData() {
    try {
      const [stats, config] = await Promise.all([
        API.getDocuments(),
        API.getConfig()
      ]);
      state.stats = stats;
      state.documents = stats.documents || [];
      state.config = config;

      renderHeaderStats();
      renderDocumentList();
    } catch (err) {
      console.error('Erreur chargement données:', err);
    }
  }

  function renderHeaderStats() {
    if (el.docCountBadge) el.docCountBadge.textContent = state.stats.total_documents || 0;
    if (el.chunkCountBadge) el.chunkCountBadge.textContent = state.stats.total_chunks || 0;
    if (el.providerBadge) {
      const prov = state.selectedProvider || state.config.llm_provider || 'gemini';
      el.providerBadge.textContent = prov.toUpperCase();
    }
  }

  function renderDocumentList() {
    if (!el.docList) return;

    if (state.documents.length === 0) {
      el.docEmptyState.classList.remove('hidden');
      el.docList.innerHTML = '';
      return;
    }

    el.docEmptyState.classList.add('hidden');
    el.docList.innerHTML = state.documents.map(doc => {
      const typeIcons = {
        pdf: 'bi-file-earmark-pdf text-red-500',
        docx: 'bi-file-earmark-word text-blue-500',
        txt: 'bi-file-earmark-text text-slate-500',
        md: 'bi-markdown text-purple-500',
        text_note: 'bi-sticky text-amber-500'
      };
      const iconClass = typeIcons[doc.file_type] || 'bi-file-earmark text-slate-400';
      const sizeKb = (doc.file_size_bytes / 1024).toFixed(1);

      return `
        <div class="group relative flex items-center justify-between p-3 rounded-xl border border-slate-200 hover:border-blue-400 bg-white hover:shadow-sm transition-all duration-150">
          <div class="flex items-center gap-3 overflow-hidden">
            <div class="w-9 h-9 flex-shrink-0 flex items-center justify-center rounded-lg bg-slate-50 border border-slate-100">
              <i class="bi ${iconClass} text-lg"></i>
            </div>
            <div class="overflow-hidden">
              <h4 class="text-sm font-semibold text-slate-800 truncate" title="${doc.filename}">${doc.filename}</h4>
              <div class="flex items-center gap-2 text-xs text-slate-500 mt-0.5">
                <span class="bg-blue-50 text-blue-700 font-medium px-1.5 py-0.5 rounded">${doc.total_chunks} chunks</span>
                <span>${sizeKb} KB</span>
              </div>
            </div>
          </div>
          <button data-doc-id="${doc.doc_id}" class="btn-delete-doc opacity-0 group-hover:opacity-100 p-1.5 text-slate-400 hover:text-red-600 rounded-lg hover:bg-red-50 transition-all duration-150" title="Supprimer">
            <i class="bi bi-trash3"></i>
          </button>
        </div>
      `;
    }).join('');

    // Attach delete handlers
    document.querySelectorAll('.btn-delete-doc').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        const docId = btn.getAttribute('data-doc-id');
        if (confirm(`Supprimer ce document et réorganiser l'index vectoriel ?`)) {
          try {
            await API.deleteDocument(docId);
            showToast('Document supprimé avec succès', 'success');
            await refreshData();
          } catch (err) {
            showToast(err.message, 'error');
          }
        }
      });
    });
  }

  // Upload Logic
  async function handleFileUpload(files) {
    if (!files || files.length === 0) return;

    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append('files', files[i]);
    }
    formData.append('chunk_size', state.config.chunk_size || 600);
    formData.append('chunk_overlap', state.config.chunk_overlap || 120);

    el.uploadProgress.classList.remove('hidden');
    el.progressStatus.textContent = `Extraction et vectorisation de ${files.length} document(s)...`;
    el.progressBar.style.width = '60%';

    try {
      const results = await API.uploadFiles(formData);
      el.progressBar.style.width = '100%';
      const successCount = results.filter(r => r.success).length;
      showToast(`${successCount}/${files.length} document(s) indexé(s) dans FAISS !`, 'success');
      await refreshData();
    } catch (err) {
      showToast(`Échec du téléversement: ${err.message}`, 'error');
    } finally {
      setTimeout(() => {
        el.uploadProgress.classList.add('hidden');
        el.progressBar.style.width = '0%';
        el.fileInput.value = '';
      }, 800);
    }
  }

  // File drag & drop
  if (el.dropZone) {
    el.dropZone.addEventListener('click', () => el.fileInput.click());
    el.fileInput.addEventListener('change', (e) => handleFileUpload(e.target.files));

    ['dragenter', 'dragover'].forEach(name => {
      el.dropZone.addEventListener(name, (e) => {
        e.preventDefault();
        el.dropZone.classList.add('border-blue-500', 'bg-blue-50/50');
      });
    });

    ['dragleave', 'drop'].forEach(name => {
      el.dropZone.addEventListener(name, (e) => {
        e.preventDefault();
        el.dropZone.classList.remove('border-blue-500', 'bg-blue-50/50');
      });
    });

    el.dropZone.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      if (dt.files && dt.files.length) {
        handleFileUpload(dt.files);
      }
    });
  }

  // Raw text note accordion toggle
  if (el.btnToggleRawText) {
    el.btnToggleRawText.addEventListener('click', () => {
      el.rawTextContainer.classList.toggle('hidden');
    });
  }

  // Raw text submit
  if (el.btnSubmitRawText) {
    el.btnSubmitRawText.addEventListener('click', async () => {
      const title = el.rawTextTitle.value.trim();
      const content = el.rawTextContent.value.trim();
      if (!title || !content) {
        showToast('Veuillez remplir le titre et le contenu du texte', 'error');
        return;
      }

      try {
        await API.indexRawText(title, content);
        showToast(`Texte "${title}" indexé avec succès !`, 'success');
        el.rawTextTitle.value = '';
        el.rawTextContent.value = '';
        el.rawTextContainer.classList.add('hidden');
        await refreshData();
      } catch (err) {
        showToast(err.message, 'error');
      }
    });
  }

  // Reset vector store
  if (el.btnResetAll) {
    el.btnResetAll.addEventListener('click', async () => {
      if (confirm('Voulez-vous réinitialiser TOUS les documents de la base FAISS ? Cette action est irréversible.')) {
        try {
          await API.resetDocuments();
          showToast('Base vectorielle réinitialisée', 'success');
          await refreshData();
        } catch (err) {
          showToast(err.message, 'error');
        }
      }
    });
  }

  // Load sample documents
  if (el.btnLoadSamples) {
    el.btnLoadSamples.addEventListener('click', async () => {
      try {
        el.btnLoadSamples.disabled = true;
        el.btnLoadSamples.innerHTML = `<i class="bi bi-arrow-repeat animate-spin"></i> Chargement...`;
        const res = await API.loadSamples();
        showToast(res.message, 'success');
        await refreshData();
      } catch (err) {
        showToast(err.message, 'error');
      } finally {
        el.btnLoadSamples.disabled = false;
        el.btnLoadSamples.innerHTML = `<i class="bi bi-file-earmark-check"></i> <span>Charger Exemples</span>`;
      }
    });
  }

  // Tab switching
  function switchTab(tab) {
    state.activeTab = tab;
    if (tab === 'chat') {
      el.tabBtnChat.className = 'tab-btn active flex items-center gap-2 px-4 py-2 font-medium text-sm rounded-lg bg-blue-600 text-white shadow-sm transition-all';
      el.tabBtnSearch.className = 'tab-btn flex items-center gap-2 px-4 py-2 font-medium text-sm rounded-lg text-slate-600 hover:text-slate-900 hover:bg-slate-100 transition-all';
      el.viewChat.classList.remove('hidden');
      el.viewSearch.classList.add('hidden');
    } else {
      el.tabBtnSearch.className = 'tab-btn active flex items-center gap-2 px-4 py-2 font-medium text-sm rounded-lg bg-blue-600 text-white shadow-sm transition-all';
      el.tabBtnChat.className = 'tab-btn flex items-center gap-2 px-4 py-2 font-medium text-sm rounded-lg text-slate-600 hover:text-slate-900 hover:bg-slate-100 transition-all';
      el.viewSearch.classList.remove('hidden');
      el.viewChat.classList.add('hidden');
    }
  }

  if (el.tabBtnChat) el.tabBtnChat.addEventListener('click', () => switchTab('chat'));
  if (el.tabBtnSearch) el.tabBtnSearch.addEventListener('click', () => switchTab('search'));

  // Preset chips click
  el.presetChips.forEach(chip => {
    chip.addEventListener('click', () => {
      const query = chip.getAttribute('data-query');
      if (query && el.chatInput) {
        el.chatInput.value = query;
        submitChat();
      }
    });
  });

  // Chat message rendering
  function appendUserMessage(text) {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'flex justify-end gap-3 max-w-3xl ml-auto animate-fade-in';
    msgDiv.innerHTML = `
      <div class="bg-blue-600 text-white rounded-2xl rounded-tr-sm px-4 py-3 shadow-sm max-w-xl text-sm leading-relaxed">
        ${escapeHtml(text)}
      </div>
      <div class="w-8 h-8 rounded-full bg-blue-100 border border-blue-200 flex items-center justify-center text-blue-700 flex-shrink-0">
        <i class="bi bi-person-fill text-sm"></i>
      </div>
    `;
    el.chatMessages.appendChild(msgDiv);
    scrollToBottom();
  }

  function appendAssistantLoading() {
    const loadingDiv = document.createElement('div');
    loadingDiv.id = 'assistant-loading';
    loadingDiv.className = 'flex gap-3 max-w-3xl mr-auto animate-fade-in';
    loadingDiv.innerHTML = `
      <div class="w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center text-white flex-shrink-0 shadow-sm">
        <i class="bi bi-robot text-sm"></i>
      </div>
      <div class="bg-white border border-slate-200 rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm text-sm text-slate-600 flex items-center gap-3">
        <div class="flex gap-1.5">
          <span class="w-2 h-2 rounded-full bg-blue-600 animate-bounce" style="animation-delay: 0ms"></span>
          <span class="w-2 h-2 rounded-full bg-blue-600 animate-bounce" style="animation-delay: 150ms"></span>
          <span class="w-2 h-2 rounded-full bg-blue-600 animate-bounce" style="animation-delay: 300ms"></span>
        </div>
        <span class="text-xs text-slate-500 font-medium">Recherche sémantique FAISS & génération RAG en cours...</span>
      </div>
    `;
    el.chatMessages.appendChild(loadingDiv);
    scrollToBottom();
  }

  function removeAssistantLoading() {
    const loading = document.getElementById('assistant-loading');
    if (loading) loading.remove();
  }

  function appendAssistantMessage(data) {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'flex gap-3 max-w-3xl mr-auto animate-fade-in';

    const renderedAnswer = window.marked ? marked.parse(data.answer) : escapeHtml(data.answer);

    // Citations Accordion
    let citationsHtml = '';
    if (data.sources && data.sources.length > 0) {
      const citationItems = data.sources.map(src => `
        <div class="p-2.5 rounded-lg bg-slate-50 border border-slate-200/80 hover:bg-slate-100 transition-colors">
          <div class="flex items-center justify-between text-xs font-semibold text-slate-700 mb-1">
            <span class="flex items-center gap-1.5 text-blue-700">
              <i class="bi bi-file-earmark-text"></i>
              ${escapeHtml(src.source)} ${src.page ? `(p.${src.page})` : ''}
            </span>
            <span class="px-1.5 py-0.5 rounded text-[11px] font-mono bg-blue-100 text-blue-800">
              Pertinence: ${(src.relevance_score * 100).toFixed(0)}%
            </span>
          </div>
          <p class="text-xs text-slate-600 italic leading-relaxed font-serif pl-2 border-l-2 border-blue-400">
            « ${escapeHtml(src.excerpt)} »
          </p>
        </div>
      `).join('');

      citationsHtml = `
        <details class="mt-3 group border border-slate-200 rounded-xl bg-white overflow-hidden">
          <summary class="flex items-center justify-between px-3.5 py-2 cursor-pointer select-none text-xs font-semibold text-slate-700 hover:bg-slate-50 transition-colors">
            <span class="flex items-center gap-2">
              <i class="bi bi-bookmark-check-fill text-blue-600"></i>
              <span>Sources & Citations documentaires (${data.sources.length})</span>
            </span>
            <i class="bi bi-chevron-down text-slate-400 group-open:rotate-180 transition-transform"></i>
          </summary>
          <div class="p-3 pt-2 space-y-2 border-t border-slate-100 bg-slate-50/50">
            ${citationItems}
          </div>
        </details>
      `;
    }

    msgDiv.innerHTML = `
      <div class="w-8 h-8 rounded-full bg-gradient-to-tr from-blue-600 to-indigo-600 flex items-center justify-center text-white flex-shrink-0 shadow-sm mt-1">
        <i class="bi bi-robot text-sm"></i>
      </div>
      <div class="flex-1 overflow-hidden">
        <div class="bg-white border border-slate-200 rounded-2xl rounded-tl-sm p-4 shadow-sm text-sm">
          <div class="prose-rag text-slate-800">
            ${renderedAnswer}
          </div>
          ${citationsHtml}
        </div>
        <div class="flex items-center gap-3 text-[11px] text-slate-400 mt-1.5 pl-1 font-mono">
          <span><i class="bi bi-cpu"></i> ${escapeHtml(data.model || data.provider)}</span>
          <span>•</span>
          <span><i class="bi bi-search"></i> FAISS: ${data.retrieval_time_ms}ms</span>
          <span>•</span>
          <span><i class="bi bi-lightning-charge"></i> Total: ${data.total_time_ms}ms</span>
        </div>
      </div>
    `;

    el.chatMessages.appendChild(msgDiv);
    scrollToBottom();
  }

  function scrollToBottom() {
    if (el.chatMessages) {
      el.chatMessages.scrollTop = el.chatMessages.scrollHeight;
    }
  }

  function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/[&<>"']/g, function (m) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' }[m];
    });
  }

  // Submit Chat
  async function submitChat() {
    const query = el.chatInput.value.trim();
    if (!query || state.isGenerating) return;

    state.isGenerating = true;
    el.btnSendChat.disabled = true;
    appendUserMessage(query);
    el.chatInput.value = '';
    appendAssistantLoading();

    const topK = parseInt(el.chatTopK.value || 4, 10);
    const payload = {
      query: query,
      top_k: topK,
      provider: state.selectedProvider,
      api_key: state.apiKey || undefined,
      history: state.chatHistory.slice(-4)
    };

    try {
      const response = await API.chat(payload);
      removeAssistantLoading();
      appendAssistantMessage(response);

      // Save turn in history
      state.chatHistory.push({ role: 'user', content: query });
      state.chatHistory.push({ role: 'assistant', content: response.answer });
    } catch (err) {
      removeAssistantLoading();
      showToast(err.message, 'error');
    } finally {
      state.isGenerating = false;
      el.btnSendChat.disabled = false;
      el.chatInput.focus();
    }
  }

  if (el.btnSendChat) el.btnSendChat.addEventListener('click', submitChat);
  if (el.chatInput) {
    el.chatInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        submitChat();
      }
    });
  }

  if (el.btnClearChat) {
    el.btnClearChat.addEventListener('click', () => {
      state.chatHistory = [];
      el.chatMessages.innerHTML = '';
      showToast('Historique de conversation effacé', 'info');
    });
  }

  // Semantic Search Explorer
  async function runSearch() {
    const query = el.searchInput.value.trim();
    if (!query) {
      showToast('Entrez une requête de recherche', 'error');
      return;
    }

    const topK = parseInt(el.searchTopK.value || 4, 10);
    el.btnRunSearch.disabled = true;
    el.btnRunSearch.innerHTML = `<i class="bi bi-arrow-repeat animate-spin"></i>`;

    try {
      const res = await API.search(query, topK);
      renderSearchResults(res);
    } catch (err) {
      showToast(err.message, 'error');
    } finally {
      el.btnRunSearch.disabled = false;
      el.btnRunSearch.innerHTML = `<i class="bi bi-search"></i> <span>Rechercher</span>`;
    }
  }

  function renderSearchResults(data) {
    if (el.searchTimingBadge) {
      el.searchTimingBadge.textContent = `${data.total_results} résultats en ${data.time_taken_ms} ms`;
      el.searchTimingBadge.classList.remove('hidden');
    }

    if (!data.results || data.results.length === 0) {
      el.searchEmptyState.classList.remove('hidden');
      el.searchResultsContainer.innerHTML = '';
      return;
    }

    el.searchEmptyState.classList.add('hidden');
    el.searchResultsContainer.innerHTML = data.results.map((item, idx) => {
      // FAISS L2 distance: lower score means higher similarity
      const similarityPct = Math.max(0, Math.min(100, Math.round(100 / (1 + item.score))));

      return `
        <div class="p-4 rounded-xl bg-white border border-slate-200 hover:border-blue-400 shadow-sm transition-all">
          <div class="flex items-center justify-between mb-2">
            <div class="flex items-center gap-2">
              <span class="w-6 h-6 rounded-full bg-blue-100 text-blue-700 text-xs font-bold flex items-center justify-center">#${idx + 1}</span>
              <span class="text-sm font-semibold text-slate-800">${escapeHtml(item.source)}</span>
              ${item.metadata.page ? `<span class="text-xs text-slate-500 bg-slate-100 px-2 py-0.5 rounded">p.${item.metadata.page}</span>` : ''}
            </div>
            <div class="flex items-center gap-2">
              <span class="text-xs font-mono font-semibold text-blue-600 bg-blue-50 px-2 py-1 rounded-md">Score: ${similarityPct}% (Dist: ${item.score.toFixed(3)})</span>
            </div>
          </div>
          <p class="text-sm text-slate-700 font-serif leading-relaxed bg-slate-50/70 p-3 rounded-lg border border-slate-100">
            ${escapeHtml(item.content)}
          </p>
          <div class="flex items-center gap-3 text-xs text-slate-400 mt-2 font-mono">
            <span>Chunk ID: ${escapeHtml(item.chunk_id)}</span>
            <span>•</span>
            <span>Taille: ${item.content.length} caractères</span>
          </div>
        </div>
      `;
    }).join('');
  }

  if (el.btnRunSearch) el.btnRunSearch.addEventListener('click', runSearch);
  if (el.searchInput) {
    el.searchInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') runSearch();
    });
  }

  // Settings Modal Handlers
  if (el.btnOpenSettings) {
    el.btnOpenSettings.addEventListener('click', () => {
      el.settingsProvider.value = state.selectedProvider;
      el.settingsApiKey.value = state.apiKey;
      el.settingsChunkSize.value = state.config.chunk_size || 600;
      el.settingsChunkOverlap.value = state.config.chunk_overlap || 120;
      el.settingsModal.classList.remove('hidden');
    });
  }

  if (el.btnCloseSettings) {
    el.btnCloseSettings.addEventListener('click', () => {
      el.settingsModal.classList.add('hidden');
    });
  }

  if (el.btnSaveSettings) {
    el.btnSaveSettings.addEventListener('click', async () => {
      const provider = el.settingsProvider.value;
      const key = el.settingsApiKey.value.trim();
      const chunkSize = parseInt(el.settingsChunkSize.value, 10);
      const chunkOverlap = parseInt(el.settingsChunkOverlap.value, 10);

      state.selectedProvider = provider;
      state.apiKey = key;
      localStorage.setItem('rag_provider', provider);
      localStorage.setItem('rag_api_key', key);

      try {
        await API.updateConfig({
          llm_provider: provider,
          gemini_api_key: provider === 'gemini' ? key : undefined,
          openai_api_key: provider === 'openai' ? key : undefined,
          chunk_size: chunkSize,
          chunk_overlap: chunkOverlap
        });
        showToast('Paramètres mis à jour !', 'success');
        el.settingsModal.classList.add('hidden');
        await refreshData();
      } catch (err) {
        showToast(err.message, 'error');
      }
    });
  }

  // Initialize
  refreshData();
});
