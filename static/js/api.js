/**
 * RAG-DocAssist API Client
 */
const API = {
  baseUrl: '',

  async getHealth() {
    const res = await fetch(`${this.baseUrl}/api/health`);
    if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
    return await res.json();
  },

  async getConfig() {
    const res = await fetch(`${this.baseUrl}/api/config`);
    if (!res.ok) throw new Error(`Get config failed: ${res.statusText}`);
    return await res.json();
  },

  async updateConfig(payload) {
    const res = await fetch(`${this.baseUrl}/api/config`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) throw new Error(`Update config failed: ${res.statusText}`);
    return await res.json();
  },

  async getDocuments() {
    const res = await fetch(`${this.baseUrl}/api/documents`);
    if (!res.ok) throw new Error(`Get documents failed: ${res.statusText}`);
    return await res.json();
  },

  async uploadFiles(formData) {
    const res = await fetch(`${this.baseUrl}/api/documents/upload`, {
      method: 'POST',
      body: formData
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Upload failed');
    }
    return await res.json();
  },

  async indexRawText(title, text, chunkSize, chunkOverlap) {
    const res = await fetch(`${this.baseUrl}/api/documents/index-text`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        title,
        text,
        chunk_size: chunkSize || undefined,
        chunk_overlap: chunkOverlap || undefined
      })
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Indexing failed');
    }
    return await res.json();
  },

  async deleteDocument(docId) {
    const res = await fetch(`${this.baseUrl}/api/documents/${encodeURIComponent(docId)}`, {
      method: 'DELETE'
    });
    if (!res.ok) throw new Error(`Delete failed: ${res.statusText}`);
    return await res.json();
  },

  async resetDocuments() {
    const res = await fetch(`${this.baseUrl}/api/documents`, {
      method: 'DELETE'
    });
    if (!res.ok) throw new Error(`Reset failed: ${res.statusText}`);
    return await res.json();
  },

  async loadSamples() {
    const res = await fetch(`${this.baseUrl}/api/documents/load-samples`, {
      method: 'POST'
    });
    if (!res.ok) throw new Error(`Load samples failed: ${res.statusText}`);
    return await res.json();
  },

  async search(query, topK = 4, scoreThreshold = null) {
    const res = await fetch(`${this.baseUrl}/api/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        top_k: parseInt(topK, 10),
        score_threshold: scoreThreshold ? parseFloat(scoreThreshold) : null
      })
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Search failed');
    }
    return await res.json();
  },

  async chat(payload) {
    const res = await fetch(`${this.baseUrl}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Chat query failed');
    }
    return await res.json();
  }
};
