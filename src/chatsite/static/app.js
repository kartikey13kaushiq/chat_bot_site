// Chat client: streams replies over SSE (fetch + ReadableStream) and keeps history in localStorage.
// All model and user text is rendered with textContent, never as HTML.
'use strict';

const STORE = 'chat.history.v1';
const log = document.getElementById('log');
const input = document.getElementById('input');
const form = document.getElementById('composer');
const send = document.getElementById('send');
const stop = document.getElementById('stop');
let history = load();
let controller = null;

function load() {
  try { return JSON.parse(localStorage.getItem(STORE)) || []; } catch { return []; }
}
function save() {
  try { localStorage.setItem(STORE, JSON.stringify(history.slice(-100))); } catch { /* storage full or blocked */ }
}

function bubble(role, text) {
  document.getElementById('empty')?.remove();
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  div.textContent = text;
  log.append(div);
  log.scrollTop = log.scrollHeight;
  return div;
}

history.forEach(m => bubble(m.role, m.content));

function setBusy(busy) {
  send.disabled = busy;
  stop.hidden = !busy;
  input.disabled = busy;
}

async function ask(text) {
  history.push({ role: 'user', content: text });
  save();
  bubble('user', text);
  const reply = bubble('assistant', '');
  reply.classList.add('pending');
  setBusy(true);
  controller = new AbortController();
  let answer = '';
  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: history }),
      signal: controller.signal,
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(typeof body.detail === 'string' ? body.detail : `Request failed (${res.status})`);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let boundary;
      while ((boundary = buffer.indexOf('\n\n')) >= 0) {
        const raw = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const isError = raw.startsWith('event: error');
        const data = raw.split('\n').find(l => l.startsWith('data: '));
        if (!data) continue;
        const payload = JSON.parse(data.slice(6));
        if (isError) throw new Error(payload.message);
        if (payload.delta) {
          answer += payload.delta;
          reply.textContent = answer;
          log.scrollTop = log.scrollHeight;
        }
      }
    }
  } catch (err) {
    if (err.name !== 'AbortError') bubble('error', err.message);
  } finally {
    reply.classList.remove('pending');
    if (answer.trim()) {
      history.push({ role: 'assistant', content: answer.trim() });
    } else {
      reply.remove();
      history.pop(); // keep history alternating so the next request is valid
    }
    save();
    setBusy(false);
    input.focus();
  }
}

form.addEventListener('submit', e => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  ask(text);
});
input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
});
stop.addEventListener('click', () => controller?.abort());
document.getElementById('new-chat').addEventListener('click', () => {
  controller?.abort();
  history = [];
  save();
  log.replaceChildren();
  const p = document.createElement('p');
  p.className = 'empty';
  p.id = 'empty';
  p.textContent = 'New conversation. Ask me anything.';
  log.append(p);
});
