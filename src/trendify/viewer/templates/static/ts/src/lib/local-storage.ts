/**
 * Storage helpers that never throw. Browsers can block `localStorage` outright (a sandboxed
 * iframe, or third-party storage settings when the viewer is embedded in another site), and the
 * plain API throws a SecurityError then. Values are kept in memory for the page's lifetime in
 * that case, so the UI still works; it just forgets them on reload.
 */
const fallback = new Map<string, string>();

export function readStorage(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return fallback.get(key) ?? null;
  }
}

export function writeStorage(key: string, value: string): void {
  try {
    localStorage.setItem(key, value);
  } catch {
    fallback.set(key, value);
  }
}

export function removeStorage(key: string): void {
  try {
    localStorage.removeItem(key);
  } catch {
    fallback.delete(key);
  }
}

/** JSON get/set over storage, tolerant of missing keys and malformed/stale JSON. */
export function loadJSON<T>(key: string): T | null {
  const raw = readStorage(key);
  if (raw === null) return null;
  try {
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}

export function saveJSON(key: string, value: unknown): void {
  writeStorage(key, JSON.stringify(value));
}
