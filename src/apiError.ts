export async function readApiError(response: Response, fallback: string): Promise<string> {
  try {
    const payload: unknown = await response.json();
    if (payload && typeof payload === 'object' && 'detail' in payload) {
      const detail = (payload as { detail?: unknown }).detail;
      if (typeof detail === 'string' && detail.trim()) return detail;
    }
  } catch {
    // Fall through to a status-specific or caller-provided safe message.
  }

  if (response.status === 503) return 'The speaking coach is temporarily unavailable. Please try again shortly.';
  if (response.status === 429) return 'Too many requests right now. Please wait a moment before trying again.';
  return fallback;
}
