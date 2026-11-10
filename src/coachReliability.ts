export function microphoneErrorMessage(error: unknown): string {
  const name = error && typeof error === 'object' && 'name' in error
    ? String((error as { name?: unknown }).name ?? '')
    : '';

  if (name === 'NotAllowedError' || name === 'PermissionDeniedError') {
    return 'Microphone access was blocked. Allow microphone access in your browser, or use text input.';
  }
  if (name === 'NotFoundError' || name === 'DevicesNotFoundError') {
    return 'No microphone was found. Connect an audio input device, or use text input.';
  }
  if (name === 'NotReadableError' || name === 'TrackStartError') {
    return 'The microphone is unavailable or being used by another app. Check your audio input, or use text input.';
  }
  if (name === 'SecurityError') {
    return 'Microphone access needs a secure browser connection. Open Bollo over HTTPS, or use text input.';
  }
  return 'Could not start the microphone. Check your audio input, or use text input instead.';
}

const nonSpeechTranscript = /^(?:\[?(?:silence|noise|background noise|music|inaudible|unintelligible|no speech|empty)\]?|<unk>)$/i;

export function hasSpeakableTranscript(transcript: string): boolean {
  const cleaned = transcript.trim();
  return Boolean(cleaned && /[\p{L}\p{N}]/u.test(cleaned) && !nonSpeechTranscript.test(cleaned));
}
