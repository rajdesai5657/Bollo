import assert from 'node:assert/strict';
import test from 'node:test';
import { hasSpeakableTranscript, microphoneErrorMessage } from './coachReliability.ts';

test('empty and non-speech transcripts are not valid conversation turns', () => {
  for (const transcript of ['', '   ', '[silence]', '[background noise]', '<unk>', '...']) {
    assert.equal(hasSpeakableTranscript(transcript), false, JSON.stringify(transcript));
  }
});

test('short but real speech is accepted', () => {
  assert.equal(hasSpeakableTranscript('No.'), true);
  assert.equal(hasSpeakableTranscript('Hi'), true);
});

test('microphone permission and missing-device errors have clear guidance', () => {
  assert.match(microphoneErrorMessage({ name: 'NotAllowedError' }), /access was blocked/i);
  assert.match(microphoneErrorMessage({ name: 'NotFoundError' }), /No microphone was found/i);
  assert.match(microphoneErrorMessage({ name: 'NotReadableError' }), /unavailable/i);
});
