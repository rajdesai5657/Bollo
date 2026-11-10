import assert from 'node:assert/strict';
import test from 'node:test';
import { readApiError } from './apiError.ts';

test('frontend displays the safe detail returned with a backend 503', async () => {
  const response = new Response(JSON.stringify({ detail: 'Gemini is temporarily unavailable. Please try again shortly.' }), {
    status: 503,
    headers: { 'content-type': 'application/json' },
  });

  assert.equal(await readApiError(response, 'fallback'), 'Gemini is temporarily unavailable. Please try again shortly.');
});

test('frontend uses a clear 503 message when the response has no readable detail', async () => {
  const response = new Response('', { status: 503 });

  assert.equal(await readApiError(response, 'fallback'), 'The speaking coach is temporarily unavailable. Please try again shortly.');
});

test('frontend gives a wait-and-retry message for HTTP 429 without retrying', async () => {
  const response = new Response('', { status: 429 });
  assert.equal(await readApiError(response, 'fallback'), 'Too many requests right now. Please wait a moment before trying again.');
});
