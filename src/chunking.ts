import type { Message } from "./types.js";
import { estimateMessageTokens, estimateTokens } from "./tokens.js";

const BASE_CHUNK_RATIO = 0.4;
const MIN_CHUNK_RATIO = 0.15;

/**
 * Split messages into chunks of at most maxTokens, applying safetyMargin.
 * Oversized single messages get their own chunk.
 */
export function chunkByMaxTokens(
  messages: Message[],
  maxTokens: number,
  safetyMargin: number,
): Message[][] {
  const effectiveMax = Math.floor(maxTokens / safetyMargin);
  if (effectiveMax <= 0) {
    return messages.map((m) => [m]);
  }

  const chunks: Message[][] = [];
  let currentChunk: Message[] = [];
  let currentTokens = 0;

  for (const msg of messages) {
    const msgTokens = estimateMessageTokens(msg);

    // Oversized single message gets its own chunk
    if (msgTokens > effectiveMax) {
      if (currentChunk.length > 0) {
        chunks.push(currentChunk);
        currentChunk = [];
        currentTokens = 0;
      }
      chunks.push([msg]);
      continue;
    }

    if (currentTokens + msgTokens > effectiveMax && currentChunk.length > 0) {
      chunks.push(currentChunk);
      currentChunk = [];
      currentTokens = 0;
    }

    currentChunk.push(msg);
    currentTokens += msgTokens;
  }

  if (currentChunk.length > 0) {
    chunks.push(currentChunk);
  }

  return chunks;
}

/**
 * Split messages into N equal-token-share parts (for parallel summarization).
 */
export function splitIntoEqualParts(
  messages: Message[],
  parts: number,
): Message[][] {
  if (parts <= 1 || messages.length === 0) {
    return messages.length > 0 ? [messages] : [];
  }

  const totalTokens = estimateTokens(messages);
  const targetPerPart = Math.ceil(totalTokens / parts);

  const result: Message[][] = [];
  let currentPart: Message[] = [];
  let currentTokens = 0;

  for (const msg of messages) {
    const msgTokens = estimateMessageTokens(msg);

    if (
      currentTokens + msgTokens > targetPerPart &&
      currentPart.length > 0 &&
      result.length < parts - 1
    ) {
      result.push(currentPart);
      currentPart = [];
      currentTokens = 0;
    }

    currentPart.push(msg);
    currentTokens += msgTokens;
  }

  if (currentPart.length > 0) {
    result.push(currentPart);
  }

  return result;
}

/**
 * Compute adaptive chunk ratio based on average message size relative to context window.
 * If messages are large relative to the context window, reduce chunk size to avoid
 * the summarization call hitting its own context limit.
 */
export function computeAdaptiveChunkRatio(
  messages: Message[],
  contextWindowTokens: number,
  safetyMargin: number = 1.2,
): number {
  if (messages.length === 0 || contextWindowTokens <= 0) {
    return BASE_CHUNK_RATIO;
  }

  const totalTokens = estimateTokens(messages);
  const avgTokensPerMessage = totalTokens / messages.length;
  const avgRatio = (avgTokensPerMessage * safetyMargin) / contextWindowTokens;

  if (avgRatio > 0.1) {
    const reduction = Math.min(avgRatio * 2, BASE_CHUNK_RATIO - MIN_CHUNK_RATIO);
    return Math.max(MIN_CHUNK_RATIO, BASE_CHUNK_RATIO - reduction);
  }

  return BASE_CHUNK_RATIO;
}
