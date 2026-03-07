import type { ContentBlock, Message } from "./types.js";

const TOKENS_PER_MESSAGE = 4;

/**
 * Regex matching CJK Unified Ideographs, Hiragana, Katakana, and Korean Hangul.
 * These characters typically consume ~1.5 tokens each in real tokenizers,
 * not ~0.25 as the chars/4 heuristic assumes.
 */
const CJK_REGEX = /[\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7AF\u3400-\u4DBF\uF900-\uFAFF]/g;

/**
 * Default token estimation from text using an improved chars/4 heuristic.
 * CJK characters are counted at ~1.5 tokens each.
 */
function defaultTokenEstimate(text: string): number {
  const cjkMatches = text.match(CJK_REGEX);
  const cjkCount = cjkMatches ? cjkMatches.length : 0;
  const nonCjkLength = text.length - cjkCount;
  return Math.ceil(nonCjkLength / 4 + cjkCount * 1.5);
}

/**
 * Recursively extract all text content from a value.
 * Handles strings, arrays, and objects with content/text fields.
 * Skips numbers and booleans — they are data values, not text content.
 */
function extractText(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }
  // Skip non-text primitives — they are data, not text content
  if (typeof value === "number" || typeof value === "boolean") {
    return "";
  }
  if (Array.isArray(value)) {
    return value.map(extractText).join("");
  }
  if (value !== null && typeof value === "object") {
    const obj = value as Record<string, unknown>;

    // Skip tool_result.details — can be huge and untrusted
    const parts: string[] = [];
    for (const [key, val] of Object.entries(obj)) {
      if (key === "details" && obj["type"] === "tool_result") {
        continue;
      }
      parts.push(extractText(val));
    }
    return parts.join("");
  }
  return "";
}

/**
 * Extract text from a content block, skipping tool_result details fields.
 */
function extractBlockText(block: ContentBlock): string {
  const parts: string[] = [];
  for (const [key, value] of Object.entries(block)) {
    if (key === "details" && block.type === "tool_result") {
      continue;
    }
    if (key === "type") {
      continue;
    }
    parts.push(extractText(value));
  }
  return parts.join("");
}

/**
 * Estimate the token count of a single message.
 * Uses an improved heuristic with CJK awareness, or a custom tokenCounter if provided.
 */
export function estimateMessageTokens(
  message: Message,
  tokenCounter?: (text: string) => number,
): number {
  let text: string;
  if (typeof message.content === "string") {
    text = message.content;
  } else if (Array.isArray(message.content)) {
    text = message.content.map(extractBlockText).join("");
  } else {
    text = "";
  }

  if (tokenCounter) {
    return tokenCounter(text) + TOKENS_PER_MESSAGE;
  }

  return defaultTokenEstimate(text) + TOKENS_PER_MESSAGE;
}

/**
 * Estimate token usage of a messages array (no API call required).
 * When a tokenCounter is provided, it is used instead of the built-in heuristic.
 */
export function estimateTokens(
  messages: Message[],
  tokenCounter?: (text: string) => number,
): number {
  let total = 0;
  for (const msg of messages) {
    total += estimateMessageTokens(msg, tokenCounter);
  }
  return total;
}
