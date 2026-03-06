import type { ContentBlock, Message } from "./types.js";

const TOKENS_PER_MESSAGE = 4;

/**
 * Recursively extract all text content from a value.
 * Handles strings, arrays, and objects with content/text fields.
 */
function extractText(value: unknown): string {
  if (typeof value === "string") {
    return value;
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
  if (value !== undefined && value !== null) {
    return String(value);
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
 * Uses chars/4 heuristic plus a fixed per-message overhead.
 */
export function estimateMessageTokens(message: Message): number {
  let text: string;
  if (typeof message.content === "string") {
    text = message.content;
  } else if (Array.isArray(message.content)) {
    text = message.content.map(extractBlockText).join("");
  } else {
    text = "";
  }
  return Math.ceil(text.length / 4) + TOKENS_PER_MESSAGE;
}

/**
 * Estimate token usage of a messages array (no API call required).
 * Uses chars/4 as a conservative heuristic. The safetyMargin option
 * in CompactionOptions accounts for underestimation on code/unicode.
 */
export function estimateTokens(messages: Message[]): number {
  let total = 0;
  for (const msg of messages) {
    total += estimateMessageTokens(msg);
  }
  return total;
}
