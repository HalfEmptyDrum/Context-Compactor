import type { ContentBlock, Message } from "./types.js";

const DEFAULT_STRIP_FIELDS = new Set([
  "details",
  "raw_response",
  "stderr",
  "debug_info",
  "raw_output",
]);

/** Fields that should never be stripped (structural/semantic fields). */
const PROTECTED_FIELDS = new Set([
  "type",
  "content",
  "role",
  "id",
  "tool_use_id",
  "name",
  "text",
  "source",
]);

/**
 * Recursively strip sensitive fields from a content block.
 */
function stripSensitiveFields(
  block: ContentBlock,
  fieldsToStrip: Set<string>,
): ContentBlock {
  if (block.type === "tool_result") {
    const cleaned: Record<string, unknown> = {};
    for (const [key, val] of Object.entries(block)) {
      if (fieldsToStrip.has(key)) continue;
      cleaned[key] = val;
    }
    // Recurse into nested content blocks
    if (Array.isArray(cleaned["content"])) {
      cleaned["content"] = (cleaned["content"] as ContentBlock[]).map((b) =>
        stripSensitiveFields(b, fieldsToStrip),
      );
    }
    return cleaned as ContentBlock;
  }
  // Recurse into nested content arrays
  if (Array.isArray(block.content)) {
    return {
      ...block,
      content: (block.content as ContentBlock[]).map((b) =>
        stripSensitiveFields(b, fieldsToStrip),
      ),
    };
  }
  return block;
}

/**
 * Strip sensitive fields from tool_result blocks and role="tool" messages.
 * Default strip list: details, raw_response, stderr, debug_info, raw_output.
 * Returns a new array — does not mutate the input.
 */
export function stripToolResultDetails(
  messages: Message[],
  customFields?: string[],
): Message[] {
  const fieldsToStrip = new Set([
    ...DEFAULT_STRIP_FIELDS,
    ...(customFields ?? []),
  ]);

  return messages.map((msg) => {
    // Handle top-level role="tool" messages — strip sensitive fields from the message itself
    if (msg.role === "tool") {
      const cleaned: Record<string, unknown> = {};
      for (const [key, val] of Object.entries(msg)) {
        if (fieldsToStrip.has(key) && !PROTECTED_FIELDS.has(key)) continue;
        cleaned[key] = val;
      }
      // If content is array of blocks, also strip within blocks
      if (Array.isArray(cleaned["content"])) {
        cleaned["content"] = (cleaned["content"] as ContentBlock[]).map((b) =>
          stripSensitiveFields(b, fieldsToStrip),
        );
      }
      return cleaned as Message;
    }

    if (typeof msg.content === "string") {
      return msg;
    }
    if (!Array.isArray(msg.content)) {
      return msg;
    }
    const hasToolResult = msg.content.some(
      (block) => block.type === "tool_result",
    );
    if (!hasToolResult) {
      return msg;
    }
    return {
      ...msg,
      content: msg.content.map((b) => stripSensitiveFields(b, fieldsToStrip)),
    };
  });
}
