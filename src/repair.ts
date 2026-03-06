import type { ContentBlock, Message } from "./types.js";

/**
 * Repair orphaned tool_result blocks whose tool_use partner was dropped.
 * After compaction drops old messages, some tool_result blocks may reference
 * a tool_use that was in the dropped chunk. These orphans cause API errors.
 *
 * Handles both:
 * - Top-level role="tool" messages with a tool_use_id field
 * - Content blocks of type "tool_result" within message content arrays
 */
export function repairToolPairing(messages: Message[]): {
  messages: Message[];
  droppedOrphanCount: number;
} {
  // First pass: collect all tool_use IDs
  const toolUseIds = new Set<string>();

  for (const msg of messages) {
    if (typeof msg.content === "string") {
      continue;
    }
    if (Array.isArray(msg.content)) {
      for (const block of msg.content) {
        if (block.type === "tool_use" && typeof block.id === "string") {
          toolUseIds.add(block.id);
        }
      }
    }
  }

  // Second pass: filter out orphaned tool_results
  let droppedOrphanCount = 0;
  const result: Message[] = [];

  for (const msg of messages) {
    // Handle top-level role="tool" messages
    if (msg.role === "tool" && typeof msg.tool_use_id === "string") {
      if (!toolUseIds.has(msg.tool_use_id)) {
        droppedOrphanCount++;
        continue;
      }
      result.push(msg);
      continue;
    }

    // Handle content block arrays
    if (Array.isArray(msg.content)) {
      const filteredBlocks: ContentBlock[] = [];
      let dropped = false;

      for (const block of msg.content) {
        if (
          block.type === "tool_result" &&
          typeof block.tool_use_id === "string" &&
          !toolUseIds.has(block.tool_use_id)
        ) {
          droppedOrphanCount++;
          dropped = true;
          continue;
        }
        filteredBlocks.push(block);
      }

      if (dropped) {
        // If all blocks were dropped, skip the message entirely
        if (filteredBlocks.length === 0) {
          continue;
        }
        result.push({ ...msg, content: filteredBlocks });
      } else {
        result.push(msg);
      }
      continue;
    }

    result.push(msg);
  }

  return { messages: result, droppedOrphanCount };
}
