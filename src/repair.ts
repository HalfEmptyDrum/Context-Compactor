import type { ContentBlock, Message } from "./types.js";

/**
 * Repair orphaned tool_result AND tool_use blocks whose partner was dropped.
 * After compaction drops old messages, some blocks may reference
 * a partner that was in the dropped chunk. These orphans cause API errors.
 *
 * Handles both directions:
 * - Orphaned tool_result blocks (tool_use was dropped)
 * - Orphaned tool_use blocks (tool_result was dropped)
 *
 * And both formats:
 * - Top-level role="tool" messages with a tool_use_id field
 * - Content blocks of type "tool_result" / "tool_use" within message content arrays
 */
export function repairToolPairing(messages: Message[]): {
  messages: Message[];
  droppedOrphanCount: number;
  droppedOrphanUseCount: number;
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
  const afterResultRepair: Message[] = [];

  for (const msg of messages) {
    // Handle top-level role="tool" messages
    if (msg.role === "tool" && typeof msg.tool_use_id === "string") {
      if (!toolUseIds.has(msg.tool_use_id)) {
        droppedOrphanCount++;
        continue;
      }
      afterResultRepair.push(msg);
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
        afterResultRepair.push({ ...msg, content: filteredBlocks });
      } else {
        afterResultRepair.push(msg);
      }
      continue;
    }

    afterResultRepair.push(msg);
  }

  // Third pass: collect all tool_result IDs, then remove orphaned tool_uses
  const toolResultIds = new Set<string>();

  for (const msg of afterResultRepair) {
    if (msg.role === "tool" && typeof msg.tool_use_id === "string") {
      toolResultIds.add(msg.tool_use_id);
    }
    if (Array.isArray(msg.content)) {
      for (const block of msg.content) {
        if (
          block.type === "tool_result" &&
          typeof block.tool_use_id === "string"
        ) {
          toolResultIds.add(block.tool_use_id);
        }
      }
    }
  }

  let droppedOrphanUseCount = 0;
  const finalResult: Message[] = [];

  for (const msg of afterResultRepair) {
    if (Array.isArray(msg.content)) {
      const hasToolUse = msg.content.some((b) => b.type === "tool_use");
      if (hasToolUse) {
        const filteredBlocks: ContentBlock[] = [];

        for (const block of msg.content) {
          if (
            block.type === "tool_use" &&
            typeof block.id === "string" &&
            !toolResultIds.has(block.id)
          ) {
            droppedOrphanUseCount++;
            continue;
          }
          filteredBlocks.push(block);
        }

        if (filteredBlocks.length === 0) {
          // All blocks were orphaned tool_uses — drop entire message
          continue;
        }
        if (filteredBlocks.length !== msg.content.length) {
          finalResult.push({ ...msg, content: filteredBlocks });
        } else {
          finalResult.push(msg);
        }
        continue;
      }
    }
    finalResult.push(msg);
  }

  return {
    messages: finalResult,
    droppedOrphanCount,
    droppedOrphanUseCount,
  };
}
