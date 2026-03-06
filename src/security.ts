import type { ContentBlock, Message } from "./types.js";

/**
 * Recursively strip `details` fields from tool_result content blocks.
 * These can contain untrusted/verbose payloads from tool executions
 * that should not reach the summarization model.
 */
function stripBlockDetails(block: ContentBlock): ContentBlock {
  if (block.type === "tool_result") {
    const { details: _, ...rest } = block;
    // Also strip details from nested content blocks
    if (Array.isArray(rest.content)) {
      rest.content = (rest.content as ContentBlock[]).map(stripBlockDetails);
    }
    return rest as ContentBlock;
  }
  // Recurse into nested content arrays
  if (Array.isArray(block.content)) {
    return {
      ...block,
      content: (block.content as ContentBlock[]).map(stripBlockDetails),
    };
  }
  return block;
}

/**
 * Strip all tool_result.details fields from a message array.
 * Returns a new array — does not mutate the input.
 */
export function stripToolResultDetails(messages: Message[]): Message[] {
  return messages.map((msg) => {
    if (typeof msg.content === "string") {
      return msg;
    }
    if (!Array.isArray(msg.content)) {
      return msg;
    }
    const hasToolResult = msg.content.some(
      (block) => block.type === "tool_result",
    );
    if (!hasToolResult && msg.role !== "tool") {
      return msg;
    }
    return {
      ...msg,
      content: msg.content.map(stripBlockDetails),
    };
  });
}
