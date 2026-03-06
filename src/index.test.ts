import { describe, expect, it, vi } from "vitest";
import {
  compact,
  compactIfNeeded,
  estimateTokens,
  repairToolPairing,
} from "./index.js";
import type { Message, SummarizeFn } from "./types.js";
import { chunkByMaxTokens, splitIntoEqualParts } from "./chunking.js";
import { stripToolResultDetails } from "./security.js";
import { buildInstructions, IDENTIFIER_INSTRUCTIONS } from "./summarize.js";

// Helper to create messages with predictable token counts
function msg(role: Message["role"], content: string): Message {
  return { role, content };
}

function longMsg(role: Message["role"], tokens: number): Message {
  // chars/4 + 4 overhead, so we need (tokens - 4) * 4 chars
  const chars = Math.max(0, (tokens - 4) * 4);
  return { role, content: "x".repeat(chars) };
}

// Mock summarize that returns predictable output
const mockSummarize: SummarizeFn = async (
  messages,
  _instructions,
  previousSummary,
) => {
  const prefix = previousSummary ? `${previousSummary} + ` : "";
  return `${prefix}Summary of ${messages.length} messages`;
};

// Failing summarize for fallback tests
const failingSummarize: SummarizeFn = async () => {
  throw new Error("Summarization failed");
};

describe("estimateTokens", () => {
  it("returns reasonable estimates for text messages", () => {
    const messages: Message[] = [
      msg("user", "Hello, how are you?"),
      msg("assistant", "I'm doing well, thank you!"),
    ];
    const tokens = estimateTokens(messages);
    // "Hello, how are you?" = 19 chars => ceil(19/4) + 4 = 9
    // "I'm doing well, thank you!" = 26 chars => ceil(26/4) + 4 = 11
    // Total = 20
    expect(tokens).toBe(20);
  });

  it("returns reasonable estimates for code messages", () => {
    const code = `function fibonacci(n: number): number {
  if (n <= 1) return n;
  return fibonacci(n - 1) + fibonacci(n - 2);
}`;
    const messages: Message[] = [msg("assistant", code)];
    const tokens = estimateTokens(messages);
    expect(tokens).toBeGreaterThan(20);
    expect(tokens).toBeLessThan(200);
  });

  it("handles content block arrays", () => {
    const messages: Message[] = [
      {
        role: "assistant",
        content: [
          { type: "text", text: "Hello world" },
          { type: "text", text: "More text" },
        ],
      },
    ];
    const tokens = estimateTokens(messages);
    expect(tokens).toBeGreaterThan(4); // At least overhead
  });

  it("skips tool_result details in estimation", () => {
    const messagesWithDetails: Message[] = [
      {
        role: "user",
        content: [
          {
            type: "tool_result",
            tool_use_id: "abc",
            content: "short result",
            details: "x".repeat(100000), // huge payload
          },
        ],
      },
    ];
    const messagesWithoutDetails: Message[] = [
      {
        role: "user",
        content: [
          {
            type: "tool_result",
            tool_use_id: "abc",
            content: "short result",
          },
        ],
      },
    ];
    const tokensWithDetails = estimateTokens(messagesWithDetails);
    const tokensWithout = estimateTokens(messagesWithoutDetails);
    // Should be similar since details are skipped
    expect(tokensWithDetails).toBe(tokensWithout);
  });
});

describe("compactIfNeeded", () => {
  it("returns compacted=false when under threshold", async () => {
    const messages: Message[] = [
      msg("user", "Hello"),
      msg("assistant", "Hi there"),
    ];
    const result = await compactIfNeeded({
      messages,
      contextWindowTokens: 100000,
      summarize: mockSummarize,
    });
    expect(result.compacted).toBe(false);
    expect(result.messages).toBe(messages); // Same reference
    expect(result.summary).toBe("");
  });

  it("triggers compaction when over threshold", async () => {
    // Create many messages to exceed threshold
    const messages: Message[] = [];
    for (let i = 0; i < 100; i++) {
      messages.push(msg("user", `Message number ${i} with some extra content to fill tokens`));
      messages.push(msg("assistant", `Response to message ${i} with additional details`));
    }

    const totalTokens = estimateTokens(messages);
    // Set context window so that 85% is below totalTokens
    const contextWindow = Math.floor(totalTokens / 0.8);

    const result = await compactIfNeeded({
      messages,
      contextWindowTokens: contextWindow,
      summarize: mockSummarize,
    });

    expect(result.compacted).toBe(true);
    expect(result.messages.length).toBeLessThan(messages.length);
    expect(result.stats.messagesCompressed).toBeGreaterThan(0);
    expect(result.stats.messagesKept).toBeGreaterThan(0);
    expect(result.stats.tokensAfter).toBeLessThan(result.stats.tokensBefore);
  });
});

describe("compact", () => {
  it("result.messages starts with summary message, then toKeep", async () => {
    const toSummarize: Message[] = [
      msg("user", "Old message 1"),
      msg("assistant", "Old response 1"),
    ];
    const toKeep: Message[] = [
      msg("user", "Recent message"),
      msg("assistant", "Recent response"),
    ];

    const result = await compact({
      toSummarize,
      toKeep,
      summarize: mockSummarize,
    });

    expect(result.compacted).toBe(true);
    expect(result.messages.length).toBe(3); // summary + 2 kept
    expect(result.messages[0].role).toBe("user");
    expect(result.messages[0].content).toContain("[Previous conversation summary]");
    expect(result.messages[0].content).toContain("Summary of 2 messages");
    expect(result.messages[1]).toEqual(toKeep[0]);
    expect(result.messages[2]).toEqual(toKeep[1]);
  });

  it("tracks stats correctly", async () => {
    const toSummarize: Message[] = [
      msg("user", "Compress me"),
      msg("assistant", "Compressed"),
    ];
    const toKeep: Message[] = [msg("user", "Keep me")];

    const result = await compact({
      toSummarize,
      toKeep,
      summarize: mockSummarize,
    });

    expect(result.stats.messagesCompressed).toBe(2);
    expect(result.stats.messagesKept).toBe(1);
    expect(result.stats.tokensBefore).toBeGreaterThan(0);
    expect(result.stats.tokensAfter).toBeGreaterThan(0);
  });
});

describe("chunking", () => {
  it("splits messages correctly at maxTokens with safetyMargin", () => {
    // Each message ~54 tokens: "x" * 200 => 200/4 + 4 = 54
    const messages: Message[] = Array.from({ length: 10 }, () =>
      msg("user", "x".repeat(200)),
    );

    // 54 tokens per msg, safetyMargin 1.0, maxTokens 120 => ~2 msgs per chunk
    const chunks = chunkByMaxTokens(messages, 120, 1.0);
    expect(chunks.length).toBeGreaterThan(1);
    for (const chunk of chunks) {
      const tokens = estimateTokens(chunk);
      expect(tokens).toBeLessThanOrEqual(120);
    }
  });

  it("oversized single message gets its own chunk", () => {
    const small = msg("user", "small");
    const large = longMsg("user", 500);
    const messages = [small, large, small];

    // maxTokens 100, safety 1.0 => effectiveMax 100
    // Large message (~500 tokens) exceeds 100 => own chunk
    const chunks = chunkByMaxTokens(messages, 100, 1.0);
    expect(chunks.length).toBe(3);
    expect(chunks[0]).toEqual([small]);
    expect(chunks[1]).toEqual([large]);
    expect(chunks[2]).toEqual([small]);
  });

  it("splitIntoEqualParts divides evenly by tokens", () => {
    const messages: Message[] = Array.from({ length: 12 }, () =>
      msg("user", "x".repeat(100)),
    );

    const parts = splitIntoEqualParts(messages, 3);
    expect(parts.length).toBe(3);
    // Each part should have roughly equal token count
    const partTokens = parts.map((p) => estimateTokens(p));
    const maxDiff =
      Math.max(...partTokens) - Math.min(...partTokens);
    // Allow some variance but should be roughly equal
    expect(maxDiff).toBeLessThan(estimateTokens(messages) * 0.3);
  });
});

describe("summarizeInStages", () => {
  it("partial summaries are merged correctly with parts > 1", async () => {
    const messages: Message[] = Array.from({ length: 20 }, (_, i) =>
      msg("user", `Message ${i} with content to give it some length for chunking`),
    );

    const summaries: string[] = [];
    const capturingSummarize: SummarizeFn = async (
      msgs,
      _instructions,
      prev,
    ) => {
      const s = prev
        ? `${prev} + Summary(${msgs.length})`
        : `Summary(${msgs.length})`;
      summaries.push(s);
      return s;
    };

    const result = await compactIfNeeded({
      messages,
      contextWindowTokens: 200, // Small to force compaction
      triggerRatio: 0.1,
      summarize: capturingSummarize,
      options: { parts: 2 },
    });

    expect(result.compacted).toBe(true);
    expect(summaries.length).toBeGreaterThan(0);
  });
});

describe("fallback", () => {
  it("final fallback note when everything fails", async () => {
    const messages: Message[] = Array.from({ length: 5 }, (_, i) =>
      msg("user", `Message ${i}`),
    );

    const result = await compact({
      toSummarize: messages,
      toKeep: [msg("assistant", "recent")],
      summarize: failingSummarize,
    });

    expect(result.compacted).toBe(true);
    // Should contain the fallback message
    expect(result.summary).toContain("messages");
    expect(result.summary).toContain("Summary unavailable");
  });

  it("when full summarization fails, partial runs if possible", async () => {
    let callCount = 0;
    const sometimesFails: SummarizeFn = async (messages, _inst, _prev) => {
      callCount++;
      // First 3 calls fail (retry attempts for the full pass), then succeed
      if (callCount <= 3) {
        throw new Error("fail");
      }
      return `Partial summary of ${messages.length}`;
    };

    // Create messages where some are oversized relative to context
    const smallMsgs: Message[] = Array.from({ length: 3 }, () =>
      msg("user", "small message here"),
    );
    // Add a huge message
    const hugeMsg = longMsg("user", 60000);
    const allMessages = [...smallMsgs, hugeMsg];

    const result = await compact({
      toSummarize: allMessages,
      toKeep: [msg("assistant", "kept")],
      summarize: sometimesFails,
      contextWindowTokens: 100000,
    });

    expect(result.compacted).toBe(true);
    // Fallback was needed
    expect(result.summary).toBeTruthy();
  });
});

describe("repairToolPairing", () => {
  it("orphaned tool_results are removed, count returned", () => {
    const messages: Message[] = [
      {
        role: "assistant",
        content: [
          { type: "tool_use", id: "tool-1", name: "search", input: {} },
        ],
      },
      {
        role: "user",
        content: [
          {
            type: "tool_result",
            tool_use_id: "tool-1",
            content: "result 1",
          },
          {
            type: "tool_result",
            tool_use_id: "tool-orphan",
            content: "orphaned result",
          },
        ],
      },
    ];

    const { messages: repaired, droppedOrphanCount } =
      repairToolPairing(messages);

    expect(droppedOrphanCount).toBe(1);
    expect(repaired.length).toBe(2);

    const resultBlocks = repaired[1].content as Array<{
      type: string;
      tool_use_id: string;
    }>;
    expect(resultBlocks.length).toBe(1);
    expect(resultBlocks[0].tool_use_id).toBe("tool-1");
  });

  it("handles top-level role=tool messages", () => {
    const messages: Message[] = [
      msg("user", "Hello"),
      {
        role: "tool",
        content: "orphaned result",
        tool_use_id: "nonexistent",
      },
      msg("assistant", "Response"),
    ];

    const { messages: repaired, droppedOrphanCount } =
      repairToolPairing(messages);

    expect(droppedOrphanCount).toBe(1);
    expect(repaired.length).toBe(2);
    expect(repaired[0].content).toBe("Hello");
    expect(repaired[1].content).toBe("Response");
  });

  it("preserves valid tool pairs", () => {
    const messages: Message[] = [
      {
        role: "assistant",
        content: [
          { type: "tool_use", id: "t1", name: "search", input: {} },
        ],
      },
      {
        role: "user",
        content: [
          { type: "tool_result", tool_use_id: "t1", content: "result" },
        ],
      },
    ];

    const { messages: repaired, droppedOrphanCount } =
      repairToolPairing(messages);

    expect(droppedOrphanCount).toBe(0);
    expect(repaired.length).toBe(2);
  });
});

describe("identifierPolicy", () => {
  it("strict adds preservation instructions to prompt", () => {
    const instructions = buildInstructions({ identifierPolicy: "strict" });
    expect(instructions).toContain("Preserve all opaque identifiers");
    expect(instructions).toContain("UUIDs");
  });

  it("off sends no identifier instructions", () => {
    const instructions = buildInstructions({ identifierPolicy: "off" });
    expect(instructions).not.toContain("Preserve all opaque identifiers");
  });

  it("custom uses custom text", () => {
    const custom = "Keep all file paths intact.";
    const instructions = buildInstructions({
      identifierPolicy: "custom",
      identifierInstructions: custom,
    });
    expect(instructions).toContain(custom);
    expect(instructions).not.toContain("Preserve all opaque identifiers");
  });

  it("default is strict", () => {
    const instructions = buildInstructions({});
    expect(instructions).toContain(IDENTIFIER_INSTRUCTIONS);
  });
});

describe("security", () => {
  it("tool_result.details are stripped before passing to summarize", async () => {
    let capturedMessages: Message[] = [];
    const capturingSummarize: SummarizeFn = async (messages) => {
      capturedMessages = messages;
      return "summary";
    };

    const toSummarize: Message[] = [
      {
        role: "user",
        content: [
          {
            type: "tool_result",
            tool_use_id: "t1",
            content: "result",
            details: "SENSITIVE DATA THAT SHOULD BE STRIPPED",
          },
        ],
      },
    ];

    await compact({
      toSummarize,
      toKeep: [msg("assistant", "kept")],
      summarize: capturingSummarize,
    });

    // The captured messages should not have details
    expect(capturedMessages.length).toBe(1);
    const block = (capturedMessages[0].content as Array<Record<string, unknown>>)[0];
    expect(block.details).toBeUndefined();
    expect(block.content).toBe("result");
  });

  it("stripToolResultDetails preserves non-tool_result blocks", () => {
    const messages: Message[] = [
      {
        role: "assistant",
        content: [
          { type: "text", text: "hello", details: "should stay" },
        ],
      },
    ];

    const stripped = stripToolResultDetails(messages);
    const block = (stripped[0].content as Array<Record<string, unknown>>)[0];
    expect(block.details).toBe("should stay");
  });
});

describe("AbortSignal", () => {
  it("abort propagates to summarize calls", async () => {
    const controller = new AbortController();
    const summarizeSpy: SummarizeFn = vi.fn(async () => {
      // Simulate some work
      await new Promise((resolve) => setTimeout(resolve, 10));
      return "summary";
    });

    // Abort immediately
    controller.abort(new Error("User cancelled"));

    const result = compactIfNeeded({
      messages: Array.from({ length: 50 }, (_, i) =>
        msg("user", `msg ${i} with enough content to trigger compaction`),
      ),
      contextWindowTokens: 200,
      triggerRatio: 0.1,
      summarize: summarizeSpy,
      options: { signal: controller.signal },
    });

    await expect(result).rejects.toThrow();
  });
});
