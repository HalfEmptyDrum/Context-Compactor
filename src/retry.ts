/**
 * Check if an error is retryable based on HTTP status code.
 * 401, 400, 403 are non-retryable (auth/format errors won't self-resolve).
 * 429 (rate limit), 5xx (server errors), and unknown errors are retryable.
 */
function isRetryableError(error: unknown): boolean {
  if (error && typeof error === "object") {
    const e = error as Record<string, unknown>;
    const status =
      typeof e["status"] === "number"
        ? e["status"]
        : typeof e["statusCode"] === "number"
          ? e["statusCode"]
          : undefined;
    if (typeof status === "number") {
      if (status === 429) return true;
      if (status >= 400 && status < 500) return false;
      if (status >= 500) return true;
    }
  }
  return true;
}

/**
 * Retry an async operation with exponential backoff.
 * Non-retryable errors (401, 400, 403) throw immediately.
 */
export async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  options: {
    attempts?: number;
    minDelayMs?: number;
    maxDelayMs?: number;
    signal?: AbortSignal;
  } = {},
): Promise<T> {
  const {
    attempts = 3,
    minDelayMs = 500,
    maxDelayMs = 5000,
    signal,
  } = options;

  let lastError: unknown;

  for (let attempt = 0; attempt < attempts; attempt++) {
    signal?.throwIfAborted();

    try {
      return await fn();
    } catch (error) {
      lastError = error;

      // Don't retry on abort
      if (signal?.aborted) {
        throw error;
      }

      // Don't retry non-retryable errors (401, 400, 403, etc.)
      if (!isRetryableError(error)) {
        throw error;
      }

      // Don't delay after the last attempt
      if (attempt < attempts - 1) {
        const delay = Math.min(
          minDelayMs * Math.pow(2, attempt),
          maxDelayMs,
        );
        await new Promise<void>((resolve, reject) => {
          let onAbort: (() => void) | undefined;
          const timeout = setTimeout(() => {
            // Clean up the abort listener to prevent memory leaks
            if (signal && onAbort) {
              signal.removeEventListener("abort", onAbort);
            }
            resolve();
          }, delay);
          if (signal) {
            onAbort = () => {
              clearTimeout(timeout);
              reject(signal.reason);
            };
            signal.addEventListener("abort", onAbort, { once: true });
          }
        });
      }
    }
  }

  throw lastError;
}
