/**
 * Retry an async operation with exponential backoff.
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

      // Don't delay after the last attempt
      if (attempt < attempts - 1) {
        const delay = Math.min(
          minDelayMs * Math.pow(2, attempt),
          maxDelayMs,
        );
        await new Promise<void>((resolve, reject) => {
          const timeout = setTimeout(resolve, delay);
          if (signal) {
            const onAbort = () => {
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
