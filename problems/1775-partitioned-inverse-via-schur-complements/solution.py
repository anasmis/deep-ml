import numpy as np


def partitioned_inverse(M: np.ndarray, k: int) -> np.ndarray:
  M = np.asarray(M, dtype=float)
  n = M.shape[0]
  if M.ndim != 2 or M.shape[0] != M.shape[1]:
    raise ValueError("M must be a square 2D matrix.")
  if not (1 <= k < n):
    raise ValueError(f"k must satisfy 1 <= k < {n}, got {k}.")

  E = M[:k, :k]
  F = M[:k, k:]
  G = M[k:, :k]
  H = M[k:, k:]

  # H_inv_G = H^{-1} G  and  H_inv = H^{-1}
  H_inv_G = np.linalg.solve(H, G)
  H_inv = np.linalg.inv(H)

  # Schur complement M/H = E - F H^{-1} G
  schur = E - F @ H_inv_G

  # Top-left block: S^{-1}
  top_left = np.linalg.inv(schur)

  # Top-right block: -S^{-1} F H^{-1}
  F_H_inv = F @ H_inv
  top_right = -top_left @ F_H_inv

  # Bottom-left block: -H^{-1} G S^{-1}
  bottom_left = -H_inv_G @ top_left

  # Bottom-right block: H^{-1} + H^{-1} G S^{-1} F H^{-1}
  bottom_right = H_inv - bottom_left @ F_H_inv

  return np.block([[top_left, top_right], [bottom_left, bottom_right]])


def block_slogdet(M: np.ndarray, k: int) -> tuple[float, float]:
  """Compute (sign, logabsdet) of M using |M| = |M/H| * |H|.

    Parameters
    ----------
    M : np.ndarray of shape (n, n)
        Square matrix.
    k : int
        Split index with 1 <= k < n.

    Returns
    -------
    tuple[float, float]
        (sign, logabsdet) of det(M).
    """
  M = np.asarray(M, dtype=float)
  n = M.shape[0]
  if M.ndim != 2 or M.shape[0] != M.shape[1]:
    raise ValueError("M must be a square 2D matrix.")
  if not (1 <= k < n):
    raise ValueError(f"k must satisfy 1 <= k < {n}, got {k}.")

  E = M[:k, :k]
  F = M[:k, k:]
  G = M[k:, :k]
  H = M[k:, k:]

  # Schur complement M/H = E - F H^{-1} G
  schur = E - F @ np.linalg.solve(H, G)

  sign_schur, logdet_schur = np.linalg.slogdet(schur)
  sign_H, logdet_H = np.linalg.slogdet(H)

  return float(sign_schur * sign_H), float(logdet_schur + logdet_H)


def bordered_inverse(A_inv: np.ndarray, b: np.ndarray, c: float) -> np.ndarray:
  """Compute the inverse of [[A, b], [b^T, c]] in O(n^2) time given A^{-1}.

    Parameters
    ----------
    A_inv : np.ndarray of shape (n, n)
        Inverse of a symmetric n x n matrix A.
    b : np.ndarray of shape (n,)
        Border vector of length n.
    c : float
        Bottom-right scalar entry.

    Returns
    -------
    np.ndarray of shape (n + 1, n + 1)
        Inverse of the bordered matrix.
    """
  A_inv = np.asarray(A_inv, dtype=float)
  b = np.asarray(b, dtype=float).ravel()
  c = float(c)

  n = A_inv.shape[0]
  if A_inv.ndim != 2 or A_inv.shape[1] != n or b.shape[0] != n:
    raise ValueError("Incompatible shapes for A_inv and b.")

  # O(n^2) matrix-vector product: u = A^{-1} b
  u = A_inv @ b

  # O(n) inner product: Schur complement s = c - b^T A^{-1} b
  s = c - float(b @ u)
  if abs(s) < 1e-12:
    raise ValueError(
        f"Bordered matrix is singular or ill-conditioned: |s| = {abs(s):.2e} <"
        " 1e-12."
    )

  inv_s = 1.0 / s
  scaled_u = inv_s * u

  # Assemble (n + 1) x (n + 1) inverse in O(n^2) time
  out = np.empty((n + 1, n + 1), dtype=float)
  out[:n, :n] = A_inv + np.outer(scaled_u, u)
  out[:n, n] = -scaled_u
  out[n, :n] = -scaled_u
  out[n, n] = inv_s

  return out