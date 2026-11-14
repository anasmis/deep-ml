import numpy as np


def woodbury_solve(
    sigma_diag: np.ndarray, X: np.ndarray, B: np.ndarray
) -> np.ndarray:
  """Compute (Sigma + X X^T)^{-1} B via the Woodbury matrix identity without forming N x N matrices.

    Parameters
    ----------
    sigma_diag : np.ndarray of shape (N,)
        Positive diagonal entries of Sigma.
    X : np.ndarray of shape (N, D)
        Low-rank factor matrix.
    B : np.ndarray of shape (N,) or (N, K)
        Right-hand side vector or matrix.

    Returns
    -------
    np.ndarray
        Solution (Sigma + X X^T)^{-1} B with the exact same shape as B.
    """
  sigma_diag = np.asarray(sigma_diag, dtype=float)
  X = np.asarray(X, dtype=float)
  B = np.asarray(B, dtype=float)

  if sigma_diag.ndim != 1:
    raise ValueError("sigma_diag must be 1D of shape (N,).")
  N = sigma_diag.shape[0]
  if np.any(sigma_diag <= 0):
    raise ValueError("All entries of sigma_diag must be strictly positive.")
  if X.ndim != 2 or X.shape[0] != N:
    raise ValueError(f"X must have shape ({N}, D), got {X.shape}.")
  if B.ndim not in (1, 2) or B.shape[0] != N:
    raise ValueError(f"B must have shape ({N},) or ({N}, K), got {B.shape}.")

  D = X.shape[1]
  if D == 0:
    return B / sigma_diag[:, None] if B.ndim == 2 else B / sigma_diag

  # Sigma^{-1} X via broadcasting: shape (N, D)
  sigma_inv_X = X / sigma_diag[:, None]

  # Capacitance matrix C = I_D + X^T Sigma^{-1} X: shape (D, D)
  capacitance = np.eye(D, dtype=float) + X.T @ sigma_inv_X

  # Sigma^{-1} B: shape (N,) or (N, K)
  if B.ndim == 1:
    sigma_inv_B = B / sigma_diag
  else:
    sigma_inv_B = B / sigma_diag[:, None]

  # X^T Sigma^{-1} B: shape (D,) or (D, K)
  rhs_inner = X.T @ sigma_inv_B

  # Solve D x D system: (I_D + X^T Sigma^{-1} X)^{-1} (X^T Sigma^{-1} B)
  inner_sol = np.linalg.solve(capacitance, rhs_inner)

  # Subtract Sigma^{-1} X @ inner_sol
  return sigma_inv_B - sigma_inv_X @ inner_sol


def sherman_morrison(
    A_inv: np.ndarray, u: np.ndarray, v: np.ndarray
) -> np.ndarray:
  """Compute (A + u v^T)^{-1} in O(n^2) time using the Sherman-Morrison formula.

    Parameters
    ----------
    A_inv : np.ndarray of shape (n, n)
        Inverse of an invertible n x n matrix A.
    u : np.ndarray of shape (n,)
        Left update vector.
    v : np.ndarray of shape (n,)
        Right update vector.

    Returns
    -------
    np.ndarray of shape (n, n)
        Inverse of (A + u v^T).
    """
  A_inv = np.asarray(A_inv, dtype=float)
  u = np.asarray(u, dtype=float).ravel()
  v = np.asarray(v, dtype=float).ravel()

  if A_inv.ndim != 2 or A_inv.shape[0] != A_inv.shape[1]:
    raise ValueError("A_inv must be a square 2D array.")
  n = A_inv.shape[0]
  if u.shape[0] != n or v.shape[0] != n:
    raise ValueError(f"u and v must have length {n}.")

  # O(n^2) matrix-vector products
  A_inv_u = A_inv @ u
  v_T_A_inv = v @ A_inv

  denom = 1.0 + float(v @ A_inv_u)
  if np.isclose(denom, 0.0, atol=1e-12):
    raise ValueError("Rank-1 update is singular: 1 + v^T A^{-1} u == 0.")

  # O(n^2) rank-1 subtraction
  return A_inv - np.outer(A_inv_u, v_T_A_inv) / denom


def logdet_rank_one_update(
    logdet_A: float, A_inv: np.ndarray, u: np.ndarray, v: np.ndarray
) -> float:
  """Compute log|A + u v^T| using the matrix determinant lemma.

    Parameters
    ----------
    logdet_A : float
        Log-determinant log|A| where |A| > 0.
    A_inv : np.ndarray of shape (n, n)
        Inverse of A.
    u : np.ndarray of shape (n,)
        Left update vector.
    v : np.ndarray of shape (n,)
        Right update vector.

    Returns
    -------
    float
        Updated log-determinant log|A + u v^T|.
    """
  A_inv = np.asarray(A_inv, dtype=float)
  u = np.asarray(u, dtype=float).ravel()
  v = np.asarray(v, dtype=float).ravel()

  if A_inv.ndim != 2 or A_inv.shape[0] != A_inv.shape[1]:
    raise ValueError("A_inv must be a square 2D array.")
  n = A_inv.shape[0]
  if u.shape[0] != n or v.shape[0] != n:
    raise ValueError(f"u and v must have length {n}.")

  factor = 1.0 + float(v @ (A_inv @ u))
  if factor <= 0.0:
    raise ValueError(
        f"Updated determinant is not positive: 1 + v^T A^{-1} u = {factor} <="
        " 0."
    )

  return float(logdet_A + np.log(factor))
