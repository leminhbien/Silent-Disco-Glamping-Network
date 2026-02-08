import numpy as np
import gurobipy as gp
from gurobipy import Model, GRB
from typing import Tuple


def qp_model(
    n: int,
    matrix: np.ndarray,
    k: int,
    timeout: int = 3600,
) -> Tuple[float, float, np.ndarray, float]:
    """Solves the quadratic problem using Gurobi's native QP solver.

    Args:
        n: The number of variables.
        matrix: The symmetric matrix Q of quadratic coefficients.
        k: The number of nodes to select.
        timeout: The maximum time in seconds for the solver.

    Raises:
        ValueError: If inputs n, k, or timeout are not positive integers,
                    if k > n, or if the matrix is not symmetric or correctly shaped.
        TypeError: If the matrix is not a numpy array.
        RuntimeError: If the Gurobi optimization fails.

    Returns:
        A tuple containing:
            - The objective value of the solution.
            - The solver runtime.
            - A numpy array representing the solution vector.
            - The final MIP gap.
    """
    # --- Input Validation ---
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer.")
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")
    if not isinstance(timeout, int) or timeout <= 0:
        raise ValueError("timeout must be a positive integer.")
    if k > n:
        raise ValueError(f"k ({k}) cannot be greater than n ({n}).")
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy array.")
    if matrix.shape != (n, n):
        raise ValueError(f"Matrix shape must be ({n}, {n}), but got {matrix.shape}.")
    if not np.allclose(matrix, matrix.T):
        raise ValueError("Matrix must be symmetric.")
    # --- End Validation ---

    qp = None
    try:
        # Define model object
        qp = Model("QP")
        qp.setParam("TimeLimit", timeout)
        qp.setParam("OutputFlag", 0)  # suppress output

        # Set variables
        x_qp = qp.addVars(n, vtype=GRB.BINARY, name="xqp")

        # Select exactly k nodes
        qp.addConstr(gp.quicksum(x_qp[i] for i in range(n)) == k)

        # Add objective function
        qp.setObjective(
            gp.quicksum(
                matrix[i, j] * x_qp[i] * x_qp[j] for i in range(n) for j in range(i, n)
            ),
            GRB.MAXIMIZE,
        )

        # Solve the quadratic program
        qp.optimize()

        # get solution
        return qp.ObjVal, qp.Runtime, np.array([v.X for v in x_qp.values()]), qp.MIPGap

    except gp.GurobiError as e:
        print(f"Gurobi error in qp_model: {e}")
        raise RuntimeError("Gurobi optimization failed.") from e
    finally:
        # Ensure the model is disposed of to free resources
        if qp:
            qp.dispose()


def first_linear(
    n: int,
    matrix: np.ndarray,
    k: int,
    timeout: int = 3600,
) -> Tuple[float, float, np.ndarray, float]:
    """Solves the problem using the first standard linearization technique.

    Args:
        n: The number of variables.
        matrix: The symmetric matrix Q of quadratic coefficients.
        k: The number of nodes to select.
        timeout: The maximum time in seconds for the solver.

    Raises:
        ValueError: If inputs n, k, or timeout are not positive integers,
                    if k > n, or if the matrix is not symmetric or correctly shaped.
        TypeError: If the matrix is not a numpy array.
        RuntimeError: If the Gurobi optimization fails.

    Returns:
        A tuple containing:
            - The objective value of the solution.
            - The solver runtime.
            - A numpy array representing the solution vector.
            - The final MIP gap.
    """
    # --- Input Validation ---
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer.")
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")
    if not isinstance(timeout, int) or timeout <= 0:
        raise ValueError("timeout must be a positive integer.")
    if k > n:
        raise ValueError(f"k ({k}) cannot be greater than n ({n}).")
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy array.")
    if matrix.shape != (n, n):
        raise ValueError(f"Matrix shape must be ({n}, {n}), but got {matrix.shape}.")
    if not np.allclose(matrix, matrix.T):
        raise ValueError("Matrix must be symmetric.")
    # --- End Validation ---

    m = None
    try:
        # Setup model
        m = Model("Linear1")
        m.setParam("TimeLimit", timeout)  # time limits
        m.setParam("OutputFlag", 0)  # suppress output

        # Set variables
        x = m.addVars(n, vtype=GRB.BINARY, name="x")

        # Set auxiliary variables
        y = m.addVars(
            [(i, j) for i in range(n) for j in range(i, n)],
            vtype=GRB.CONTINUOUS,
            name="y",
        )

        # Add constraints to models m
        # Select exactly k nodes
        m.addConstr(gp.quicksum(x[i] for i in range(n)) == k)

        # Add auxiliary constraints x_i+y_j <= y_ij
        m.addConstrs(x[i] + x[j] - 1 <= y[i, j] for i in range(n) for j in range(i, n))

        # Add auxiliary constraints y_ij <= x_i
        m.addConstrs(y[i, j] <= x[i] for i in range(n) for j in range(i, n))

        # Add auxiliary constraints y_ij <= x_j
        m.addConstrs(y[i, j] <= x[j] for i in range(n) for j in range(i, n))

        # Add objective function
        m.setObjective(
            gp.quicksum(matrix[i, j] * y[i, j] for i in range(n) for j in range(i, n)),
            GRB.MAXIMIZE,
        )

        # Solve
        m.optimize()

        return m.ObjVal, m.Runtime, np.array([v.X for v in x.values()]), m.MIPGap

    except gp.GurobiError as e:
        print(f"Gurobi error in first_linear: {e}")
        raise RuntimeError("Gurobi optimization failed.") from e
    finally:
        if m:
            m.dispose()


def second_linear(
    n: int,
    matrix: np.ndarray,
    k: int,
    timeout: int = 3600,
) -> Tuple[float, float, np.ndarray, float]:
    """Solves the problem using the second linearization technique.

    Args:
        n: The number of variables.
        matrix: The symmetric matrix Q of quadratic coefficients.
        k: The number of nodes to select.
        timeout: The maximum time in seconds for the solver.

    Raises:
        ValueError: If inputs n, k, or timeout are not positive integers,
                    if k > n, or if the matrix is not symmetric or correctly shaped.
        TypeError: If the matrix is not a numpy array.
        RuntimeError: If the Gurobi optimization fails.

    Returns:
        A tuple containing:
            - The objective value of the solution.
            - The solver runtime.
            - A numpy array representing the solution vector.
            - The final MIP gap.
    """
    # --- Input Validation ---
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer.")
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")
    if not isinstance(timeout, int) or timeout <= 0:
        raise ValueError("timeout must be a positive integer.")
    if k > n:
        raise ValueError(f"k ({k}) cannot be greater than n ({n}).")
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy array.")
    if matrix.shape != (n, n):
        raise ValueError(f"Matrix shape must be ({n}, {n}), but got {matrix.shape}.")
    if not np.allclose(matrix, matrix.T):
        raise ValueError("Matrix must be symmetric.")
    # --- End Validation ---

    g = None
    try:
        # Get upperbound and lowerbound
        ubi = [k * max(matrix[i, j] for j in range(n)) for i in range(n)]

        lbi = [k * min(matrix[i, j] for j in range(n)) for i in range(n)]

        # Lowerbound for all variables
        lowerbound = min(0, min(lbi))

        # Setup model
        g = Model("Linear2")

        # Set time limit
        g.setParam("TimeLimit", timeout)
        g.setParam("OutputFlag", 0)

        # Get variables
        x_g = g.addVars(n, vtype=GRB.BINARY, name="x_g")

        # Get auxiliary variables
        w = g.addVars(n, vtype=GRB.CONTINUOUS, name="w", lb=lowerbound)

        # Select exactly k nodes
        g.addConstr(gp.quicksum(x_g[i] for i in range(n)) == k)

        # Constraint for w, x, ub where w <= x * ub
        g.addConstrs(w[i] <= x_g[i] * ubi[i] for i in range(n))

        # Constraint for w, x, lb where w >= x * lb
        g.addConstrs(w[i] >= x_g[i] * lbi[i] for i in range(n))

        # Constraint for w, x, q, lb where w <= sum(x * Q[i, j]) - lbi[i] * (1 - x[i])
        g.addConstrs(
            w[i]
            <= gp.quicksum(x_g[j] * matrix[i, j] for j in range(n))
            - lbi[i] * (1 - x_g[i])
            for i in range(n)
        )

        # Constraint for w, x, q, ub
        g.addConstrs(
            w[i]
            >= gp.quicksum(x_g[j] * matrix[i, j] for j in range(n))
            - ubi[i] * (1 - x_g[i])
            for i in range(n)
        )

        # Add objective function
        g.setObjective(
            0.5
            * (
                gp.quicksum(w[i] for i in range(n))
                + gp.quicksum(matrix[i, i] * x_g[i] for i in range(n))
            ),
            GRB.MAXIMIZE,
        )

        # Solve
        g.optimize()

        return g.ObjVal, g.Runtime, np.array([v.X for v in x_g.values()]), g.MIPGap

    except gp.GurobiError as e:
        print(f"Gurobi error in second_linear: {e}")
        raise RuntimeError("Gurobi optimization failed.") from e
    finally:
        if g:
            g.dispose()
