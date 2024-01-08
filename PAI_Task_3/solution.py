"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel, ConstantKernel
import matplotlib.pyplot as plt


# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA


class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""

        # Used to track values and plotting
        self.x_list = np.array([])
        self.f_list = np.array([])
        self.v_list = np.array([])
        self.const = 0

        # Parameters
        noise_level_f = 0.15 ** 2  # Square of the standard deviation for f
        noise_level_v = 0.0001 ** 2  # Square of the standard deviation for v, logP and SA
        self.prior_mean_v = 4

        # Kernel for the objective function 'f'
        matern_kernel_f = Matern(length_scale=10, nu=2.5, length_scale_bounds=(0.5, 10)) + WhiteKernel(noise_level=noise_level_f)
        self.gp_objective = GaussianProcessRegressor(kernel=matern_kernel_f)

        # Kernel for the constraint function 'v'
        matern_kernel_v = (Matern(length_scale=10, nu=2.5, length_scale_bounds=(0.5, 10)) + WhiteKernel(noise_level=noise_level_v))
        self.gp_constraint = GaussianProcessRegressor(kernel=matern_kernel_v, normalize_y=False)


    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: float
            the next point to evaluate
        """
        x_opt = self.optimize_acquisition_function()
        return x_opt


    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                 np.random.rand(DOMAIN.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()
        return x_opt


    def acquisition_function(self, x: np.ndarray):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        x = np.atleast_2d(x)

        # Make predictions for previous and new values 
        mu_f_safe, _ = self.gp_objective.predict(np.atleast_2d(self.x), return_std=True)
        mu_f, sigma_f = self.gp_objective.predict(x, return_std=True)
        mu_v, _ = self.gp_constraint.predict(x, return_std=True)

        # Implement EI
        kappa = 0.5  
        ei = np.maximum(mu_f - mu_f_safe, 0)
        ei_less_explored = ei /(sigma_f / kappa)
        
        # Penalize points predicted to be unsafe
        safety_penalty = np.minimum(0, mu_v + 4*np.ones(len(mu_v)) - SAFETY_THRESHOLD)
        af_value = ei_less_explored - 20 * safety_penalty

        return af_value


    def add_data_point(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """

        self.x = x
        self.x_list = np.append(self.x_list, x)
        self.f_list = np.append(self.f_list, f)
        self.v_list = np.append(self.v_list, v)
        self.gp_objective.fit(self.x_list.reshape(-1, 1), self.f_list)
        self.gp_constraint.fit(self.x_list.reshape(-1, 1), self.v_list - 4*np.ones(len(self.v_list)))

        if self.const == 0:
            self.const = x


    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        # Generate a dense grid of points within the domain
        x_candidates = np.linspace(*DOMAIN[0], 1000)[:, None]

        # Predict objective and constraint values at each point
        mu_f, _ = self.gp_objective.predict(x_candidates, return_std=True)
        mu_v, _ = self.gp_constraint.predict(x_candidates, return_std=True)

        # Apply the safety constraint
        safe_indices = mu_v.flatten() < SAFETY_THRESHOLD
        safe_x_candidates = x_candidates[safe_indices]
        safe_mu_f = mu_f[safe_indices]

        if len(safe_x_candidates) == 0:
            # No safe points found
            raise ValueError("No safe points found within the domain.")

        # Select the best point based on the objective function
        best_index = np.argmax(safe_mu_f)
        solution = safe_x_candidates[best_index].item()

        cb_value = 0
        for index, i in enumerate(self.f_list):
            if self.v_list[index] < 4 and i > cb_value:
                cb_value = i

        return solution
    

    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        x_candidates = np.linspace(*DOMAIN[0], 200)

        # Predict objective and constraint values at each point
        mu_f, fs = self.gp_objective.predict(x_candidates, return_std=True)
        mu_v, vs = self.gp_constraint.predict(x_candidates, return_std=True)
        mu_v += 4

        plt.fill_between(x_candidates, mu_f + fs, mu_f - fs, color = "black")
        plt.fill_between(x_candidates, mu_v + vs, mu_v - vs, color = "blue")
        plt.plot(x_candidates, mu_f, color = "grey")
        plt.plot(x_candidates, mu_v, color = "green")
        plt.scatter(self.x_list, self.f_list, color= "red")
        plt.scatter(self.x_list, self.v_list, color= "green")
        plt.scatter(self.const, 1)

        plt.show()

# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


def f(x: float):
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return - np.linalg.norm(x - mid_point, 2)


def v(x: float):
    """Dummy SA"""
    return 2.0


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

    return x_init


def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)

    # Loop until budget is exhausted
    for j in range(50):
        # Get next recommendation
        x = agent.next_recommendation()

        # Obtain objective and constraint observation
        obj_val = f(x) + np.random.randn()
        cost_val = v(x) + np.random.randn()
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = agent.get_solution()
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the' \
        f'DOMAIN, {solution} returned instead'

    # Compute regret
    regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}\nUnsafe-evals TODO\n')

    agent.plot()

if __name__ == "__main__":
    main()