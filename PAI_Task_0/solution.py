import numpy
from scipy.stats import laplace, norm, t
import scipy
import math
import numpy as np

VARIANCE = 2.0

normal_scale = math.sqrt(VARIANCE)
student_t_df = (2 * VARIANCE) / (VARIANCE - 1)
laplace_scale = VARIANCE / 2

HYPOTHESIS_SPACE = [norm(loc=0.0, scale=math.sqrt(VARIANCE)),
                    laplace(loc=0.0, scale=laplace_scale),
                    t(df=student_t_df)]

PRIOR_PROBS = np.array([0.35, 0.25, 0.4])


def generate_sample(n_samples, seed=None):
    """ data generating process of the Bayesian model """
    random_state = np.random.RandomState(seed)
    hypothesis_idx = np.random.choice(3, p=PRIOR_PROBS)
    dist = HYPOTHESIS_SPACE[hypothesis_idx]
    return dist.rvs(n_samples, random_state=random_state)


""" Solution """

from scipy.special import logsumexp


def log_posterior_probs(x):
    """
    Computes the log posterior probabilities for the three hypotheses, given the data x

    Args:
        x (np.ndarray): one-dimensional numpy array containing the training data
    Returns:
        log_posterior_probs (np.ndarray): a numpy array of size 3, containing the Bayesian log-posterior probabilities
                                          corresponding to the three hypotheses
    """
    assert x.ndim == 1

    # TODO: enter your code here
    # Using the Bayesian inference relies on using the Bayes Theorem to calculate the posterior probability
    # Applying log to calculate the probabilities with Bayes is Log(P(Hi|X)) = Log(P(X|Hi)) + Log(P(Hi)) - Log(P(X))

    #Log(P(X|Hi)) can be calculated using the pdf for the distribution
    XHi_norm = np.sum(np.log(norm.pdf(x, loc=0.0, scale=math.sqrt(VARIANCE))))
    XHi_laplace = np.sum(np.log(laplace.pdf(x, loc=0.0, scale=laplace_scale)))
    XHi_t = np.sum(np.log(t.pdf(x, df=student_t_df)))

    #Log(P(Hi)) is the probabilty given in PRIOR_PROBS
    Hi_norm = np.log(PRIOR_PROBS[0])
    Hi_laplace = np.log(PRIOR_PROBS[1])
    Hi_t = np.log(PRIOR_PROBS[2])

    #Log(P(X)) is the probability that the set X shows under any hypothesis
    x_all = logsumexp([XHi_norm + Hi_norm, XHi_laplace + Hi_laplace, XHi_t + Hi_t])

    #Log(P(Hi|X)) = Log(P(X|Hi)) + Log(P(Hi)) - Log(P(X))
    log_posterior_prob_norm = XHi_norm + Hi_norm - x_all
    log_posterior_prob_laplace = XHi_laplace + Hi_laplace - x_all
    log_posterior_prob_t = XHi_t + Hi_t - x_all
    log_p = np.array([log_posterior_prob_norm, log_posterior_prob_laplace, log_posterior_prob_t])

    assert log_p.shape == (3,)
    return log_p


def posterior_probs(x):
    return np.exp(log_posterior_probs(x))


""" """


def main():
    """ sample from Laplace dist """
    dist = HYPOTHESIS_SPACE[1]
    x = dist.rvs(1000, random_state=28)

    print("Posterior probs for 1 sample from Laplacian")
    p = posterior_probs(x[:1])
    print("Normal: %.4f , Laplace: %.4f, Student-t: %.4f\n" % tuple(p))

    print("Posterior probs for 50 samples from Laplacian")
    p = posterior_probs(x[:50])
    print("Normal: %.4f , Laplace: %.4f, Student-t: %.4f\n" % tuple(p))

    print("Posterior probs for 1000 samples from Laplacian")
    p = posterior_probs(x[:1000])
    print("Normal: %.4f , Laplace: %.4f, Student-t: %.4f\n" % tuple(p))

    print("Posterior for 100 samples from the Bayesian data generating process")
    x = generate_sample(n_samples=100)
    p = posterior_probs(x)
    print("Normal: %.4f , Laplace: %.4f, Student-t: %.4f\n" % tuple(p))


if __name__ == "__main__":
    main()
