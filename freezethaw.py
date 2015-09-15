from __future__ import division, print_function

__author__ = 'James Robert Lloyd'
__description__ = 'Freeze thaw related code including samplers - will be expanded into several modules eventually probs'

import os
import sys
root_dir = os.path.dirname(__file__)
sys.path.append(root_dir)

import copy
import random

import numpy as np
import scipy.stats


class WarmLearner(object):
    """Wrapper around things like random forest that don't have a warm start method"""
    def __init__(self, base_model):
        self.base_model = base_model
        self.model = copy.deepcopy(self.base_model)
        self.n_estimators = self.model.n_estimators
        self.first_fit = True

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def fit(self, X, y):
        if self.first_fit:
            self.model.fit(X, y)
            self.first_fit = False
        # Keep training and appending base estimators to main model
        while self.model.n_estimators < self.n_estimators:
            self.base_model.fit(X, y)
            self.model.estimators_ += self.base_model.estimators_
            self.model.n_estimators = len(self.model.estimators_)
        # Clip any extra models produced
        self.model.estimators_ = self.model.estimators_[:self.n_estimators]
        self.model.n_estimators = self.n_estimators


def trunc_norm_mean_upper_tail(a, mean, std):
    alpha = (a - mean) / std
    num = scipy.stats.norm.pdf(alpha)
    den = (1 - scipy.stats.norm.cdf(alpha))
    if num == 0 or den == 0:
        # Numerical nasties
        if a < mean:
            return mean
        else:
            return a
    else:
        lambd = scipy.stats.norm.pdf(alpha) / (1 - scipy.stats.norm.cdf(alpha))
        return mean + std * lambd


def ft_K_t_t(t, t_star, scale, alpha, beta):
    """Exponential decay mixture kernel"""
    # Check 1d
    # TODO - Abstract this checking behaviour - check pybo and gpy for inspiration
    t = np.array(t)
    t_star = np.array(t_star)
    assert t.ndim == 1 or (t.ndim == 2 and t.shape[1] == 1)
    assert t_star.ndim == 1 or (t_star.ndim == 2 and t_star.shape[1] == 1)
    # Create kernel
    K_t_t = np.zeros((len(t), len(t_star)))
    for i in range(len(t)):
        for j in range(len(t_star)):
            K_t_t[i, j] = scale * (beta ** alpha) / ((t[i] + t_star[j] + beta) ** alpha)
    return K_t_t


def ft_K_t_t_plus_noise(t, t_star, scale, alpha, beta, log_noise):
    """Ronseal - clearly this behaviour should be abstracted"""
    # TODO - abstract kernel addition etc
    noise = np.exp(log_noise)
    K_t_t = ft_K_t_t(t, t_star, scale=scale, alpha=alpha, beta=beta)
    K_noise = cov_iid(t, t_star, scale=noise)
    return K_t_t + K_noise


def cov_iid(x, z=None, scale=1):
    """Identity kernel, scaled"""
    if z is None:
        z = x
    # Check 1d
    x = np.array(x)
    z = np.array(z)
    assert x.ndim == 1 or (x.ndim == 2 and x.shape[1] == 1)
    assert z.ndim == 1 or (z.ndim == 2 and z.shape[1] == 1)
    # Create kernel
    K = np.zeros((len(x), len(z)))
    if not np.all(x == z):
        # FIXME - Is this the correct behaviour?
        return K
    for i in range(min(len(x), len(z))):
        K[i, i] = scale
    return K


def cov_matern_5_2(x, z=None, scale=1, ell=1):
    """Identity kernel, scaled"""
    if z is None:
        z = x
    # Check 1d
    x = np.array(x, ndmin=2)
    z = np.array(z, ndmin=2)
    if x.shape[1] > 1:
        x = x.T
    if z.shape[1] > 1:
        z = z.T
    assert (x.ndim == 2 and x.shape[1] == 1)
    assert (z.ndim == 2 and z.shape[1] == 1)
    # Create kernel
    x = x * np.sqrt(5) / ell
    z = z * np.sqrt(5) / ell
    sqdist = np.sum(x**2,1).reshape(-1,1) + np.sum(z**2,1) - 2*np.dot(x, z.T)
    K = sqdist
    f = lambda a: 1 + a * (1 + a / 3)
    m = lambda b: f(b) * np.exp(-b)
    for i in range(len(K)):
        for j in range(len(K[i])):
            K[i, j] = m(K[i, j])
    K *= scale
    return K


def slice_sample_bounded_max(N, burn, logdist, xx, widths, step_out, max_attempts, bounds):
    """
    Slice sampling with bounds and max iterations
    Iain Murray May 2004, tweaks June 2009, a diagnostic added Feb 2010
    See Pseudo-code in David MacKay's text book p375
    Modified by James Lloyd, May 2012 - max attempts
    Modified by James Lloyd, Jan 2015 - bounds
    Ported to python by James Lloyd, Feb 2015
    """
    xx = copy.deepcopy(xx)
    D = len(xx)
    samples = []
    if (not isinstance(widths, list)) or len(widths) == 1:
        widths = np.ones(D) * widths

    log_Px = logdist(xx)

    for ii in range(N + burn):
        log_uprime = np.log(random.random()) + log_Px
        # print('xx = %s' % xx)
        # print('Current ll = %f' % log_Px)
        # print('Slice = %f' % log_uprime)
        for dd in random.sample(range(D), D):
            # print('dd = %d' % dd)
            # print('xx = %s' % xx)
            x_l = copy.deepcopy(xx)
            # print('x_l = %s' % x_l)
            x_r = copy.deepcopy(xx)
            xprime = copy.deepcopy(xx)

            # Create a horizontal interval (x_l, x_r) enclosing xx
            rr = random.random()
            # print(xx[dd])
            # print(rr)
            # print(widths[dd])
            # print(bounds[dd][0])
            x_l[dd] = max(xx[dd] - rr*widths[dd], bounds[dd][0])
            x_r[dd] = min(xx[dd] + (1-rr)*widths[dd], bounds[dd][1])
            # print('x_l = %s' % x_l)
            # if x_l[3] > 0:
            #     print('Large noise')
            if step_out:
                steps = 0
                while logdist(x_l) > log_uprime and x_l[dd] > bounds[dd][0]:
                    # if x_l[3] > 0:
                    #     print('Large noise')
                    x_l[dd] = max(x_l[dd] - widths[dd], bounds[dd][0])
                    steps += 1
                    if steps > max_attempts:
                        break
                steps = 0
                while logdist(x_r) > log_uprime and x_r[dd] < bounds[dd][1]:
                    x_r[dd] = min(x_r[dd] + widths[dd], bounds[dd][1])
                    steps += 1
                    if steps > max_attempts:
                        break
            # print(x_l[dd])

            # Propose xprimes and shrink interval until good one found
            zz = 0
            num_attempts = 0
            while True:
                zz += 1
                # print(x_l)
                xprime[dd] = random.random()*(x_r[dd] - x_l[dd]) + x_l[dd]
                # print(x_l[dd])
                # if xprime[3] > 0:
                #     print('Large noise')
                log_Px = logdist(xprime)
                if log_Px > log_uprime:
                    xx[dd] = xprime[dd]
                    # print(dd)
                    # print(xx)
                    break
                else:
                    # Shrink in
                    num_attempts += 1
                    if num_attempts >= max_attempts:
                        # print('Failed to find something')
                        break
                    elif xprime[dd] > xx[dd]:
                        x_r[dd] = xprime[dd]
                    elif xprime[dd] < xx[dd]:
                        x_l[dd] = xprime[dd]
                    else:
                        raise Exception('Slice sampling failed to find an acceptable point')
        # Record samples
        if ii >= burn:
            samples.append(copy.deepcopy(xx))
    return samples


# noinspection PyTypeChecker
def ft_ll(m, t, y, x, x_kernel, x_kernel_params, t_kernel, t_kernel_params):
    """Freeze thaw log likelihood"""
    # Take copies of everything - this is a function
    m = copy.deepcopy(m)
    t = copy.deepcopy(t)
    y = copy.deepcopy(y)
    x = copy.deepcopy(x)

    K_x = x_kernel(x, x, **x_kernel_params)
    N = len(y)

    lambd = np.zeros((N, 1))
    gamma = np.zeros((N, 1))

    K_t = [None] * N

    for n in range(N):
        K_t[n] = t_kernel(t[n], t[n], **t_kernel_params)
        lambd[n] = np.dot(np.ones((1, len(t[n]))), np.linalg.solve(K_t[n], np.ones((len(t[n]), 1))))
        # Making sure y[n] is a column vector
        y[n] = np.array(y[n], ndmin=2)
        if y[n].shape[0] == 1:
            y[n] = y[n].T
        gamma[n] = np.dot(np.ones((1, len(t[n]))), np.linalg.solve(K_t[n], y[n] - m[n] * np.ones(y[n].shape)))

    Lambd = np.diag(lambd.ravel())

    ll = 0

    # Terms relating to individual curves
    for n in range(N):
        ll += - 0.5 * np.dot((y[n] - m[n] * np.ones(y[n].shape)).T,
                             np.linalg.solve(K_t[n], y[n] - m[n] * np.ones(y[n].shape)))
        ll += - 0.5 * np.log(np.linalg.det(K_t[n]))

    # Terms relating to K_x
    ll += + 0.5 * np.dot(gamma.T, np.linalg.solve(np.linalg.inv(K_x) + Lambd, gamma))
    ll += - 0.5 * np.log(np.linalg.det(np.linalg.inv(K_x) + Lambd))
    ll += - 0.5 * np.log(np.linalg.det(K_x))

    # Prior on kernel params
    # TODO - abstract me
    # ll += scipy.stats.norm.logpdf(np.log(t_kernel_params['a']))
    # ll += scipy.stats.norm.logpdf(np.log(t_kernel_params['b']))
    # ll += np.log(1 / t_kernel_params['scale'])
    ll -= t_kernel_params['log_noise']  # Zero loving prior

    return ll


# noinspection PyTypeChecker
def ft_posterior(m, t, y, t_star, x, x_kernel, x_kernel_params, t_kernel, t_kernel_params):
    """Freeze thaw posterior (predictive)"""
    # Take copies of everything - this is a function
    m = copy.deepcopy(m)
    t = copy.deepcopy(t)
    y = copy.deepcopy(y)
    t_star = copy.deepcopy(t_star)
    x = copy.deepcopy(x)

    K_x = x_kernel(x, x, **x_kernel_params)
    N = len(y)

    lambd = np.zeros((N, 1))
    gamma = np.zeros((N, 1))
    Omega = [None] * N

    K_t = [None] * N
    K_t_t_star = [None] * N

    y_mean = [None] * N

    for n in range(N):
        K_t[n] = t_kernel(t[n], t[n], **t_kernel_params)
        # TODO - Distinguish between the curve we are interested in and 'noise' with multiple kernels
        K_t_t_star[n] = t_kernel(t[n], t_star[n], **t_kernel_params)
        lambd[n] = np.dot(np.ones((1, len(t[n]))), np.linalg.solve(K_t[n], np.ones((len(t[n]), 1))))
        # Making sure y[n] is a column vector
        y[n] = np.array(y[n], ndmin=2)
        if y[n].shape[0] == 1:
            y[n] = y[n].T
        gamma[n] = np.dot(np.ones((1, len(t[n]))), np.linalg.solve(K_t[n], y[n] - m[n] * np.ones(y[n].shape)))
        Omega[n] = np.ones((len(t_star[n]), 1)) - np.dot(K_t_t_star[n].T,
                                                         np.linalg.solve(K_t[n], np.ones(y[n].shape)))

    Lambda_inv = np.diag(1 / lambd.ravel())
    C = K_x - np.dot(K_x, np.linalg.solve(K_x + Lambda_inv, K_x))
    mu = m + np.dot(C, gamma)

    # f_mean = mu
    # f_var = C

    for n in range(N):
        y_mean[n] = np.dot(K_t_t_star[n].T, np.linalg.solve(K_t[n], y[n])) + Omega[n] * mu[n]

    K_t_star_t_star = [None] * N
    y_var = [None] * N

    for n in range(N):
        K_t_star_t_star[n] = t_kernel(t_star[n], t_star[n], **t_kernel_params)
        y_var[n] = K_t_star_t_star[n] - \
                   np.dot(K_t_t_star[n].T,
                          np.linalg.solve(K_t[n], K_t_t_star[n])) + \
                   C[n, n] * np.dot(Omega[n], Omega[n].T)

    return y_mean, y_var


def exp_ll(t, y, decay, scale, log_noise, asymptote):
    """Exponential decay log likelihood"""
    fit = asymptote + scale * np.exp(-decay * np.array(t))
    # fit = asymptote * np.ones(np.array(y).shape)
    residuals = y - fit

    ll = -0.5 * np.sum(residuals * residuals) / 0.001 # / np.exp(2 * log_noise)

    return ll


def exp_post(t, y, t_star, decay, scale, log_noise, asymptote):
    """Exponential decay posterior"""
    fit = asymptote + scale * np.exp(-decay * np.array(t_star))
    # fit = asymptote * np.ones(np.array(t_star).shape)

    return fit, np.zeros((fit.size, fit.size))


def lin_ll(t, y, grad, offset, log_noise):
    """Linear log likelihood"""
    fit = offset + grad * np.array(t)
    residuals = y - fit

    ll = -0.5 * np.sum(residuals * residuals) / np.exp(log_noise)

    ll -= log_noise

    return ll


def lin_post(t, y, t_star, grad, offset, log_noise):
    """Linear decay posterior"""
    fit = offset + grad * np.array(t_star)

    return fit, np.zeros((fit.size, fit.size))


if __name__ == "__main__":
    # t = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    # y = [1.01, 2.02, 3.01, 4.02, 5.01, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    t = [0.5161519999999999, 1.0442660000000001, 1.5786660000000001, 2.120883, 2.663332, 3.1997910000000003, 3.728019, 4.240011999999999, 4.720604, 5.102015, 5.718002, 6.329281]
    y = [0.86295700031475897, 0.86313937339512437, 0.86339091654661948, 0.86314174086762296, 0.86320711553360607, 0.86343798926948101, 0.86321143542880985, 0.86326081536386767, 0.86338694205417421, 0.86342758473838543, 0.86353182772260073, 0.86360189759536021]
    xx = [-0.044717167581553585, 1.0209133493188212, -3.3370671968699268]

    bounds = [[-np.inf, np.inf],
              [-np.inf, np.inf],
              [np.log(0.0000001), np.log(1)]]
    logdist = lambda xx: lin_ll(t, y,
                                grad=xx[0], offset=xx[1], log_noise=xx[2])
    for _ in range(1000):
        xx = slice_sample_bounded_max(1, 1, logdist, xx, 0.5, False, 10, bounds)[0]
        print(xx, logdist(xx))