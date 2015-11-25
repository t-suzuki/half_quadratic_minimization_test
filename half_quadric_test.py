#!env python
# reference:
#  - http://www.math.hkbu.edu.hk/genearoundtheworld/pdf/rChan.pdf
#  - http://www.math.cuhk.edu.hk/~rchan/paper/halfquad.pdf
#  - http://mnikolova.perso.math.cnrs.fr/hq.pdf
import numpy as np
import numpy.fft
import scipy.misc
import scipy.ndimage
import matplotlib
import matplotlib.pyplot as plt

def plot_img(ax, img_target, title):
    ax.imshow(img_target, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    ax.set_title(title)

def plot_diff(ax, img_target, img, title, err):
    ax.imshow(np.abs(img_target - img), cmap='cool', vmin=-1, vmax=1, interpolation='nearest')
    ax.set_title('{} diff (err={:.2e})'.format(title, err))

def make_blur_kernel(h, w):
    # simple blur observation matrix.
    # y = Ax .. observation process.
    #  y: observed image
    #  x: latent image
    A = np.diag([0.5]*(h*w))
    for y in range(h):
        for x in range(w):
            i = y*w + x
            if x > 0:     A[i, y*w + (x - 1)] = 0.125
            if x < w - 1: A[i, y*w + (x + 1)] = 0.125
            if y > 0:     A[i, (y - 1)*w + x] = 0.125
            if y < h - 1: A[i, (y + 1)*w + x] = 0.125
    return A

def solve2d(A, y):
    return np.linalg.solve(A, y.ravel()).reshape(y.shape)

def recover_tikhnov_regularization(A, y, beta, D):
    # Tikhnov regularization: minimize 1/2 ||Ax-y||_2^2 + beta*||Dx||_2^2
    # => (A'A + beta D'D)x = A'y
    #  D: regularization transform. (e.g. identity)
    return solve2d(np.dot(A.T, A) + beta*np.dot(D.T, D), np.dot(A.T, y.ravel()).reshape(y.shape))

def recover2d_half_quadratic(A, y, beta, Ds, dphi2_0, dphi, n_iter=10):
    # solve the nonlinear minimization problem:
    #  min_{x} J(x) = min_{x} {||Ax-y||_2^2 + beta sum_i(phi(||D_i x||_2))},
    # by minimizing the auxiliary (introducing b) functional
    #  J*(x, b) = ||Ax-y||_2^2 + beta sum_i(b_i ||D_i x||_2^2 + psi(b_i)),
    #  psi(b) = sup_{t <- R+}(-1/2 b t^2 + phi(t)),
    # with respect to x and b.
    x = y # start from observation ~ latent approximation.
    for i in range(n_iter):
        # 1. b := argmin_b J*(x, b) given x. (J*(b; x) is elementwise and has closed form)
        bs = []
        for D in Ds:
            t = np.linalg.norm(np.dot(D, x))
            if t < 1.0e-5:
                b = dphi2_0 # phi''(0+)
            else:
                b = dphi(t)/t
            bs.append(b)
        # 2. x := argmin_x J*(x, b) given b. (J*(x; b) is quadratic)
        H = 2.0*np.dot(A.T, A)
        for b, D in zip(bs, Ds):
            H += beta*b*np.dot(D.T, D)
        x = np.linalg.solve(H, 2.0*np.dot(A.T, y))
    return x

def test_naive_inverse_problem_with_noise(img, img_blur, img_blur_noise, A, D):
    N = np.product(img.shape)
    h, w = img.shape

    def mae(img_target):
        return np.abs(img_target - img).mean()

    # blur+recover
    img_recover = solve2d(A, img_blur)
    print('Blur recovery error: {}'.format(mae(img_recover)))

    # noise+blur, recover
    img_blur_noise_recover = solve2d(A, img_blur_noise)
    print('Blur+Noise recovery error: {}'.format(mae(img_blur_noise_recover)))

    # Tikhnov regularization: minimize 1/2 ||Ax-y||_2^2 + beta*||Dx||_2^2
    # => (A'A + beta D'D)x = A'y
    #  D: regularization transform. (e.g. identity)
    beta = 0.01
    img_blur_noise_regularization_recover = recover_tikhnov_regularization(A, img_blur_noise, beta, D)
    print('Tikhnov error: {}'.format(mae(img_blur_noise_regularization_recover)))

    fig, axs = plt.subplots(2, 5, figsize=(13, 5))
    plot_img(axs[0, 0], img, 'org')
    plot_img(axs[0, 1], img_blur, 'blur')
    plot_img(axs[0, 2], img_recover, 'blur+deconv')
    plot_diff(axs[0, 3], img_recover, img, 'deconv', mae(img_recover))
    fig.delaxes(axs[0, 4])
    plot_img(axs[1, 0], img_blur_noise, r'blur+noise($\sigma={:.2e}$, err={:.2e})'.format(sigma, mae(img_blur_noise)))
    plot_img(axs[1, 1], img_blur_noise_recover, 'blur+noise+recover')
    plot_diff(axs[1, 2], img_blur_noise_recover, img, 'blur', mae(img_blur_noise_recover))
    plot_img(axs[1, 3], img_blur_noise_regularization_recover, r'Tikhnov($\beta={:.2e}$)'.format(beta))
    plot_diff(axs[1, 4], img_blur_noise_regularization_recover, img, 'Tikhnov', mae(img_blur_noise_regularization_recover))

def test_half_quadratic_minimization(img, img_blur, img_blur_noise, A):
    N = np.product(img.shape)
    h, w = img.shape

    def mae(img_target):
        return np.abs(img_target - img).mean()

    # blur+recover
    img_recover = solve2d(A, img_blur)
    print('Blur recovery error: {}'.format(mae(img_recover)))

    # noise+blur, recover
    img_blur_noise_recover = solve2d(A, img_blur_noise)
    print('Blur+Noise recovery error: {}'.format(mae(img_blur_noise_recover)))

    # Half-Quadratic minimization.
    # D0 : x finite difference.
    # D1 : y finite difference.
    D0, D1 = np.zeros((N, N)), np.zeros((N, N))
    for y in range(h):
        for x in range(w):
            i = y*w + x
            if x > 0:     D0[i, y*w + (x - 1)] = -1.0
            if x < w - 1: D0[i, y*w + (x + 1)] = +1.0
            if y > 0:     D1[i, (y - 1)*w + x] = -1.0
            if y < h - 1: D1[i, (y + 1)*w + x] = +1.0
    # potential function: phi(t) = |t|
    dphi2_0 = 0.0
    def dphi(t): return np.sign(t)

    beta = 0.3
    img_hq = recover2d_half_quadratic(A, img_blur_noise.ravel(), beta, [D0, D1], dphi2_0, dphi).reshape(img.shape)
    print('Half-Quadratic error: {}'.format(mae(img_hq)))

    fig, axs = plt.subplots(2, 5, figsize=(13, 5))
    plot_img(axs[0, 0], img, 'org')
    plot_img(axs[0, 1], img_blur, 'blur')
    plot_img(axs[0, 2], img_recover, 'blur+deconv')
    plot_diff(axs[0, 3], img_recover, img, 'deconv', mae(img_recover))
    fig.delaxes(axs[0, 4])
    plot_img(axs[1, 0], img_blur_noise, r'blur+noise($\sigma={:.2e}$, err={:.2e})'.format(sigma, mae(img_blur_noise)))
    plot_img(axs[1, 1], img_blur_noise_recover, 'blur+noise+recover')
    plot_diff(axs[1, 2], img_blur_noise_recover, img, 'blur', mae(img_blur_noise_recover))
    plot_img(axs[1, 3], img_hq, r'HQ($\beta={:.2e}$)'.format(beta))
    plot_diff(axs[1, 4], img_hq, img, 'HQ', mae(img_hq))

if __name__=='__main__':
    # prepare image
    img = scipy.misc.lena()
    img = scipy.ndimage.zoom(img, 1.0/16.0) / 255.0
    print(img.shape)
    print(img.min(), img.max())
    h, w = img.shape

    # simple blur observation matrix.
    A = make_blur_kernel(h, w)

    # L2 regularization matrix.
    D = np.eye(h*w)

    # blur
    img_blur = np.dot(A, img.ravel()).reshape(img.shape)

    # noise, blur
    sigma = 0.02
    img_blur_noise = img_blur + np.random.randn(*img.shape)*sigma

    matplotlib.rc('font', size=9)
    test_naive_inverse_problem_with_noise(img, img_blur, img_blur_noise, A, D)
    test_half_quadratic_minimization(img, img_blur, img_blur_noise, A)

    plt.show()
