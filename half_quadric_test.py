#!env python
import numpy as np
import numpy.fft
import scipy.misc
import scipy.ndimage
import matplotlib.pyplot as plt

def make_D_eye(h, w):
    return np.eye(h*w)

def test_naive_inverse_problem_with_noise(img, make_D_func=make_D_eye):
    N = np.product(img.shape)
    h, w = img.shape

    # simple blur observation matrix
    A = np.diag([0.5]*N)
    for y in range(h):
        for x in range(w):
            i = y*w + x
            if x > 0:     A[i, y*w + (x - 1)] = 0.125
            if x < w - 1: A[i, y*w + (x + 1)] = 0.125
            if y > 0:     A[i, (y - 1)*w + x] = 0.125
            if y < h - 1: A[i, (y + 1)*w + x] = 0.125

    # convolution kernel
    Ak = np.zeros((h, w))
    Ak[+0, +0] = 0.5
    Ak[-1, +0] = 0.125
    Ak[+1, +0] = 0.125
    Ak[+0, -1] = 0.125
    Ak[+0, +1] = 0.125

    def solve2d(A, b):
        return np.linalg.solve(A, b.ravel()).reshape(b.shape)

    def solve2dfft(k, b, r=0.001):
        return np.real(np.fft.ifft2(np.fft.fft2(b) / (np.fft.fft2(k) + r)))

    # blur, recover
    img_blur = np.dot(A, img.ravel()).reshape(img.shape)
    img_recover = solve2d(A, img_blur)

    # noise, blur, recover
    sigma = 0.02
    img_blur_noise = img_blur + np.random.randn(*img.shape)*sigma
    img_blur_noise_recover = solve2d(A, img_blur_noise)

    # Tikhnov regularization: minimize 1/2 ||Ax-y||_2^2 + beta*||Dx||_2^2
    # => (A'A + beta D'D)x = A'y
    beta = 0.01
    D = make_D_func(h, w)
    img_blur_noise_regularization_recover = solve2d(np.dot(A.T, A) + beta*np.dot(D.T, D), np.dot(A.T, img_blur_noise.ravel()).reshape(img.shape))

    fig, axs = plt.subplots(2, 5, figsize=(13, 5))
    ax = axs[0, 0]; ax.imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='nearest'); ax.set_title('org')
    ax = axs[0, 1]; ax.imshow(img_blur, cmap='gray', vmin=0, vmax=1, interpolation='nearest'); ax.set_title('blur')
    ax = axs[0, 2]; ax.imshow(img_recover, cmap='gray', vmin=0, vmax=1, interpolation='nearest'); ax.set_title('blur+deconv')
    ax = axs[0, 3]; ax.imshow(np.abs(img_recover - img), cmap='cool', vmin=-1, vmax=1, interpolation='nearest'); ax.set_title('diff')
    fig.delaxes(axs[0, 4])
    ax = axs[1, 0]; ax.imshow(img_blur_noise, cmap='gray', vmin=0, vmax=1, interpolation='nearest'); ax.set_title(r'blur+noise($\sigma={:.2e}$)'.format(sigma))
    ax = axs[1, 1]; ax.imshow(img_blur_noise_recover, cmap='gray', vmin=0, vmax=1, interpolation='nearest'); ax.set_title('blur+noise+recover')
    ax = axs[1, 2]; ax.imshow(np.abs(img_blur_noise_recover - img), cmap='cool', vmin=-1, vmax=1, interpolation='nearest'); ax.set_title('diff')
    ax = axs[1, 3]; ax.imshow(img_blur_noise_regularization_recover, cmap='gray', vmin=0, vmax=1, interpolation='nearest'); ax.set_title(r'Tikhnov($\beta={:.2e}$)'.format(beta))
    ax = axs[1, 4]; ax.imshow(np.abs(img_blur_noise_regularization_recover - img), cmap='cool', vmin=-1, vmax=1, interpolation='nearest'); ax.set_title('Tikhnov diff')

if __name__=='__main__':
    # prepare image
    img = scipy.misc.lena()
    img = scipy.ndimage.zoom(img, 1.0/16.0) / 255.0
    print(img.shape)
    print(img.min(), img.max())

    test_naive_inverse_problem_with_noise(img, make_D_eye)
    plt.show()
