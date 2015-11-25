# Half Quadratic Minimization test

Solve a nonlinear optimization problem:

    argmin_x ||Ax - y||^2 + sum_i(phi_i(||D_i x||))

by iteratively solving the half-quadratic minimization.

* x: latent variable (R^p)
* y: observed signal (R^d, p=d in image recovery)
* A: observation matrix (R^{d*p}).
* phi: nonlinear potential function for group i.
* D: regularization matrix (R^{s*p}) for group i

# License
Public Domain

# Demo
recover noisy 2D image.

![demo](https://raw.githubusercontent.com/t-suzuki/half_quadratic_minimization_test/master/doc/images/demo_result.png)

# Reference

 * [http://www.math.hkbu.edu.hk/genearoundtheworld/pdf/rChan.pdf](http://www.math.hkbu.edu.hk/genearoundtheworld/pdf/rChan.pdf)
 * [http://www.math.cuhk.edu.hk/~rchan/paper/halfquad.pdf](http://www.math.cuhk.edu.hk/~rchan/paper/halfquad.pdf)

