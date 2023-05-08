# Differential Equation Solver for Plasma Field using Finite Difference Method and Neural Networks
## Neural Network for Approximating a Function
This code implements a neural network for approximating a function using PyTorch.

The neural network uses the Softplus activation function and consists of three hidden layers with N * 12, N * 4, and N neurons, respectively.

The model is trained on data generated by solving a differential equation using the finite difference method with the coefficients determined by a given set of parameters.

### Requirements
This code requires the following libraries:

- os
- random
- numpy
- torch
- torch.nn
- torch.optim
- sklearn.model_selection
- torch.utils.data

# Electromagnetic field model in high-frequency plasmatron
## To model the electrical parameters of a high-frequency plasma torch, we write down the Maxwell equation. The complex form of the Maxwell equation in one dimension is given by
<p style="text-align: center;">
$$\Large\frac{d\dot{H}}{dr}=(\sigma + i\epsilon_0\epsilon)\dot{E}$$
</p>
### Applying the complex amplitude method to this:
<p style="text-align: center;">
$$\Large\dot{H}=H\exp(iωt+ψ_H)$$
</p>
Getting the equation:
<p style="text-align: center;">
$$\Large\frac{1}{r}\frac{d}{dr}(\frac{r}{σ}\frac{dH^2}{dr})=2σE^2$$
</p>
Boundary conditions are presented in the following form:
<p style="text-align: center;">
$$\Large\frac{dH^2}{dr}(0)=0$$
$$\LargeH^2(R)=H_R^2$$
</p>
Let us write the equation in a general form:
<p style="text-align: center;">
$$\Large\frac{d}{dr}(rV\frac{dy}{dr})=rf$$
</p>
And we will develop a numerical scheme for it. We will approximate the coefficients $$r$$ and $$V$$ at points shifted by $$h/2$$ to achieve an accuracy of $$O(h^2)$$ for the difference scheme. We will write the equation in the difference form:
<p style="text-align: center;">
$$\Large r_i^{-}V_i^{-}y_{i-1}-(r_i^{-}V_i^{-}+r_i^{+}V_i^{+})y_i+r_i^{+}V_i^{+}y_{i+1}=-(-r_if_ih^2)$$
</p>
Where
<p style="text-align: center;">
$$\Large r_i^{-}=r_i-\frac{h}{2}$$
</p>
<p style="text-align: center;">
$$\Large r_i^{+}=r_i+\frac{h}{2}$$
</p>
