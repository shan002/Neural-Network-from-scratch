Derivation of differentiation of $\sigma(x)$  
$$
\sigma(x) = \frac{1}{1 + \exp(-x)}\\
\sigma\prime(x) = \frac{exp(-x)}{(1 + \exp(-x))^2}\\
= \frac{1}{1 + \exp(-x)}.\frac{exp(-x)}{1 + \exp(-x)}\\
= \frac{1}{1 + \exp(-x)}.(1 - \frac{1}{1 + \exp(-x)})\\
= \sigma(x)(1 - \sigma(x))
$$

Now,  
$$
z = np.dot(W^T, X)\\
a = \sigma(z)\\
L = -yln(a) - (1-y)ln(1-a)\\
\frac{\partial L}{\partial a} = -\frac{y}{a} + \frac{1-y}{1-a}\\
$$

So,  
$$
\frac{\partial L}{\partial z} = \frac{\partial L}{\partial a}\times\frac{\partial a}{\partial z}\\
= \frac{\partial L}{\partial z} = \frac{\partial L}{\partial a}\times\frac{\partial \sigma(z)}{\partial z}\\
= (-\frac{y}{a} + \frac{1-y}{1-a})\sigma(z)(1 - \sigma(z))\\
= (-\frac{y}{a} + \frac{1-y}{1-a})a(1 - a)\\
= -y(1-a) + a(1-y)\\
= a - y\\
$$
