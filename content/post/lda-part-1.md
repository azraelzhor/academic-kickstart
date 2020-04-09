---
date: 2020-03-20
title: "Variational Expectation Maximization for Latent Dirichlet Allocation"
tags: ["lda", "topic modeling", "em", "variational inference"]
---

Text data is everywhere. When having massive amounts of them, a need naturally arises is that we want them to be organized efficiently. One naive way is to organize them based on topics. It means that texts covering the same topics should be put in the same groups. The problem is that we do not know which topic a text document belongs to and manually labeling topics for every document is a very expensive task. Hence, topic modeling comes as a way to automatically discover abstract topics contained in these text documents.

<img src="https://i.imgur.com/sC2oNEc.png" width="300"/>
<p align="center">Figure _. Text documents</p>

One of the most common topic models is Latent Dirichlet Allocation (LDA), was introduced long time ago (D. Blei et al, 2003) but is still powerful now. LDA is a complex, hierarchical latent variable model with some probabilistic assumptions over it. Thus, before diving into detail of LDA, let us review some basic knowledges about `probability` and `latent variable model`.

>**Note**: My blog on LDA contains two parts. This is the first part about theoretical understanding of LDA. The  second part involves a basic implementation of LDA, which you can check out here.

## Probabilistic assumptions
### Categorical distribution
A categorical distribution is a `discrete` probability distribution which describes the possibility that one random variable belongs to one of $K$ categories. The distribution is parameterized by a $K$-dimensional vector $p$ denoting probabilities assigned to each category.

![](https://i.imgur.com/EDPbqzv.png)


For example, assume that we have 4 categories and $p = [0.4, 0.2, 0.3, 0.1]$, then we have
$$
p(x=i) = p_i
$$

### Dirichlet distribution
![](https://miro.medium.com/max/1400/1*Pepqn_v-WZC9iJXtyA-tQQ.png)
A Dirichlet distribution is a `continuous` probability distribution which describes the possibility of generating $(K-1)$-simplex. It is parameterized by a positive, $K$-dimensional vector $\alpha$.

$$
p(x;\alpha) = \frac{1}{B(\alpha)} \prod_{k=1}^{K}x_k^{\alpha_i - 1}
$$

where $x$ is a $K$-dimensional vector and $B(\cdot)$ denotes Beta function.

## Latent variable model
A latent variable model assumes that data, which we can observe, is controlled by some underlying unknown factor we can not observe. This dependency is often parameterized by a known distribution $p(\cdot)$ with its associated parameters known as model parameter. Formally, a simple latent variable model consists of three parts: observed data $x$, latent variable $z$ that controls $x$ and model parameters $\theta$ like the picture below.
<img src="https://i.imgur.com/bT71bXf.png" width="300"/>
<p align="center">Figure _. A typical latent variable model</p>

[TODO - example of latent variable model]
For an example of latent variable models, imagine that you are an observer at a casino, where people are playing dice game...

Latent variables increases our model's expressiveness (meaning our model can represents more complex data) but there's no such thing as a free lunch. Typically, there are two main problems associated with latent variable models that need to be solved
* The first one is **learning** in which we try to find the "optimal" parameters $\color{blue}{\theta^*}$ based on some criterion. One powerful technique for learning is `maximum likelihood estimation` preferring to chose the parameters that maximize the likelihood $p(x;\theta)$. Maximum likelihood estimation in latent variable models is difficult. Then comes a method that will be introduced in the next section, named `Expectation Maximization`.

<!-- When $z$ is discrete, we have
$$
p(x; \theta) = \sum_z p(x, z; \theta)
$$ when $z$ is discrete or
$$
p(x; \theta) = \int_z p(x, z; \theta) dz
$$
when $z$ is continuous. -->

<!-- \begin{align}
& \mathbf{E}_{x \sim p_{data}(x)}{p(x; \theta)} \\
& = \frac{1}{N} \prod_i^N p(x^{(i)}; \theta) \\
& = \frac{1}{N} \prod_i^N  \int p(x^{(i)}, z^{(i)}; \theta) dz^{(i)} \\
& = \frac{1}{N} \prod_i^N \int p(x^{(i)} | z^{(i)}; \theta) p(z^{(i)})  dz^{(i)} \\
\end{align} -->

* In many cases, latent variables can capture meaningful pattern in the data. Hence, given new data, we are often interested in the value of latent variables. This raises the problem of **inference** where we want to deduce the posterior $p(x|z;\theta)$.
$$
p(x|z;\theta) = \frac{p(x, z ;\theta)}{p(x;\theta)} = \frac{p(x, z ;\theta)}{}
$$
A method to approximate the posterior distribution, named `Variational Inference`, will be introduced later.

### Expectation Maximization (EM)
Introducing latent variables to a statistical model makes its likelihood function non-convex, which is hard to find a maximum likelihood solution. The EM algorithm was introduced to solve the maximum likelihood estimation problem in these kind of statistical models. The algorithm iteratively alternates between building an expected log-likelihood (`E step`), which is a convex lower bound to the non-convex log-likelihood, and maximizing it over parameters (`M step`).

But how does EM construct the expected log-likelihood?
We have
<div>
\begin{align}
\log p(x; \theta) & \geq \log p(x;\theta) - KL({\color{blue}{q(z)}}||p(z|x;\theta)) \\
& = \log p(x;\theta) - (\mathrm{E}_{z\sim \color{blue}{q(z)}}\log {\color{blue}{q(z)}} - \mathrm{E}_{z \sim \color{blue}{q(z)}}\log p(z|x; \theta)) \\
& = \mathrm{E}_{z\sim \color{blue}{q(z)}}(\log p(x;\theta) + \log p(z|x;\theta)) - \mathrm{E}_{z\sim \color{blue}{q{(z)}}}\log {\color{blue}{q(z)}} \\
& = \mathrm{E}_{z\sim \color{blue}{q(z)}}\log p(x, z;\theta) - \mathrm{E}_{z\sim \color{blue}{q(z)}}\log {\color{blue}{q(z)}} = L(q, \theta) \tag{1} \\
\end{align}
</div>

for any choice of $\color{blue}{q(z)}$. It is obvious that $L(q, \theta)$ is a lower bound of $\log p(x;\theta)$ and the equality holds if and only if ${\color{blue}{q(z)}} = p(z|x;\theta)$. EM aims to construct a lower bound that is easy to maximize. By initializing parameter $\theta_{old}$ and choosing $\color{blue}{q(z)} = p(z|x;\theta_{old})$ (`E-step`), the lower bound becomes
$$
L(\theta) = \mathrm{E}_{z\sim p(z|x;\theta_{old})} \log p(x,z;\theta) - \mathrm{E}_{z\sim {p(z|x;\theta_{old})}}\log {p(z|x;\theta_{old})}
$$

EM then maximizes $L(\theta)$ at each `M-step`

<div>
\begin{align}
\mathop{max}_{\theta} L(\theta) & = \mathop{max}_{\theta} \mathrm{E}_{z\sim p(z|x;\theta_{old})} \log p(x,z;\theta) - \mathrm{E}_{z\sim {p(z|x;\theta_{old})}}\log {p(z|x;\theta_{old})} \\
& = \mathop{max}_{\theta} \mathrm{E}_{z\sim p(z|x;\theta_{old})} \log p(x,z;\theta) \\
\end{align}
</div>

<!-- $L(q, \theta)$ is still involved two unknown components which is hard to optimize. EM deals with this problem by initializing parameter $\theta_{old}$ and construct the lower bound by choosing $\color{blue}{q(z)} = p(z|x;\theta_{old})$. The lower bound becomes
$$
L(\theta) = \mathrm{E}_{z\sim p(z|x;\theta_{old})} \log p(x,z;\theta) - \mathrm{E}_{z\sim {p(z|x;\theta_{old})}}\log {p(z|x;\theta_{old})}
$$ -->

The EM algorithm can be summarized as follows
* Initialize parameter $\theta = \theta^{(0)}$
* For each loop $t$ start from $0$
    <img src="https://i.imgur.com/PbhzDrF.png" width="400"/>
    * Estimate the posterior $p(z|x; \color{blue}{\theta^{(t)}})$
    * Maximize the expected log-likelihood $\mathop{max}_{\color{red}{\theta^{(t+1)}}}$
    * If the convergence standard is satisfied, stop
    <!-- * Maximize the expected log-likelihood $\mathop{max}_{\color{red}{\theta^{(t+1)}}}\mathrm{E}_{z\sim p(z|x ; \color{blue}{\theta^{(t)}})}{p(x, z ;{\color{red}{\theta^{(t+1)}}}})$ -->

>**Note**: It is easy to notice that the EM algorithm can only be applied if the posterior distribution can be computed analytically, given the current parameter $\color{blue}{\theta^{(t)}}$.

If you want to go into the details of EM, **Gaussian Mixture** (when $z$ is discrete) and **Probabilistic Principal Component Analysis** (when $z$ is continuous) are the two perfect cases to study.

### Variational Inference
<!-- The EM is used with the assumption that we can keep track of the posterior. -->

In many of the cases, the posterior distribution $p(z|x;\theta)$ that we are interested in can not be inferred analytically, or in other words, it is intractable. This leads naturally to the field of `approximate inference`, in which we try to approximate the intractable posterior. Variational inference is such a technique in approximate inference which is fast and effective enough for a good approximation of $p(z|x;\theta)$.

The idea of variational inference is simple that we reformulate the problem of inference as an optimization problem by
* First, posit a variational family $\color{blue}{q(z;v)}$ controlled by variational parameter $v$
* Then, find the optimal $\color{blue}{q(z;v^{\*})}$ in this family, which is as "close" to $p(z|x;\theta)$ as possible

<img src="https://i.imgur.com/zXCukxg.png" width="400"/>
<p align="center">Figure _. Mean field approximation</p>

Specifically, the goal of variational inference is then to minimize the $KL$ divergence between the variational family and the true posterior: $\mathop{min}_{q}KL({\color{blue}{q(z;v)}}||p(z|x;\theta))$. But how can we minimize such an intractable term?
<!-- But how can we minimize a term that can not be evaluated analytically? -->

Recall from (1) (with the variational distribution $\color{blue}{q(z;v)}$  being chosen as $\color{blue}{q(z)}$) we have the ELBO

<div>
\begin{align}
& \log p(x;\theta) - KL({\color{blue}{q(z;v)}}||p(z|x;\theta))\\
& = \mathrm{E}_{z\sim \color{blue}{q(z;v)}}\log p(x, z;\theta) - \mathrm{E}_{z\sim \color{blue}{q(z;v)}}\log \color{blue}{q(z;v)}\\
\end{align}
</div>

Since $\log p(x;\theta)$ is considered as constant, minimizing the KL divergence is equivalent to maximizing the ELBO. The optimization problem becomes

<div>
$$
\mathop{max}_{q}\mathrm{E}_{z\sim \color{blue}{q(z;v)}}\log p(x, z;\theta) - \mathrm{E}_{z\sim \color{blue}{q(z;v)}}\log \color{blue}{q(z;v)}
$$
</div>

which now can be optimized with suitable choice of $\color{blue}{q(z;v)}$.

>**Note**: Relationship between EM and VI

#### Mean field approximation
[TODO]There are many form of variational inference. The simplest form of variational inference is `mean-field approximation`, which makes the strong assumption that all latent variables are mutually independent. The variational distribution can be factorized as
$$
q(z;v) = \prod_{k=1}^{K}q(z_k;v_k)
$$

where $z$ consists of $K$ latent variables $(z_1, z_2, ..., z_K)$. Each latent variable $z_k$ now is controlled by its own variational parameter $v_k$.

[TODO] $q(z_k;v_k^*) \propto \mathrm{E}_{z_{-k} \sim q_{-k}(z_{-k};{v_{-k}}^{(t)})} \log p(z_k, z_{-k}, x;\theta)$

#### Coordinate ascent update

The coordinate ascent algorithm can be summarized as follows
* Initialize $v = v^{(0)}$
* For each loop $t$ start from $0$
    * For each loop  $k$ from $1$ to $K$
        * Estimate $q(z_k;v_k^*) \propto \mathrm{E}_{z_{-k} \sim q_{-k}(z_{-k};{v_{-k}}^{(t)})} \log p(z_k, z_{-k}, x;\theta)$
        * Set $v_k^{(t+1)} = v_k^*$
    * Compute the ELBO to check convergence

## Latent Dirichlet Allocation
LDA assumes that each `document` is a distribution over `topics` and each `topic` is considered as a distribution over `words`. For instance, suppose that we have $4$ topics and a total of $6$ `words`.

<img src="https://i.imgur.com/ek3asj1.png" width="500"/>
<p align="center">Figure _. The two probabilistic assumptions of LDA</p>

### Generative process
LDA is a generative model. Hence, to understand its structure clearly, let us see how it generates documents. Suppose that we have $T$ topics and a vocabulary of $V$ words. Model LDA has 2 parameters $(\alpha, \beta)$ where
* $\alpha$ denotes the Dirichlet prior that controls topic distribution of each document.
* $\beta$ is a $2D$ matrix of size $T \times V$ denotes word distribution of  all topics ($\beta_i$ is a word distribution of the `i + 1`th topic).

<img src="https://i.imgur.com/LGAlXEg.png" width="500"/>
<p align="center">Figure _. How a document is generated in LDA</p>

The generative process is then pictured as above. Specifically,
* For each document with $N_d$ words
    * Sample document's topic distribution $\theta \sim Dir(\alpha)$
    * For each word positions $j$ from $1$ to $N_d$
        * Sample the topic of the current word $t_j \sim Cat(\theta)$
    * Sample the current word based on the topic $t_j$ and the word distribution parameters $\beta$, $w_j \sim Cat(\beta_{t_j})$

>**Warning**: $\theta$ now is a latent variable. I keep the notation the same as the original paper for consistency.

### The two problems of LDA
<img src="https://i.imgur.com/VUMrTKJ.png" width="400"/>
<p align="center">Figure _. LDA as a probabilistic graphical model</p>

LDA is a latent variable model, consisting of: observed data $w$;  model parameters $\alpha, \beta$; and latent variables $z, \theta$; as shown in the figure above. Hence, just like any typical latent variable model, LDA also have two problems needed to be solved.

<!--
$$
p(w;\alpha, \beta) = \int p(\theta;\alpha) \prod_{i=1}^{N_d} \sum_{t=0}^{T - 1} p(z_i = t|\theta) p(w_i | \beta, z_i=t) d\theta
$$ -->

#### Inference
Given a document $d$ has $N$ words $\{w_1^{(d)}, ..., w_N^{(d)}\}$ and model parameters $\alpha$, $\beta$; infer the posterior distribution $p(z, \theta| w^{(d)}; \alpha, \beta)$.

We can then use mean field approximation to approximate $p(z, \theta| w^{(d)}; \alpha, \beta)$ by introducing the mean-field variational distribution $q(z, \theta; \lambda, \phi) = q(\theta;\lambda)\prod_{i=1}^{N}q(z_i;\phi_i)$.
<img src="https://i.imgur.com/O5a0mY3.png" width="250"/>
<p>Figure _. The mean-field variational distribution</p>


#### Parameter estimation
Likelihood function for each document is
$$
p(w;\alpha, \beta) = \int p(\theta;\alpha) \prod_{i=1}^{N_d} \sum_{t=0}^{T - 1} p(z_i = t|\theta) p(w_i | \beta, z_i=t) d\theta
$$

Given a collection of $D$ documents, find $\alpha, \beta$ that maximizes the likelihood function over all documents

$$
p(D;\alpha, \beta) = \prod_{d}^{D} \int p(\theta_d;\alpha) \prod_{i=1}^{N_d} \sum_{t=0}^{T - 1} p(z_i = t|\theta_d) p(w_i | \beta, z_i=t) d\theta_d
$$

Since the posterior $p(z, \theta| w^{(d)}; \alpha, \beta)$ can not be computed exactly but can only be approximated (for instance, via variational inference in the previous section), we can not apply the EM algorithm directly to solve the estimation problem. To handle this, an algorithm named variational EM algorithm, which combines EM and mean-field inference, was introduced.

The variational EM algorithm for LDA can be summarized as follows
* Initialize parameter $\alpha, \beta$ to $\alpha^{(0)}, \beta^{(0)}$
* For each loop $t$ start from $0$
    * **E step**:
        * For each document $d$, use mean-field approximation to approximate the posterior $p(z^{(d)}, \theta^{(d)}|w^{(d)};{\color{blue}{\alpha^{(t)}, \beta^{(t)}}})$:
            * Introduce the mean-field $q(z, \theta; \lambda, \phi) = q(\theta;\lambda)\prod_{i=1}^{N}q(z_i;\phi_i)$
            * Use coordinate ascent update algorithm to yield optimal $\lambda^*, \phi^*$
    * **M step**: Maximize the expected log-likelihood
        <p>$\mathop{max}_{\color{red}{\alpha^{(t+1)}, \beta^{(t+1)}}} \mathrm{E}_{z, \theta \sim q(z, \theta; \lambda^{*}, \phi^{*})} {p(w, z, \theta ;{\color{red}{\alpha^{(t+1)}, \beta^{(t+1)}}}})$</p>
    * If the convergence standard is satisfied, stop

>**Note**: Actually, there are many techniques to solve the 2 problems of LDA. In the scope of this blog, we only discuss about Variational EM.

__References__

1. Latent Dirichlet Allocation ([pdf](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf))
