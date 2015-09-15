# automl-phase-2

1st place submission to the AutoML competition - phase 2

A reduced implementation of [freeze-thaw Bayesian optimization](http://arxiv.org/abs/1406.389) extended to choose computations in the context of ensemble construction via stacking. Base estimators include most things in SKLearn.

Code is mixed quality. We made an architectural choice of message passing early on - it has some nice properties but makes it difficult to understand and debug.
