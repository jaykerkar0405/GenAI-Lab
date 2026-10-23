# Aim

To detect, quantify and mitigate bias in a machine-learning model and to apply other responsible-AI practices
(explainability, privacy, safety filtering and documentation) that are also essential for generative AI systems.

# Theory

AI systems learn from historical data and can reproduce or amplify its bias, even when the protected attribute $A$
(e.g. gender) is removed, because other features act as **proxies**. *Fairness* is measured per group:

* **Demographic parity:** $P(\hat Y=1\mid A=a)$ equal for all groups; reported as the difference of selection rates and as the
  *disparate-impact ratio* (the "four-fifths rule" asks for a ratio $\ge 0.8$).
* **Equal opportunity / equalised odds:** equal true-positive rates (and false-positive rates) across groups.

Bias mitigation can be applied at three stages:

1. **Pre-processing** - change the data, e.g. *reweighing* with $w(a,y)=\dfrac{P(a)P(y)}{P(a,y)}$.
2. **In-processing** - add a fairness constraint to the learning problem (Agarwal et al.'s *reduction* approach /
   exponentiated gradient).
3. **Post-processing** - adjust the decision thresholds per group (Hardt et al.'s equalised-odds post-processing).

Mitigation usually trades some accuracy for fairness. Further pillars of **Responsible AI**: *explainability* (permutation
importance), *privacy* (the Laplace mechanism gives $\varepsilon$-differential privacy: noise of scale
$\Delta/\varepsilon$), *safety* (filtering PII and toxic content in generated text) and *transparency* (model cards).

# Steps

1. Create a hiring data set with equal merit in both groups, biased historical labels and a proxy feature.
2. Train a baseline logistic regression and compute accuracy, selection rate, TPR, FPR, demographic-parity difference,
   disparate impact and equalised-odds difference with **Fairlearn**.
3. Apply four mitigation strategies: dropping the protected attribute, reweighing, exponentiated gradient (DP and EO
   constraints) and threshold optimisation (DP and EO).
4. Compare all models on fairness and accuracy.
5. Use permutation importance to show the role of the proxy feature.
6. Demonstrate differential privacy with the Laplace mechanism.
7. Implement a PII/toxicity safety filter for generated text and write a model card.

# Output

@@OUTPUT@@

# Conclusion

The baseline model (accuracy 0.758) was clearly biased: group B was selected at 17.7\% versus 47.9\% for group A, giving a
demographic-parity difference of 0.302 and a disparate-impact ratio of 0.369, far below the 0.8 threshold. Simply dropping the
protected attribute only reduced the gap to 0.202 because the proxy feature \texttt{career\_gap} still carries group
information, and a boosted model without the attribute still had a parity difference of 0.178. Reweighing (pre-processing) lowered the
difference to 0.077 (disparate impact 0.795) with accuracy 0.751, while the constrained optimiser and the threshold optimiser with a
demographic-parity constraint reached a difference below 0.01 (disparate impact above 0.97) at an accuracy cost of
about 2.5 points (0.741 and 0.733). The equalised-odds variants reduced the true-positive-rate gap from 0.267 to about 0.04. The
results show the fairness--accuracy trade-off and that different fairness definitions cannot all be satisfied at once. The
Laplace mechanism, the PII/toxicity filter and the model card complete a basic responsible-AI toolkit for both predictive and
generative systems.
