# Using AI to Create Synthetic Data

An ever-increasing number of companies rely on significant amounts of data to help guide their business decisions. This analytical process, however, can sometimes contain highly sensitive information such as Personally Identifiable Information (PII), making its use particularly problematic.
 
Due to these privacy constraints which are further bound by regulatory frameworks (e.g. GDPR in the EU), SIX Banking Services is investigating new solutions to “anonymize” its data as an initial step in the analytical process. This is where Artificial Intelligence (AI) can be used to create synthetic data.
 
Thanks to various Machine Learning (ML) models, it is possible to create synthetic tabular data that: (1) keep sensitive information private and non-identifiable, (2) maintain statistical properties of the original data. One of the main deep learning models which the project group retained was CT-GAN, which is a Generative Adversarial Model (GAN).

After generating the synthetic data, the team evaluated the performance of various models through the lens of 3 pillars: Resemblance, Utility, and Privacy. The first two pillars pertain to the synthetic data’s similarity to the real data while the last pillar concerns privacy.
  
Although they observed a general trade-off between Utility and Privacy, i.e. maximizing Utility translates to lower Privacy and vice versa, the usage of GANs models, especially with Differential Privacy (DP) can be a viable solution to generating synthetic data while protecting sensitive data.
