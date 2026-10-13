"""SLM seed-variance audit.

A small, readable toolkit for measuring how much of a published small
language model "improvement" could be explained by the random seed alone.

Modules
-------
config    : load and resolve the YAML config
seeding   : one place that controls every source of randomness
data      : SFT corpus and multiple-choice benchmark loaders
modeling  : model/tokenizer loading plus the model size ladder
evaluate  : length-normalised log-likelihood scoring over prompt templates
train     : fine-tune one model for one seed
claims    : the harvested database of published comparison claims
analysis  : seed sigma, bootstrap intervals, and Seed-Equivalent Significance
plots     : the two paper figures, regenerated from your own results
"""

__version__ = "0.1.0"

__all__ = [
    "analysis",
    "claims",
    "config",
    "data",
    "evaluate",
    "modeling",
    "plots",
    "seeding",
    "train",
]
