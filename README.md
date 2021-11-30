# ML-tools
ML tools that we use internally and which you may find useful too.


## Pretty Print Named Entities
```python
pretty_print_ne("On April 1, 1976, Apple Computer Company was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne.",
                 "O B-DATE I-DATE I-DATE B-ORG I-ORG I-ORG O O O B-PER I-PER B-PER I-PER O B-PER I-PER")

```

```
   _____DATE_____ _________ORG__________                ____PER____ _____PER______     _____PER_____
On April 1, 1976, Apple Computer Company was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne.

```