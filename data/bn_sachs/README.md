# SACHS

## Target dataset generation

Script for generating target.csv:
```python
import random
# pgmpy==0.1.22
from pgmpy.utils import get_example_model

random.seed(42)

get_example_model("sachs") \
  .simulate(5000, seed=42) \
  .to_csv("target.csv", index=False)
```