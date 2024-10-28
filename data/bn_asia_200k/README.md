
# ASIA 


## Target dataset generation

Script for getting file:
```R
library("bnlearn", quietly = TRUE, warn.conflicts = FALSE)

dataset <- get("asia")

write.csv(dataset, file = "target.csv")
```