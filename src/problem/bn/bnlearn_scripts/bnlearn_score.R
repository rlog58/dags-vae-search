# Usage: Rscript script.R <dataset_name> <adjacency_matrix> <metric_name> <output_file>

# Load necessary libraries and dataset
library("bnlearn", quietly = TRUE, warn.conflicts = FALSE)

# Function to convert string adjacency matrix to matrix
read_adjacency_matrix <- function(adj_str, vertex_names) {
  # Split the string into lines and then into individual elements
  matrix_lines <- strsplit(adj_str, " ")[[1]]
  adj <- do.call(rbind, lapply(matrix_lines, function(line) as.numeric(strsplit(line, "")[[1]])))

  return(adj)
}

# Main execution block
if (!interactive()) {
  # Read command-line arguments
  args <- commandArgs(trailingOnly = TRUE)

  dataset_name <- args[1]
  metric_name <- args[2]
  adjacency_matrix_str <- args[3]

  # Load the specified dataset
  data(list = dataset_name)
  dataset <- get(dataset_name)

  # Create an empty graph with the same node names as the dataset
  net <- empty.graph(names(dataset))

  # Read the adjacency matrix from the provided string
  adj <- read_adjacency_matrix(adjacency_matrix_str, names(dataset))

  # Set the adjacency matrix of the graph
  amat(net) <- adj

  # Compute and print the BIC score
  score_value <- score(net, dataset, type = metric_name)

  cat(sprintf("%.12f", score_value), "\n")
}