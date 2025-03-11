#!/usr/bin/env Rscript

# R script to visualize a multigraph from an adjacency list CSV file and output as a high-resolution PNG
# with scaling support for integration with image data
# Usage: Rscript adjacency_to_multigraph.R input.csv output.png [scale_factor]

# Load required libraries
if (!require(igraph)) {
  install.packages("igraph", repos = "https://cran.rstudio.com/")
}
library(igraph)

# Get command-line arguments
args <- commandArgs(trailingOnly = TRUE)

# Check if we have the right number of arguments
if (length(args) < 2) {
  cat("Error: Need at least two arguments\n")
  cat("Usage: Rscript adjacency_to_multigraph.R input.csv output.png [scale_factor]\n")
  quit(status = 1)
}

# Get input and output paths
input_file <- args[1]
output_path <- args[2]

# Get optional scale factor
scale_factor <- 1.0
if (length(args) >= 3) {
  scale_factor <- as.numeric(args[3])
  if (is.na(scale_factor)) {
    scale_factor <- 1.0
    cat("Warning: Invalid scale factor. Using default (1.0)\n")
  }
}

cat(sprintf("Using scale factor: %f\n", scale_factor))

# Check if input file exists
if (!file.exists(input_file)) {
  cat("Error: Input file does not exist:", input_file, "\n")
  quit(status = 1)
}

# Process the input file
tryCatch({
  # Read the adjacency list CSV data
  adjacency_data <- read.csv(input_file)
  
  # Check if the CSV has the expected columns
  expected_columns <- c("component_num", "node", "pos_x", "pos_y", "pos_z", "adjacencies")
  if (!all(expected_columns %in% colnames(adjacency_data))) {
    missing_cols <- expected_columns[!expected_columns %in% colnames(adjacency_data)]
    cat("Error: CSV is missing expected columns:", paste(missing_cols, collapse=", "), "\n")
    quit(status = 1)
  }
  
  # Instead of using igraph's multigraph support, we'll construct an edge table and create a simplified graph
  # Add all nodes first
  unique_nodes <- unique(adjacency_data$node)
  node_lookup <- data.frame(
    id = unique_nodes,
    name = as.character(unique_nodes),
    stringsAsFactors = FALSE
  )
  
  # Function to parse adjacency string to vector of node IDs
  parse_adjacencies <- function(adj_str) {
    # Handle empty adjacencies
    if (adj_str == "[]" || is.na(adj_str) || adj_str == "") {
      return(numeric(0))
    }
    
    # Remove brackets and split by comma
    clean_str <- gsub("\\[|\\]", "", adj_str)
    # Handle cases with no commas
    if (!grepl(",", clean_str)) {
      if (grepl("\\d+", clean_str)) {
        return(as.numeric(clean_str))
      } else {
        return(numeric(0))
      }
    }
    
    # Split by comma and convert to numeric
    adj_vec <- as.numeric(strsplit(clean_str, ",\\s*")[[1]])
    return(adj_vec)
  }
  
  # Create an edge table to track all edges
  edge_table <- data.frame(
    from = numeric(),
    to = numeric(),
    stringsAsFactors = FALSE
  )
  
  # Collect all edges in the edge table
  for (i in 1:nrow(adjacency_data)) {
    node_id <- adjacency_data$node[i]
    adj_str <- as.character(adjacency_data$adjacencies[i])
    
    # Parse adjacencies
    neighbors <- parse_adjacencies(adj_str)
    
    # Add all edges to the edge table
    for (neighbor in neighbors) {
      edge_table <- rbind(edge_table, data.frame(from = node_id, to = neighbor, stringsAsFactors = FALSE))
    }
  }
  
  # Count the occurrence of each edge (both directions)
  edge_table$edge_key <- ifelse(edge_table$from < edge_table$to, 
                               paste(edge_table$from, edge_table$to, sep = "-"),
                               paste(edge_table$to, edge_table$from, sep = "-"))
  
  edge_counts <- table(edge_table$edge_key)
  edge_table$count <- as.numeric(edge_counts[edge_table$edge_key])
  
  # Calculate edge curvature - each multiple edge gets increasingly more curved
  edge_table$edge_id <- ave(edge_table$edge_key, edge_table$edge_key, FUN = function(x) seq_along(x))
  edge_table$edge_total <- ave(edge_table$edge_key, edge_table$edge_key, FUN = length)
  
  # Calculate curve for each edge
  edge_table$curve <- 0
  for (key in unique(edge_table$edge_key)) {
    subset <- edge_table[edge_table$edge_key == key, ]
    
    if (nrow(subset) > 1) {
      total_edges <- nrow(subset)
      max_curve <- 0.7 * min(1, sqrt(total_edges) / 10)  # Scale curve based on number of edges
      
      # Distribute curves symmetrically around 0
      if (total_edges %% 2 == 0) {  # Even number of edges
        curves <- seq(-max_curve, max_curve, length.out = total_edges)
      } else {  # Odd number of edges
        curves <- seq(-max_curve, max_curve, length.out = total_edges)
      }
      
      edge_table$curve[edge_table$edge_key == key] <- curves
    }
  }
  
  # Create a new graph using vertices from node_lookup
  g <- make_empty_graph(n = nrow(node_lookup), directed = FALSE)
  
  # Set vertex names
  V(g)$name <- as.character(node_lookup$name)
  
  # Add edges - now we include every edge exactly once for visualization
  # but we'll use the curve attribute to separate multiple edges
  edge_index <- 1
  for (i in 1:nrow(edge_table)) {
    g <- add_edges(g, c(
      match(edge_table$from[i], node_lookup$id),
      match(edge_table$to[i], node_lookup$id)
    ))
    E(g)[edge_index]$curve <- edge_table$curve[i]
    edge_index <- edge_index + 1
  }
  
  # Calculate node degrees
  node_degrees <- degree(g)
  
  # Edge width and color settings
  edge_widths <- rep(max(0.5, 1.5/scale_factor), ecount(g))
  edge_colors <- rep("gray40", ecount(g))
  
  # Node colors: degree 1 as blue (endpoints), degree 3+ as red (junctions), others as lightblue
  node_colors <- ifelse(node_degrees == 1, "blue", 
                   ifelse(node_degrees >= 3, "red", "lightblue"))
  
  # Define node size based on scale factor (smaller nodes for larger graphs)
  node_size <- max(5, 15 / scale_factor)
  
  # Create high-resolution image
  # PNG with very high resolution (4000x3000 pixels, 300 dpi)
  png(output_path, width = 4000, height = 3000, res = 300, pointsize = 14)
  
  # Set up a nice color palette for better visualization
  par(bg = "white", mar = c(2, 2, 3, 2))
  
  # Create layout with more iterations for better positioning and apply scale factor
  layout_coords <- layout_with_fr(g, 
                                niter = 10000,      # More iterations for better layout
                                area = vcount(g)^2.3 * scale_factor,  # Adjusted area based on graph size and scale
                                repulserad = vcount(g) * 2 * scale_factor)  # Better spacing
  
  # Apply scale factor directly to layout coordinates
  x_range <- range(layout_coords[,1])
  y_range <- range(layout_coords[,2])
  
  x_center <- mean(x_range)
  y_center <- mean(y_range)
  
  # Rescale around center
  layout_coords[,1] <- (layout_coords[,1] - x_center) * scale_factor + x_center
  layout_coords[,2] <- (layout_coords[,2] - y_center) * scale_factor + y_center
  
  # Plot the graph with enhanced aesthetics
  plot(g, layout = layout_coords,
       main = paste("Multigraph from", basename(input_file)),
       vertex.size = node_size,      # Smaller node size
       vertex.label = NA,            # No labels
       vertex.color = node_colors,
       vertex.frame.color = "black", # Black borders around vertices
       vertex.frame.width = 1.5,     # Thicker borders
       edge.width = edge_widths,     # Fixed edge width
       edge.color = edge_colors,     # Uniform edge color
       edge.curved = TRUE,           # Use the curve attribute we set for each edge
       margin = c(0.2, 0.2, 0.2, 0.2))
  
  # Add a legend for node types
  legend("topright", 
         legend = c("Endpoints (deg 1)", "Junctions (deg 3+)", "Other Nodes"), 
         fill = c("blue", "red", "lightblue"), 
         border = "black",
         cex = 1.5,              # Larger text
         box.lwd = 2,            # Thicker box
         bg = "white",           # White background
         box.col = "black")      # Black border
  
  # Count total unique edges (pairs of nodes)
  unique_edges <- length(unique(edge_table$edge_key))
  multiple_edges <- sum(table(edge_table$edge_key) > 1)
  
  # Add title with additional information and scale factor
  title(sub = paste("Total nodes:", vcount(g), 
                   "- Unique edges:", unique_edges,
                   "- Total connections:", ecount(g),
                   "- Edges with multiple connections:", multiple_edges,
                   "- Scale:", scale_factor, "x"), cex.sub = 1.2)
  
  # Close the device
  dev.off()
  
  cat("High-resolution multigraph saved to:", output_path, "\n")
  cat("Unique edges:", unique_edges, "\n")
  cat("Total connections:", ecount(g), "\n")
  cat("Edges with multiple connections:", multiple_edges, "\n")
  
  # Print the distribution of edge multiplicities
  edge_mult_table <- table(table(edge_table$edge_key))
  cat("Edge multiplicity distribution:\n")
  for (i in 1:length(edge_mult_table)) {
    count <- names(edge_mult_table)[i]
    freq <- edge_mult_table[i]
    cat(sprintf("  %s connection(s) between same nodes: %s occurrence(s)\n", count, freq))
  }
  
}, error = function(e) {
  cat("Error processing file:", e$message, "\n")
  quit(status = 1)
})

quit(status = 0)
