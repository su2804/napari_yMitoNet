#!/usr/bin/env Rscript

# R script to visualize a graph from an adjacency list CSV file and output as a high-resolution PNG
# with scaling support for integration with image data
# Usage: Rscript adjacency_to_graph.R input.csv output.png [scale_factor]

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
  cat("Usage: Rscript adjacency_to_graph.R input.csv output.png [scale_factor]\n")
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
  
  # Create empty graph
  g <- make_empty_graph(directed = FALSE)
  
  # Add all nodes first
  unique_nodes <- unique(adjacency_data$node)
  for (node_id in unique_nodes) {
    g <- add_vertices(g, 1, name = as.character(node_id))
  }
  
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
  
  # Add edges based on adjacencies
  edge_list <- c()
  for (i in 1:nrow(adjacency_data)) {
    node_id <- adjacency_data$node[i]
    adj_str <- as.character(adjacency_data$adjacencies[i])
    
    # Parse adjacencies
    neighbors <- parse_adjacencies(adj_str)
    
    # Add edges - only add each edge once
    for (neighbor in neighbors) {
      if (node_id < neighbor) {  # This ensures we only add each edge once
        edge_list <- c(edge_list, node_id, neighbor)
      }
    }
  }
  
  # Add all edges to the graph
  if (length(edge_list) > 0) {
    g <- add_edges(g, edge_list)
  }
  
  # Calculate node degrees
  node_degrees <- degree(g)
  
  # Node colors: degree 1 as blue (endpoints), degree 3+ as red (junctions), others as lightblue
  node_colors <- ifelse(node_degrees == 1, "blue", 
                   ifelse(node_degrees >= 3, "red", "lightblue"))
  
  # Define node size based on scale factor (smaller nodes for larger graphs)
  node_size <- max(10, 25 / scale_factor)
  
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
       main = paste("Graph from", basename(input_file)),
       vertex.size = node_size,    # Scaled node size
       vertex.label = NA,        # No labels
       vertex.color = node_colors,
       vertex.frame.color = "black",  # Black borders around vertices
       vertex.frame.width = 1.5,      # Thicker borders
       edge.width = max(1, 2/scale_factor),  # Scaled edge width
       edge.color = "gray40",
       edge.curved = 0.2,
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
  
  # Add title with additional information and scale factor
  title(sub = paste("Total nodes:", vcount(g), 
                   "- Total edges:", ecount(g),
                   "- Scale:", scale_factor, "x"), cex.sub = 1.2)
  
  # Close the device
  dev.off()
  
  cat("High-resolution graph saved to:", output_path, "\n")
  
}, error = function(e) {
  cat("Error processing file:", e$message, "\n")
  quit(status = 1)
})

quit(status = 0)
