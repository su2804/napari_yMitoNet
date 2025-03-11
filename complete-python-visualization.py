#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Napari plugin for analyzing 3D skeletal structures using the Nellie library,
with integrated adjacency list graph visualization using Python.
"""
#%% Libraries 
import re
import numpy as np
import pandas as pd
from tifffile import imread
import os
import tempfile
import csv
import warnings
import matplotlib.pyplot as plt
import networkx as nx
from scipy.ndimage import label as labell
from qtpy.QtWidgets import QLabel, QScrollArea
from qtpy.QtGui import QPixmap, QImage
from qtpy.QtCore import Qt

try:
    from nellie.im_info.im_info import ImInfo
    from nellie.segmentation.filtering import Filter
    from nellie.segmentation.labelling import Label
    from nellie.segmentation.networking import Network
    NELLIE_AVAILABLE = True
except ImportError:
    NELLIE_AVAILABLE = False
    warnings.warn("Nellie library not found. Some functionality will be limited.")

import napari
from napari.utils.notifications import show_info, show_warning, show_error
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QCheckBox, QSpinBox, QDoubleSpinBox, QFormLayout,
    QFileDialog, QComboBox, QGroupBox, QTextEdit,
    QMessageBox
)
from skimage import io
from magicgui import magic_factory, magicgui
from qtpy.QtCore import Qt

#%% Global state
class AppState:
    def __init__(self):
        self.loaded_folder = None
        self.folder_type = "Single TIFF"
        self.nellie_output_path = None
        self.raw_layer = None
        self.skeleton_layer = None
        self.points_layer = None
        self.node_path = None
        self.node_dataframe = None
        self.temp_dir = tempfile.mkdtemp()
        self.adjacency_path = None
        self.graph_image_path = os.path.join(self.temp_dir, "graph_output.png")

# Initialize global state
app_state = AppState()

#%% Utility functions
def get_float_pos(st):
    """Parse string representation of position to get coordinates."""
    st = re.split(r'[ \[\]]', st)
    pos = [int(element) for element in st if element != '']
    return pos

def get_float_pos_comma(st):
    """Parse string representation of position to get coordinates."""
    st = re.split(r'[ \[\,\]]', st)
    pos = [int(element) for element in st if element != '']
    return pos

def visualize_adjacency_graph_python(adjacency_path, output_path, scale_factor=1.0):
    """Visualize a graph from an adjacency list CSV using pure Python.
    
    Args:
        adjacency_path (str): Path to adjacency list CSV file
        output_path (str): Path where the output image will be saved
        scale_factor (float): Scale factor to apply to the graph
        
    Returns:
        bool: True if visualization succeeded, False otherwise
    """
    try:
        # Check if file exists
        if not os.path.exists(adjacency_path):
            return False
            
        # Read the adjacency list CSV data
        adjacency_data = pd.read_csv(adjacency_path)
        
        # Check if the CSV has the expected columns
        expected_columns = ["component_num", "node", "pos_x", "pos_y", "pos_z", "adjacencies"]
        if not all(col in adjacency_data.columns for col in expected_columns):
            missing_cols = [col for col in expected_columns if col not in adjacency_data.columns]
            print(f"Error: CSV is missing expected columns: {', '.join(missing_cols)}")
            return False
        
        # Create an empty multigraph
        G = nx.MultiGraph()
        
        # Parse adjacency string to vector of node IDs
        def parse_adjacencies(adj_str):
            # Handle empty adjacencies
            if adj_str == "[]" or pd.isna(adj_str) or adj_str == "":
                return []
            
            # Remove brackets and split by comma
            clean_str = adj_str.strip('[]')
            # Handle cases with no commas
            if ',' not in clean_str:
                if clean_str.strip() and clean_str.strip().isdigit():
                    return [int(clean_str)]
                else:
                    return []
            
            # Split by comma and convert to numeric
            try:
                adj_vec = [int(x.strip()) for x in clean_str.split(',')]
                return adj_vec
            except ValueError:
                return []
        
        # Add all nodes first with positions
        for _, row in adjacency_data.iterrows():
            G.add_node(row['node'], 
                      pos_x=row['pos_x'], 
                      pos_y=row['pos_y'], 
                      pos_z=row['pos_z'],
                      component=row['component_num'])
        
        # Create an edge table to track all connections
        edge_table = []
        
        # Collect all edges
        for _, row in adjacency_data.iterrows():
            node_id = row['node']
            adj_list = parse_adjacencies(row['adjacencies'])
            
            # Add all edges to the edge table
            for neighbor in adj_list:
                edge_table.append((node_id, neighbor))
        
        # Add edges to the graph
        for source, target in edge_table:
            G.add_edge(source, target)
        
        # Calculate node degrees
        node_degrees = dict(G.degree())
        
        # Prepare node colors: degree 1 as blue (endpoints), degree 3+ as red (junctions), others as lightblue
        node_colors = []
        for node in G.nodes():
            if node_degrees[node] == 1:
                node_colors.append('blue')  # Endpoint nodes
            elif node_degrees[node] >= 3:
                node_colors.append('red')   # Junction nodes
            else:
                node_colors.append('lightblue')  # Other nodes
        
        # Calculate edge colors and widths
        # Count the number of parallel edges
        edge_count = {}
        for u, v, k in G.edges(keys=True):
            edge_key = tuple(sorted([u, v]))
            if edge_key not in edge_count:
                edge_count[edge_key] = 0
            edge_count[edge_key] += 1
        
        # Create curved edges for multigraph visualization
        edge_curves = {}
        for edge_key, count in edge_count.items():
            if count > 1:
                max_curve = 0.7 * min(1, np.sqrt(count) / 10)
                if count % 2 == 0:  # Even number
                    curves = np.linspace(-max_curve, max_curve, count)
                else:  # Odd number
                    curves = np.linspace(-max_curve, max_curve, count)
                edge_curves[edge_key] = curves
        
        # Set up plot
        plt.figure(figsize=(13.33, 10), dpi=300)  # High resolution
        
        # Apply scaling to the layout
        pos = nx.spring_layout(G, 
                              seed=42,
                              iterations=1000, 
                              k=1.0/np.sqrt(len(G.nodes())) * scale_factor,
                              scale=2.0)
        
        # Draw nodes
        node_size = max(5, 15 / scale_factor)  # Smaller nodes for larger graphs
        nx.draw_networkx_nodes(G, pos, 
                              node_color=node_colors, 
                              node_size=node_size*25,  # Adjust for visual size
                              edgecolors='black',
                              linewidths=1.5)
        
        # Draw edges with appropriate curvature for parallel edges
        edge_idx = {}
        edge_width = max(0.5, 1.5/scale_factor)
        
        for u, v, k in G.edges(keys=True):
            edge_key = tuple(sorted([u, v]))
            if edge_key not in edge_idx:
                edge_idx[edge_key] = 0
            
            # Get or compute curvature
            if edge_key in edge_curves and edge_count[edge_key] > 1:
                curves = edge_curves[edge_key]
                curve = curves[edge_idx[edge_key]]
                edge_idx[edge_key] += 1
            else:
                curve = 0
            
            # Draw single edge with appropriate curvature
            nx.draw_networkx_edges(G, pos, 
                                 edgelist=[(u, v)], 
                                 width=edge_width, 
                                 edge_color='gray40',
                                 connectionstyle=f'arc3, rad={curve}')
        
        # Add title
        plt.title(f"Multigraph from {os.path.basename(adjacency_path)}")
        
        # Add summary statistics
        unique_edges = len(edge_count)
        multiple_edges = sum(1 for count in edge_count.values() if count > 1)
        plt.figtext(0.5, 0.01, 
                   f"Total nodes: {len(G.nodes())} - Unique edges: {unique_edges} - " + 
                   f"Total connections: {len(G.edges())} - " + 
                   f"Edges with multiple connections: {multiple_edges} - " + 
                   f"Scale: {scale_factor}x", 
                   ha="center", fontsize=12)
        
        # Add a legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Endpoints (deg 1)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Junctions (deg 3+)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10, label='Other Nodes')
        ]
        plt.legend(handles=legend_elements, loc='upper right', 
                  frameon=True, framealpha=1, facecolor='white', edgecolor='black', 
                  fontsize=12)
        
        # Remove axis
        plt.axis('off')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print some stats
        print(f"Graph visualization saved to: {output_path}")
        print(f"Unique edges: {unique_edges}")
        print(f"Total connections: {len(G.edges())}")
        print(f"Edges with multiple connections: {multiple_edges}")
        
        # Edge multiplicity distribution
        edge_mult_table = {}
        for count in edge_count.values():
            if count not in edge_mult_table:
                edge_mult_table[count] = 0
            edge_mult_table[count] += 1
        
        print("Edge multiplicity distribution:")
        for count, freq in sorted(edge_mult_table.items()):
            print(f"  {count} connection(s) between same nodes: {freq} occurrence(s)")
        
        return True
        
    except Exception as e:
        import traceback
        print(f"Error visualizing graph: {str(e)}")
        traceback.print_exc()
        return False

def visualize_adjacency_graph(adjacency_path, output_path, scale_factor=1.0):
    """Visualize a graph from an adjacency list CSV using Python.
    
    Args:
        adjacency_path (str): Path to adjacency list CSV file
        output_path (str): Path where the output image will be saved
        scale_factor (float): Scale factor to apply to the graph (default: 1.0)
        
    Returns:
        bool: True if visualization succeeded, False otherwise
    """
    if not adjacency_path or not os.path.exists(adjacency_path):
        show_error("Adjacency list CSV file does not exist")
        return False
        
    try:
        # Call our Python visualization function
        result = visualize_adjacency_graph_python(adjacency_path, output_path, scale_factor)
        
        # Check if image was created
        if not os.path.exists(output_path):
            show_error("Graph visualization failed to generate an image")
            return False
            
        show_info(f"Graph visualization saved to: {output_path}")
        return result
        
    except Exception as e:
        show_error(f"Error visualizing graph: {str(e)}")
        return False

def run_nellie_processing(im_path, num_t=None, remove_edges=False, ch=0):
    """Run Nellie processing pipeline on an image."""
    if not NELLIE_AVAILABLE:
        show_error("Nellie library is required for processing. Please install it first.")
        return None
    
    try:
        # Initialize ImInfo with the image
        im_info = ImInfo(im_path, ch=ch)
        
        # Set dimension sizes
        im_info.dim_sizes = {'Z': 0.30, 'Y': 0.17, 'X': 0.17, 'T': 0}
        show_info(f"Dimension sizes set: {im_info.dim_sizes}")
        
        # Filtering step
        preprocessing = Filter(im_info, num_t, remove_edges=remove_edges)
        preprocessing.run()
        show_info("Filtering complete")
        
        # Segmentation step
        segmenting = Label(im_info, num_t)
        segmenting.run()
        show_info("Segmentation complete")
        
        # Network analysis
        networking = Network(im_info, num_t)
        networking.run()
        show_info("Networking complete")
        
        return im_info
    
    except Exception as e:
        show_error(f"Error in Nellie processing: {str(e)}")
        return None

def get_network(pixel_class_path):
    """Generate network representation from a skeleton image."""
    try:
        # Define output file paths
        base_name = os.path.basename(pixel_class_path).split(".")[0]
        save_name = f"{base_name}_adjacency_list.csv"
        save_path = os.path.join(os.path.dirname(pixel_class_path), save_name)
        
        edge_name = f"{base_name}_edge_list.txt"
        edge_path = os.path.join(os.path.dirname(pixel_class_path), edge_name)
        
        # Store adjacency path in app state for later use
        app_state.adjacency_path = save_path
        
        # Load the skeleton image
        skeleton = imread(pixel_class_path)
        skeleton = np.transpose(skeleton)
        show_info(f"Skeleton shape: {np.shape(skeleton)}")
        
        # Define 3D connectivity structure
        struct = np.ones((3, 3, 3))
        
        # Extract tree structures
        trees, num_trees = labell(skeleton > 0, structure=struct)
        show_info(f"Found {num_trees} tree structures")
        
        # Convert tips and lone-tips to nodes (all nodes will have value 4)
        skeleton[skeleton == 2] = 4  # Tips
        skeleton[skeleton == 1] = 4  # Lone-tips
        
        # Extract edges (all voxels except nodes)
        no_nodes = np.where(skeleton == 4, 0, skeleton)
        edges, num_edges = labell(no_nodes > 0, structure=struct)
        show_info(f"Found {num_edges} edges")
        
        # Extract nodes
        nodes = np.where(skeleton == 4, 4, 0)
        node_labels, num_nodes = labell(nodes > 0, structure=struct)
        show_info(f"Found {num_nodes} nodes")
        
        # Map nodes to their connected edges
        node_edges = {}
        node_positions = {}
        
        # For each node, find connected edges
        for j_id in range(1, num_nodes + 1):
            # Get coordinates of all voxels in this node
            j_coords = np.argwhere(node_labels == j_id)
            
            # Track edges connected to this node
            connected_edges = set()
            
            if len(j_coords) > 0:
                # Take the first voxel's coordinates
                x, y, z = j_coords[0]
                node_positions[j_id] = (x, y, z)
            else:
                # Fallback if node has no voxels (shouldn't happen)
                node_positions[j_id] = (0, 0, 0)
            
            # Check 3x3x3 neighborhood around each node voxel
            for (x, y, z) in j_coords:
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for dz in [-1, 0, 1]:
                            # Skip the center voxel
                            if dx == 0 and dy == 0 and dz == 0:
                                continue
                                
                            # Neighbor coordinates
                            xx, yy, zz = x + dx, y + dy, z + dz
                            
                            # Check bounds
                            if (0 <= xx < skeleton.shape[0] and
                                0 <= yy < skeleton.shape[1] and
                                0 <= zz < skeleton.shape[2]):
                                
                                # If neighbor is part of an edge, add to connected edges
                                edge_label = edges[xx, yy, zz]
                                if edge_label != 0:
                                    connected_edges.add(edge_label)
            
            # Store edges connected to this node
            node_edges[j_id] = connected_edges
        
        # Map edges to connected nodes
        edge_nodes = {}
        for n_id, e_set in node_edges.items():
            for e_id in e_set:
                if e_id not in edge_nodes:
                    edge_nodes[e_id] = set()
                edge_nodes[e_id].add(n_id)
        
        # Create network graph
        G = nx.MultiGraph()
        
        # Add all nodes to graph
        for j_id in range(1, num_nodes + 1):
            x, y, z = node_positions[j_id]
            G.add_node(j_id, pos_x=x, pos_y=y, pos_z=z)
        
        # Add edges between nodes
        for e_id, connected_nodes in edge_nodes.items():
            cn = list(connected_nodes)
            
            if len(cn) == 2:
                # Standard edge between two nodes
                n1, n2 = cn
                G.add_edge(n1, n2, edge_id=e_id)
            elif len(cn) == 1:
                # Self-loop (edge connects to same node)
                (n1,) = cn
                G.add_edge(n1, n1, edge_id=e_id)
            elif len(cn) > 2:
                # Edge connects multiple nodes - add edges between all pairs
                for i in range(len(cn)):
                    for j in range(i + 1, len(cn)):
                        G.add_edge(cn[i], cn[j], edge_id=e_id)
        
        # Find connected components (separate trees)
        components = list(nx.connected_components(G))
        show_info(f"Found {len(components)} connected components")
        
        # Write adjacency list to CSV
        with open(save_path, "w", newline="") as f:
            writer = csv.writer(f)
            # Header row
            writer.writerow(["component_num", "node", "pos_x", "pos_y", "pos_z", "adjacencies"])
            
            # Write each component
            for comp_num, comp in enumerate(components, start=1):
                # Create subgraph for this component
                subG = G.subgraph(comp).copy()
                
                # For each node, write its adjacencies
                for node in sorted(subG.nodes()):
                    # Get node attributes (positions)
                    pos_x = subG.nodes[node]['pos_x']
                    pos_y = subG.nodes[node]['pos_y']
                    pos_z = subG.nodes[node]['pos_z']
                    
                    adjacencies = sorted(list(subG[node]))
                    writer.writerow([comp_num, node, pos_x, pos_y, pos_z, adjacencies])
        
        # Write edge list
        nx.write_edgelist(G, edge_path)
        
        show_info(f"Network analysis complete. Files saved to:\n- {save_path}\n- {edge_path}")
        return save_path, edge_path
        
    except Exception as e:
        show_error(f"Error generating network: {str(e)}")
        return None, None

def load_image_and_skeleton(nellie_output_path):
    """Load raw image and skeleton from Nellie output directory.
    
    Args:
        nellie_output_path (str): Path to Nellie output directory
        
    Returns:
        tuple: (raw_image, skeleton_image, face_colors, positions, colors)
    """
    try:
        # Find relevant files in the output directory
        tif_files = os.listdir(nellie_output_path)
        
        # Find raw image file (channel 0)
        raw_files = [f for f in tif_files if f.endswith('-ch0-ome.ome.tif')]
        if not raw_files:
            show_error("No raw image file found in the output directory")
            return None, None, [], [], []
            
        raw_file = raw_files[0]
        basename = raw_file.split(".")[0]
        print('Basename is: '+basename)
        
        # Find skeleton image file
        skel_files = [f for f in tif_files if f.endswith('-ch0-im_pixel_class.ome.tif')]
        if not skel_files:
            show_error("No skeleton file found in the output directory")
            return None, None, [], [], []
            
        skel_file = skel_files[0]
        
        # Get full paths
        raw_im_path = os.path.join(nellie_output_path, raw_file)
        skel_im_path = os.path.join(nellie_output_path, skel_file)
        
        # Check for node data file
        node_path_extracted = os.path.join(nellie_output_path, f"{basename}_extracted.csv")
        adjacency_path = os.path.join(nellie_output_path, f"{basename}_adjacency_list.csv")
        app_state.node_path = node_path_extracted
        
        # Load images
        raw_im = imread(raw_im_path)
        skel_im = imread(skel_im_path)
        skel_im = np.transpose(np.nonzero(skel_im))
        
        # Default all points to red
        face_color_arr = ['red' for _ in range(len(skel_im))]
        
        #Check if an adjaceny list exists and convert to extracted csv if so
        if os.path.exists(adjacency_path) and not os.path.exists(node_path_extracted):
            adjacency_to_extracted(node_path_extracted,adjacency_path)
        
        if os.path.exists(adjacency_path) and os.path.exists(node_path_extracted):
            node_df = pd.read_csv(node_path_extracted)
            app_state.node_dataframe = node_df            
            if node_df.empty or pd.isna(node_df.index.max()):
                adjacency_to_extracted(node_path_extracted,adjacency_path)
        
        # Process extracted nodes if available
        if os.path.exists(node_path_extracted):
            node_df = pd.read_csv(node_path_extracted)
            app_state.node_dataframe = node_df
            
            if not node_df.empty and not pd.isna(node_df.index.max()):
                # Extract node positions and degrees
                pos_extracted = node_df['Position(ZXY)'].values
                show_info(f"Extracted positions: {pos_extracted}")
                
                deg_extracted = node_df['Degree of Node'].values.astype(int)
                positions = [get_float_pos_comma(el) for el in pos_extracted]
                print(positions)
                # Generate colors based on node degree
                colors = []
                for i, degree in enumerate(deg_extracted):
                    if degree == 1:
                        colors.append('blue')  # Endpoint nodes
                    else:
                        colors.append('green')  # Junction nodes
                        
                return raw_im, skel_im, face_color_arr, positions, colors
                
            else:
                # Create empty dataframe if no data
                app_state.node_dataframe = pd.DataFrame(columns=['Degree of Node', 'Position(ZXY)'])
                app_state.node_dataframe.to_csv(node_path_extracted, index=False)
                return raw_im, skel_im, face_color_arr, [], []
        else:
            # Create new node file if none exists
            app_state.node_dataframe = pd.DataFrame(columns=['Degree of Node', 'Position(ZXY)'])
            app_state.node_dataframe.to_csv(node_path_extracted, index=False)
            return raw_im, skel_im, face_color_arr, [], []
            
    except Exception as e:
        show_error(f"Error loading image and skeleton: {str(e)}")
        return None, None, [], [], []

def adjacency_to_extracted(extracted_csv_path, adjacency_path):
    """Convert adjacency list to extracted node data."""
    adj_df = pd.read_csv(adjacency_path)
    if os.path.exists(extracted_csv_path):
        ext_df = pd.read_csv(extracted_csv_path)
    else:
        ext_df = {}
        
    adjs_list = adj_df['adjacencies'].tolist()
    deg_nd_i = []
    deg_nd = []
    
    for el in adjs_list:
        elf = get_float_pos_comma(el)
        deg_nd_i.append(len(elf))
        if (len(elf) > 0):
            deg_nd.append(len(elf))
        
    pos_x = adj_df['pos_x'].tolist()
    pos_y = adj_df['pos_y'].tolist()
    pos_z = adj_df['pos_z'].tolist()

    pos_zxy = [[pos_z[i_n], pos_y[i_n], pos_x[i_n]] for i_n, i in enumerate(deg_nd_i) if i > 0]    
    
    ext_df['Degree of Node'] = deg_nd
    ext_df['Position(ZXY)'] = pos_zxy
    
    ext_df = pd.DataFrame.from_dict(ext_df)
    
    print(ext_df)
    
    ext_df.to_csv(extracted_csv_path, index=False)

#%% GUI Widgets
class FileLoaderWidget(QWidget):
    """Widget for loading image files and setting processing options."""
    
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.setup_ui()
        
    def setup_ui(self):
        """Create the user interface."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Title
        title = QLabel("Nellie Network Analysis")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)
        
        # File selection section
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout()
        file_group.setLayout(file_layout)
        
        # File path display and browse button
        path_layout = QHBoxLayout()
        self.path_label = QLabel("No file selected")
        path_layout.addWidget(self.path_label)
        
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self.on_browse_clicked)
        path_layout.addWidget(self.browse_btn)
        file_layout.addLayout(path_layout)
        
        # File type selection
        type_layout = QHBoxLayout()
        type_label = QLabel("File Type:")
        type_layout.addWidget(type_label)
        
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Single TIFF", "Time Series"])
        type_layout.addWidget(self.type_combo)
        file_layout.addLayout(type_layout)
        
        layout.addWidget(file_group)
        
        # Processing options section
        proc_group = QGroupBox("Processing Options")
        proc_layout = QFormLayout()
        proc_group.setLayout(proc_layout)
        
        # Channel selection
        self.channel_spin = QSpinBox()
        self.channel_spin.setRange(0, 10)
        self.channel_spin.setValue(0)
        proc_layout.addRow("Channel Number:", self.channel_spin)
        
        # Remove edges option
        self.remove_edges_check = QCheckBox()
        self.remove_edges_check.setChecked(False)
        proc_layout.addRow("Remove Edge Artifacts:", self.remove_edges_check)
        
        layout.addWidget(proc_group)
        
        # Buttons section
        button_layout = QHBoxLayout()
        
        self.process_btn = QPushButton("Run Nellie Processing")
        self.process_btn.clicked.connect(self.on_process_clicked)
        self.process_btn.setEnabled(False)
        button_layout.addWidget(self.process_btn)
        
        self.view_btn = QPushButton("View Results")
        self.view_btn.clicked.connect(self.on_view_clicked)
        self.view_btn.setEnabled(False)
        button_layout.addWidget(self.view_btn)
        
        layout.addLayout(button_layout)
        
        # Graph Visualization Options group
        graph_options_group = QGroupBox("Graph Visualization Options")
        graph_options_layout = QFormLayout()
        graph_options_group.setLayout(graph_options_layout)
        
        # Scale factor setting
        self.scale_factor_spin = QDoubleSpinBox()
        self.scale_factor_spin.setRange(0.1, 10.0)
        self.scale_factor_spin.setValue(1.0)
        self.scale_factor_spin.setSingleStep(0.1)
        self.scale_factor_spin.setDe