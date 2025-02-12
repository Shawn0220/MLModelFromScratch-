import json
import math
import matplotlib.pyplot as plt
import networkx as nx
import os

JSON_FILE = 'grid_data.json'

def load_grid_data_json(filename=JSON_FILE):
    """
    Load the grid data from a JSON file.
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(f"JSON file '{filename}' does not exist.")
    with open(filename, 'r') as f:
        grid_data = json.load(f)
    print(f"Grid data loaded from JSON file '{filename}'.")
    
    # Convert lists back to tuples where necessary
    for key, value in grid_data.items():
        if key in ['intersections', 'normalized_intersections']:
            for sub_key, sub_value in value.items():
                grid_data[key][sub_key] = tuple(sub_value)
    return grid_data

def find_overlapping_nodes(coordinates):
    """
    Identify nodes that have overlapping coordinates.
    """

    coord_to_nodes = {}
    for node, coord in coordinates.items():
        coord_key = (round(coord[0], 6), round(coord[1], 6))  # Round to 6 decimal places
        if coord_key not in coord_to_nodes:
            coord_to_nodes[coord_key] = []
        coord_to_nodes[coord_key].append(node)
    overlapping_nodes = {coord: nodes for coord, nodes in coord_to_nodes.items() if len(nodes) > 1}
    return overlapping_nodes

def adjust_overlapping_coordinates(coordinates):
    """
    Adjust coordinates of overlapping nodes to make them visually distinct.
    """

    adjusted_coords = coordinates.copy()
    overlap_groups = find_overlapping_nodes(coordinates)
    for nodes in overlap_groups.values():
        num_nodes = len(nodes)
        for i, node in enumerate(nodes):
            angle = (2 * math.pi / num_nodes) * i
            dx = 0.05 * math.cos(angle)  # Increased shift
            dy = 0.05 * math.sin(angle)  # Increased shift
            adjusted_coords[node] = (coordinates[node][0] + dx, coordinates[node][1] + dy)
    return adjusted_coords

# -------------------------------------------
# Visualization
# -------------------------------------------

def visualize_traversal(visited, time_map, coordinates, edge_list, title, path=None, dis_map=None, end_node=None, start_node=None):
    """
    Visualize the traversal of search algorithms on the graph.

    Args:
        visited (list): List of nodes in the order they were visited.
        time_map (dict): Adjacency list with distances.
        coordinates (dict): Mapping of node names to their coordinates.
        edge_list (list): List of tuples representing edges with properties.
        title (str): Title of the visualization.
        path (list, optional): The final path found by the search algorithm.
        dis_map (dict, optional): Distance map for heuristic values (used in A*).
        end_node (str, optional): The end node for heuristic calculations.
        start_node (str, optional): The start node for labeling.
    """

    # Adjust coordinates if needed
    overlapping_nodes = find_overlapping_nodes(coordinates)
    if overlapping_nodes:
        print("Overlapping nodes detected. Adjusting coordinates for visualization.")
        adjusted_coordinates = adjust_overlapping_coordinates(coordinates)
    else:
        adjusted_coordinates = coordinates

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes with positions based on adjusted coordinates
    for node, coord in adjusted_coordinates.items():
        G.add_node(node, pos=coord)

    # Add edges with properties from edge_list
    for from_node, to_node, properties in edge_list:
        bidirectional = properties['bidirectional']
        if from_node == to_node:
            continue  # Avoid loops
        if bidirectional:
            G.add_edge(from_node, to_node)
            G.add_edge(to_node, from_node)
        else:
            G.add_edge(from_node, to_node)

    # Get positions for the nodes
    pos = nx.get_node_attributes(G, 'pos')

    # Set up the matplotlib figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    # EDITABLE 2: Modify this line to toggle the window between full-screen and non-full-screen (default)
    # plt.get_current_fig_manager().full_screen_toggle()

    # Turn off the axis
    plt.axis('off')

    # Initialize node colors and sizes
    node_colors = {node: 'lightgray' for node in adjusted_coordinates.keys()}
    node_sizes = {node: 50 for node in adjusted_coordinates.keys()}  # Default size

    # Initialize labels for nodes (optional)
    labels = {node: node for node in adjusted_coordinates.keys()}

    # Enhance Start and End nodes' appearance
    if start_node:
        node_colors[start_node] = 'green'  # Start node colored green
        node_sizes[start_node] = 150       # Larger size for Start node
    if end_node:
        node_colors[end_node] = 'red'      # End node colored red
        node_sizes[end_node] = 150         # Larger size for End node

    # Initialize the stop flag
    stop_visualization = [False]

    # Define the event handler
    def on_key(event):
        if event.key.lower() == 'q':
            print("Visualization interrupted by user.")
            stop_visualization[0] = True
            plt.close('all')  # Close all open plot windows immediately

    # Connect the event handler to the figure
    cid = fig.canvas.mpl_connect('key_press_event', on_key)

    # Visualization of the traversal step by step
    previous_node = None  # To keep track of the previous node for resizing
    for i in range(len(visited)):
        ax.clear()  # Clear the axes for the next step

        # Reset node sizes to default
        node_sizes_step = {node: 50 for node in adjusted_coordinates.keys()}

        # Draw 'Start' and 'End' labels prominently
        if start_node:
            plt.text(
                pos[start_node][0], pos[start_node][1] + 0.02,
                "Start",
                fontsize=8,
                ha='center',
                va='bottom',
                color='green',
                fontweight='bold'
            )
        if end_node:
            plt.text(
                pos[end_node][0], pos[end_node][1] + 0.02,
                "End",
                fontsize=8,
                ha='center',
                va='bottom',
                color='red',
                fontweight='bold'
            )

        # Update the node colors
        # Nodes that have been visited are colored black
        for node in visited[:i]:
            if node != start_node and node != end_node:
                node_colors[node] = 'black'

        # Current node being visited is colored light red (frontier)
        current_node = visited[i]
        if current_node != start_node and current_node != end_node:
            node_colors[current_node] = 'lightcoral'
            node_sizes_step[current_node] = 200  # Larger size for the current node

        # If there was a previous node, revert its size back to default
        if previous_node and previous_node != current_node:
            if previous_node != start_node and previous_node != end_node:
                node_sizes_step[previous_node] = 50
        previous_node = current_node

        # Draw the nodes with updated colors and sizes
        node_color_list = [node_colors[node] for node in G.nodes()]
        node_size_list = [node_sizes_step[node] for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=node_size_list, node_color=node_color_list, ax=ax)

        # Draw bidirectional edges without arrows
        bidirectional_edges = [(u, v) for u, v in G.edges() if G.has_edge(v, u) and u < v]
        nx.draw_networkx_edges(G, pos, edgelist=bidirectional_edges, edge_color='black', arrows=False, ax=ax)

        # Draw unidirectional edges with arrows
        unidirectional_edges = [(u, v) for u, v in G.edges() if not G.has_edge(v, u)]
        nx.draw_networkx_edges(
            G, pos,
            edgelist=unidirectional_edges,
            edge_color='black',
            arrows=True,
            arrowstyle='->',
            arrowsize=10,
            ax=ax
        )

        # # Draw labels below the nodes
        # # Adjust the vertical alignment to place labels below
        # nx.draw_networkx_labels(
        #     G, pos,
        #     labels=labels,
        #     font_size=3,
        #     font_color='black',
        #     verticalalignment='top',  # Align text below the node
        #     ax=ax
        # )

        # Update the plot title with the current step
        ax.set_title(f"{title} - Step {i + 1}/{len(visited)}", fontsize=12)

        # Display the current node being visited at the bottom
        plt.text(0.5, 0.02, f"Current node: {current_node}", ha='center', va='bottom', fontsize=9, transform=fig.transFigure)

        # Check if the stop flag is set
        if stop_visualization[0]:
            print("Stopping visualization as per user request.")
            break  # Exit the loop

        # EDITABLE 1: Modify this line to adjust the time between search steps in visualization
        plt.pause(0.01)  # Pause to update the visualization

    # After traversal, highlight the path (for 10 seconds, by default) if provided
    if path:
        ax.clear()  # Clear the axes for the final path visualization

        # Reset node colors to default before highlighting the path
        node_colors_final = {node: 'lightgray' for node in adjusted_coordinates.keys()}
        node_sizes_final = {node: 50 for node in adjusted_coordinates.keys()}

        # Re-apply enhanced Start and End nodes' appearance
        if start_node:
            node_colors_final[start_node] = 'green'
            node_sizes_final[start_node] = 150
        if end_node:
            node_colors_final[end_node] = 'red'
            node_sizes_final[end_node] = 150

        for node in path:
            if node != start_node and node != end_node:
                node_colors_final[node] = 'orange'  # Path nodes colored orange
                node_sizes_final[node] = 100        # Slightly larger size for path nodes

        # Draw the nodes with updated colors and sizes
        node_color_list_final = [node_colors_final[node] for node in G.nodes()]
        node_size_list_final = [node_sizes_final[node] for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=node_size_list_final, node_color=node_color_list_final, ax=ax)

        # Draw bidirectional edges without arrows
        nx.draw_networkx_edges(G, pos, edgelist=bidirectional_edges, edge_color='black', arrows=False, ax=ax)

        # Draw unidirectional edges with arrows
        nx.draw_networkx_edges(
            G, pos,
            edgelist=unidirectional_edges,
            edge_color='black',
            arrows=True,
            arrowstyle='->',
            arrowsize=10,
            ax=ax
        )

        # Draw the path with thick dotted arrows
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(
            G, pos,
            edgelist=path_edges,
            edge_color='orange',
            style='dotted',
            width=2,
            arrows=True,
            arrowstyle='-|>',
            arrowsize=15,
            ax=ax
        )

        # Re-apply 'Start' and 'End' labels prominently
        if start_node:
            plt.text(
                pos[start_node][0], pos[start_node][1] + 0.02,
                "Start",
                fontsize=8,
                ha='center',
                va='bottom',
                color='green',
                fontweight='bold'
            )
        if end_node:
            plt.text(
                pos[end_node][0], pos[end_node][1] + 0.02,
                "End",
                fontsize=8,
                ha='center',
                va='bottom',
                color='red',
                fontweight='bold'
            )

        # Draw arrows along the path to indicate direction
        for from_node, to_node in path_edges:
            ax.annotate(
                '',
                xy=(pos[to_node][0], pos[to_node][1]),
                xytext=(pos[from_node][0], pos[from_node][1]),
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5)
            )

        # Update the plot title
        ax.set_title(f"{title} - Path Found", fontsize=12)

        # Add text at the bottom showing the path
        plt.text(0.5, 0.02, f"Path: {' -> '.join(path)}", ha='center', va='bottom', fontsize=4, transform=fig.transFigure)

        plt.pause(10)  # Pause to show the final visualization (Adjust as needed)
