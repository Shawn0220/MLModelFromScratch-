expand_count = 0

def expand(node, _map):
	"""

	Args:
		node (str): Name of node to expand
		_map (dict): Map where every node is a dictionary key, and every value is an inner dictionary whose keys are
		the children of that node
	"""

	global expand_count
	expand_count = expand_count + 1
	return [next for next in _map[node] if _map[node][next] is not None]
