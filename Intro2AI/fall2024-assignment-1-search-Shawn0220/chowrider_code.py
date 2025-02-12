import expand
from collections import deque


def breadth_first_search(time_map, start, end):
    """
    Breadth-first Search

    Args:
        time_map (dict): A map containing travel times between connected nodes (places or intersections), where every
        node is a dictionary key, and every value is an inner dictionary whose keys are the children of that node and
        values are travel times. Travel times are "null" for nodes that are not connected.
        start (str): The name of the node from where to start traversal
        end (str): The name of the node from where to start traversal

    Returns:
        visited (list): A list of visited nodes in the order in which they were visited
        path (list): The final path found by the search algorithm
    """
    # 用于记录已访问的节点
    visited = []
    # 用于存储每个节点的前置节点，以便构建路径
    parent_map = {start: None}
    
    # 使用双端队列来存储待探索的节点
    queue = deque([start])
    
    # 广度优先搜索
    while queue:
        # pop from left to get the node to expand
        current_node = queue.popleft()
        visited.append(current_node)
        
        # 如果找到终点，停止搜索并构建路径
        if current_node == end:
            break

        exp_neighbors = expand.expand(current_node, time_map)
        # print("expand neighbors", exp_neighbors)
        
        neighbors = time_map[current_node]
        # print(neighbors)
        # 遍历当前节点的所有相邻节点
        for neighbor in exp_neighbors:
            if neighbor not in visited and neighbor not in queue:
                queue.append(neighbor)
                # print("expanding ", neighbor)
                parent_map[neighbor] = current_node
        #         parent_map[neighbor] = current_node
        # for neighbor, travel_time in neighbors.items():
        #     if travel_time is not None and travel_time is not "null" and neighbor not in visited and neighbor not in queue:
        #         queue.append(neighbor)
        #         print("expanding ", neighbor)
        #         parent_map[neighbor] = current_node
    
    # 构建从终点到起点的路径
    path = []
    node = end
    while node is not None:
        path.append(node)
        node = parent_map.get(node)
    
    # 反转路径以得到从起点到终点的路径
    path.reverse()
    
    # 如果终点不在路径中，说明没有找到有效路径
    if path[0] != start:
        path = []
    
    return visited, path

# TO DO: Implement Depth-first Search.
def depth_first_search(time_map, start, end):
    """
    Depth-first Search

    Args:
        time_map (dict): A map containing travel times between connected nodes (places or intersections), where every
        node is a dictionary key, and every value is an inner dictionary whose keys are the children of that node and
        values are travel times. Travel times are "null" for nodes that are not connected.
        start (str): The name of the node from where to start traversal
        end (str): The name of the node from where to start traversal

    Returns:
        visited (list): A list of visited nodes in the order in which they were visited
        path (list): The final path found by the search algorithm
    """
    # print("==========================================")
    # 用于记录已访问的节点
    visited = []
    # 用于存储每个节点的前置节点，以便构建路径
    parent_map = {start: None}

    path = []
    
    stack = [start]
    
    # 深度优先搜索
    while stack:
        # pop from left to get the node to expand
        current_node = stack.pop()

        visited.append(current_node)
        
        # 如果找到终点，停止搜索并构建路径
        if current_node == end:
            break
        # print("popped ", current_node)
        neighbors = time_map[current_node]
        # print("neigbors: ", neighbors)

        # expand.expand(start, time_map)
        exp_neighbors = expand.expand(current_node, time_map)

        to_add = []
        # 遍历当前节点的所有相邻节点
        print(exp_neighbors)
        for neighbor in exp_neighbors:
            if neighbor not in visited:
                stack.append(neighbor)
                # print("expanding ", neighbor)
                parent_map[neighbor] = current_node
        # print('\n')
        # print(neighbors)
        # for neighbor, travel_time in neighbors.items():
        #     if travel_time is not None and travel_time is not "null" and neighbor not in visited:
        #         # queue.append(neighbor)
        #         print("expanding ", neighbor)
        #         to_add.append(neighbor)
        #         parent_map[neighbor] = current_node
        #         # parent_map[neighbor] = current_node
        # print('\n')
        # print((to_add))
        # a = list(reversed(to_add))
        # print(to_add)
        # print(stack)
        # stack.extend(list(reversed(to_add)))
        stack.extend(to_add)
        # print(stack)
        # 构建从终点到起点的路径
    path = []
    node = end
    while node is not None:
        path.append(node)
        node = parent_map.get(node)
    
    # 反转路径以得到从起点到终点的路径
    path.reverse()
    
    # 如果终点不在路径中，说明没有找到有效路径
    if path[0] != start:
        path = []
    return visited, path

# TO DO: Implement Greedy Best-first Search.
def best_first_search(dis_map, time_map, start, end):
    """
    Greedy Best-first Search

    Args:
        time_map (dict): A map containing travel times between connected nodes (places or intersections), where every
        node is a dictionary key, and every value is an inner dictionary whose keys are the children of that node and
        values are travel times. Travel times are "null" for nodes that are not connected.
        dis_map (dict): A map containing straight-line (Euclidean) distances between every pair of nodes (places or
        intersections, connected or not), where every node is a dictionary key, and every value is an inner dictionary whose keys are the
        children of that node and values are straight-line distances.
        start (str): The name of the node from where to start traversal
        end (str): The name of the node from where to start traversal

    Returns:
        visited (list): A list of visited nodes in the order in which they were visited
        path (list): The final path found by the search algorithm
    """
    # 使用优先队列（最小堆）来根据启发函数（直线距离）选择最优节点
    import heapq
    # print("\n\n\n\n\n\n\n\n\n", dis_map, (dis_map[start][end], start), "\n\n\n\n\n\n")
    priority_queue = []
    heapq.heappush(priority_queue, (dis_map[start][end], start))
    
    # 记录已访问节点和路径
    visited = []
    parent_map = {start: None}
    
    while priority_queue:
        # 从优先队列中弹出估值最小的节点
        current_priority, current_node = heapq.heappop(priority_queue)
        print("current_node ", current_node)

        if current_node in visited:
            continue

        visited.append(current_node)

        if current_node == end:
            break

        exp_neighbors = expand.expand(current_node, time_map)

        # 遍历当前节点的邻居
        for neighbor in exp_neighbors:
            travel_time = time_map[current_node][neighbor]
            if travel_time is not None and neighbor not in visited:
                
                heapq.heappush(priority_queue, (dis_map[neighbor][end], neighbor))
                parent_map[neighbor] = current_node

        # for neighbor, travel_time in time_map[current_node].items():
        #     if travel_time is not None and neighbor not in visited:
                
        #         heapq.heappush(priority_queue, (dis_map[neighbor][end], neighbor))
        #         parent_map[neighbor] = current_node

    path = []
    node = end
    while node is not None:
        path.append(node)
        node = parent_map.get(node)
    
    path.reverse()
    
    return visited, path

# TO DO: Implement A* Search.
def a_star_search(dis_map, time_map, start, end):
    """
    A* Search

    Args:
        time_map (dict): A map containing travel times between connected nodes (places or intersections), where every
        node is a dictionary key, and every value is an inner dictionary whose keys are the children of that node and
        values are travel times. Travel times are "null" for nodes that are not connected.
        dis_map (dict): A map containing straight-line (Euclidean) distances between every pair of nodes (places or
        intersections, connected or not), where every node is a dictionary key, and every value is an inner dictionary whose keys are the
        children of that node and values are straight-line distances.
        start (str): The name of the node from where to start traversal
        end (str): The name of the node from where to start traversal

    Returns:
        visited (list): A list of visited nodes in the order in which they were visited
        path (list): The final path found by the search algorithm
    """
    import heapq
    # 优先队列：存储待扩展节点及其估计代价（f = g + h）
    priority_queue = []
    # g_cost存储从起点到当前节点的实际代价
    g_cost = {start: 0}
    # 将起点推入优先队列，优先级为启发值（起点到终点的直线距离）
    heapq.heappush(priority_queue, (dis_map[start][end], start))
    
    # 记录已访问的节点
    visited = []
    # parent_map 记录每个节点的前驱节点
    parent_map = {start: None}
    
    while priority_queue:
        _, current_node = heapq.heappop(priority_queue)
        
        if current_node in visited:
            continue
        
        visited.append(current_node)
        
        if current_node == end:
            break
        
        exp_neighbors = expand.expand(current_node, time_map)

        for neighbor in exp_neighbors:
            travel_time = time_map[current_node][neighbor]
            if travel_time is not None and neighbor not in visited:
                # 计算从起点到邻居的实际代价 g_cost
                tentative_g_cost = g_cost[current_node] + travel_time
                
                # 如果邻居节点没有被探索过，或找到更优的路径
                if neighbor not in g_cost or tentative_g_cost < g_cost[neighbor]:
                    # 更新 g_cost
                    g_cost[neighbor] = tentative_g_cost
                    # 计算优先级 f = g_cost + h_cost（直线距离）
                    f_cost = tentative_g_cost + dis_map[neighbor][end]
                    # 将邻居节点推入优先队列
                    heapq.heappush(priority_queue, (f_cost, neighbor))
                    # 更新父节点
                    parent_map[neighbor] = current_node
    
    # 构建从终点到起点的路径
    path = []
    node = end
    while node is not None:
        path.append(node)
        node = parent_map.get(node)
    
    # 反转路径得到从起点到终点的路径
    path.reverse()
    
    # 如果终点不在路径中，说明没有找到有效路径
    if path[0] != start:
        path = []
    
    return visited, path