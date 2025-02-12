# Assignment 1 - Search

## The Premise

You founded a startup called ChowRider, which specializes in speedy delivery of fast food in Chicago by motorcycle. For the delivery component of your startup, you hired an army of diehard Chicagoans who know the Windy City inside out and use their intimate knowledge of city streets to efficiently make deliveries. Lately, though, you've hired some new folks who are mostly from out of town, and it seems they're not dealing so well with the numerous one-way streets that characterize Chicago. Having recently graduated from Northwestern University, you remember the cool search algorithms you learned in the Intro to AI course, and you realize there's a big opportunity to enable your bikers with some AI-powered navigation to find their way through the city. For now, you're not keen on finding the shortest path from point A to point B; rather, you want to compare some search algorithms by running them against some paths to identify which algorithm you should use in the app you're building for your dynamic delivery team. Specifically, these are the algorithms you've decided to experiment with:

1. Breadth-first Search (BFS)
2. Depth-first Search (DFS)
3. Greedy Best-first Search (GBFS)
4. A*

You've used some of your Series A funding to have the company TestFriendForever (TFF) generate some tests for you that you plan on using to test your implementations of the aforementioned algorithms. Most of the tests are based on a street map of a large swath of Chicago. They've also authored some tests based on parts of your campus neighborhood in Evanston. TFF recommends that you use the Evanston-based tests first, as they're simpler and can help you to debug your code, after which you can test your code with the Chicago-based tests. To ensure robustness of your code, you have also entrusted TFF to certify your code by scoring it based not only on the Evanston- and Chicago-based tests but also on some secret tests that they will not reveal to you, the secret tests serving the sole purpose of ensuring that your code has genuinely passed the other tests. If your code passes the Evanston-based tests but not the Chicago-based ones, you likely have a code design that luckily passes the former tests but not the latter. Further debugging the code should help it to pass the Chicago-based tests, at which point you should be confident that it will pass the secret tests too and earn TFF certification.

## The Task


TFF has kindly supplied you with a codebase as a framework. In order to obtain TFF certification, all you have to do is to implement the four (4) unimplemented functions in the file `chowrider_code.py` that are preceded by the string `"TO DO"` such that they pass the tests below. The four functions correspond to the four search algorithms that you wish to compare. The tests they need to pass are:

1. All four (4) search algorithm variants of each of the four (4) tests cases in the file `test_evanston.py`. Those 16 tests are based on the Evanston-based map in `map_evanston.json`.
2. All four (4) search algorithm variants of each of the seven (7) tests cases in the file `test_chicago.py`. Those 28 tests are based on the Chicago-based map in `map_chicago.json`.

## The Details

TFF has provided you with some details pertaining to the maps and tests:

1. Note that each of the two maps actually has two map representations within it: a **time map**, indicated by the `time_map` component in the JSON file and a **distance map**, indicated by the `dis_map` component in the JSON file. The *time map* has the travel time between every pair of adjacent nodes in the map (places in Evanston or intersections in Chicago) while the *distance map* has the linear (Euclidean) distance ("as the crow flies") between every pair of (not necessarily adjacent) nodes in the map.
2. For BFS and DFS, you will use the **time map**, even though travel times are not relevant for them; node adjacency is all that is needed.
3. For A*, you will use both the **time map** and the **distance map**. You will use the travel times from the *time map* to calculate the distance traveled from the `start` node to the current node while you will use the distances from the *distance map* to calculate the `heuristic` -- the distance remaining from the current node to the `end` node.
4. For GBFS (referred to in the code as simply Best-first Search or `best_fs()`), you will use both the **time map** and the **distance map**, and you will use the same heuristic as with A*, keeping your priority-queue sorted from the node with the smallest heuristic value to that with the largest, i.e., from the node with the least distance remaining to the end to the node with the largest distance remaining to the end.
5. To expand nodes during any of the searches, use the `expand()` function provided in the file `expand.py`. Do not write your own.
6. When pushing expanded nodes onto your queue or stack, push them in the same order in which the `expand()` function returns/yields them. This means BFS will traverse the search tree **from left to right** while DFS will traverse it **from right to left**. The tests assume those directional traversals, so traversing the search trees in any other way will prevent the code from passing the tests.
7. With A*, if two or more nodes have the same `f(n)`, you should use `h(n)` to break the tie, i.e., pick the node that has the smallest `h(n)`. If two or more nodes have the same `h(n)`, then pick the node that entered the open list first when returned/yielded by the `expand()` function in `expand.py`. *If you use a priority-queue, though, then the order of the nodes returned/yielded by `expand()` will not be preserved due to sorting, so you may need to use a separate variable to keep track of the order.*
8. Remember that the goal is not to find the shortest path (though A* will yield it anyway) but to find the path that the search algorithms yield in their standard ("vanilla") implementation. As such, how long your search algorithm takes to complete the search is not something you should worry about, unless it takes an unreasonable amount of time (e.g., a whole minute for just one search), leading TFF to believe your code has encountered an infinite loop or something. As long as your code exists in a reasonable amount of time, you don't need to worry about optimization.

## The Visualizations

In order to aid your comparison of the four search algorithms, TFF has also provided an argument in the Chicago-based test functions that lets you visualize the searches. **TFF wants to quiz you on your understanding and appreciation of the different nature of each search algorithm, so they urge you to utilize that function argument to enable visualization so you can see how distinctly characteristic the search algorithms are.** You will be asked to answer a question based on your visualizations. The function argument is called `visualize` and is set to `False` by default. For example, notice the `visualize=False` argument on line 130 of the file `test_chicago.py`. Note that the `start` and `end` nodes are not the same for all test cases, though they are same for all four search algorithms associated with the same test case, so visualizing the four search algorithms one by one for the same test case is recommended. Notice, also, the line right beneath the string `"EDITABLE 1"` in the file `util.py`. The function called on that line allows you to set the time delay between steps in the search visualization. Feel free to adjust the number of seconds (default is `0.01`) per your preference. Similarly, the line right beneath the string `"EDITABLE 2"` allows you to toggle the visualization window between full screen and non-full screen (default).

## The Rules

1. You must implement **all four (4)** unimplemented functions in `chowrider_code.py`.
2. You must not edit any file other than `chowrider_code.py`, unless you decide to adjust the step time in `util.py` for visualization as mentioned earlier.
3. Each function must return a path from the `start` node to the `end` node, inclusive, which will be checked by the tests. The path must be a list of strings, each string being the name of a node. For example, if the `start` and `end` nodes are `A` and `H`, and if the path found goes through `B`, `D`, and `E`, then your code must return the list `[A, B, D, E, H]`.
4. The tests will also check for the number of nodes expanded during search, which will be fetched from the global variable `expand_count` in the file `expand.py`.
5. You should feel invited to use Python modules for your data structures, but you need to implement BFS, DFS, and A\* yourself. You must not use functions or packages that implement them.
