from util import load_grid_data_json, visualize_traversal
import expand
import json
import chowrider_code as sc
import time
import unittest

class ChicagoMapTestCase(unittest.TestCase):
    MAX_EXPAND_LIMIT = 1000  # Set a limit to prevent excessive expansions

    @classmethod
    def setUpClass(cls):
        """
        Set up the map and load expected results.
        This method is called once before any tests are run.
        """

        cls.grid_data = load_grid_data_json('map_chicago.json')

        # Extract necessary data from grid_data
        cls.normalized_intersections = cls.grid_data['normalized_intersections']
        cls.time_map = cls.grid_data['time_map']
        cls.edge_list = cls.grid_data['edge_list']
        cls.dis_map = cls.grid_data['dis_map']

        # Define Start and End Nodes for Test Cases (1 to 7)
        cls.test_cases = {
            1: {
                'start': "Van Buren Street / McClurg Court",
                'end': "Van Buren Street / Western Avenue",
            },
            2: {
                'start': "Irving Park Road / Sheffield Avenue",
                'end': "Van Buren Street / Sheffield Avenue",
            },
            3: {
                'start': "North Avenue / Racine Avenue",
                'end': "Waveland Avenue / LaSalle Street",
            },
            4: {
                'start': "Fulton Market / Dearborn Street",
                'end': "Waveland Avenue / Wells Street",
            },
            5: {
                'start': "Hubbard Street / Western Avenue",
                'end': "North Avenue / Wolcott Avenue",
            },
            6: {
                'start': "Van Buren Street / McClurg Court",
                'end': "Irving Park Road / Western Avenue",
            },
            7: {
                'start': "North Avenue / McClurg Court",
                'end': "North Avenue / Western Avenue",
            }
        }

        # Load expected results from JSON
        with open('expected_results_chicago.json', 'r') as f:
            cls.expected_results = json.load(f)

    def run_algorithm_test(self, algorithm, test_case_num, visualize=False):
        """
        Helper method to run a specific algorithm and perform assertions.

        Args:
            algorithm (str): The algorithm to run ('bfs', 'dfs', 'a_star', 'best_fs').
            test_case_num (int): The test case number (1 to 7).
            visualize (bool): Whether to visualize the traversal.
        """

        # Retrieve test case details
        test_case = self.test_cases[test_case_num]
        start = test_case['start']
        end = test_case['end']

        # Reset expand_count before running the algorithm
        expand.expand_count = 0

        # Start timing the algorithm execution
        start_time = time.time()

        try:
            # Run the selected algorithm
            if algorithm == 'bfs':
                visited, path = sc.breadth_first_search(self.time_map, start, end)
            elif algorithm == 'dfs':
                visited, path = sc.depth_first_search(self.time_map, start, end)
            elif algorithm == 'a_star':
                visited, path = sc.a_star_search(self.dis_map, self.time_map, start, end)
            elif algorithm == 'best_fs':
                visited, path = sc.best_first_search(self.dis_map, self.time_map, start, end)
            else:
                self.fail(f"Unknown algorithm: {algorithm}")

            # End timing
            end_time = time.time()
            execution_time = end_time - start_time

            # Print path, expand_count, and execution time for debugging purposes
            print(f"\nAlgorithm: {algorithm.upper()}, Test Case: {test_case_num}")
            print(f"Start: {start}")
            print(f"End: {end}")
            print(f"Path Found: {path}")
            # expand.expand_count = len(visited) - 1
            print(f"Expand Count: {expand.expand_count}")
            print(f"Execution Time: {execution_time:.4f} seconds")

            # If the expansion limit is exceeded, print a warning
            if expand.expand_count > self.MAX_EXPAND_LIMIT:
                print(f"WARNING: Expand count exceeded limit ({self.MAX_EXPAND_LIMIT}) for Test Case {test_case_num} with {algorithm.upper()}.")

        except Exception as e:
            print(f"Error occurred during {algorithm.upper()} for Test Case {test_case_num}: {e}")
            return

        # Visualization (optional)
        if visualize:
            visualize_traversal(
                visited,
                self.time_map,
                self.normalized_intersections,
                self.edge_list,
                f"{algorithm.upper()} Traversal - Test Case {test_case_num}",
                path=path,
                start_node=start,
                end_node=end
            )

        # Retrieve expected results
        test_case_key = f"test_case_{test_case_num}"
        if test_case_key not in self.expected_results:
            self.fail(f"No expected results found for {test_case_key}. Ensure 'expected_results_chicago.json' is correctly populated.")

        if algorithm not in self.expected_results[test_case_key]:
            self.fail(f"No expected results for {algorithm.upper()} in {test_case_key}. Ensure 'expected_results_chicago.json' includes all algorithms.")

        expected = self.expected_results[test_case_key][algorithm]
        expected_path = expected['path']
        expected_count = expected['expand_count']
        
        # Assertions
        self.maxDiff = None  # Allows full diff output for assertion failures
        self.assertIsNotNone(path, f"{algorithm.upper()} did not find a path in Test Case {test_case_num}.")
        self.assertEqual(path, expected_path, f"{algorithm.upper()} did not find the expected path in Test Case {test_case_num}.")
        self.assertEqual(expand.expand_count, expected_count, f"{algorithm.upper()} did not have the expected expand count in Test Case {test_case_num}.")

    # ------------------------------------
    # Test Cases for Each Search Algorithm
    # ------------------------------------

    def test_case_1_bfs(self):
        """Test Case 1 - BFS"""

        self.run_algorithm_test('bfs', 1, visualize=False)

    def test_case_1_dfs(self):
        """Test Case 1 - DFS"""

        self.run_algorithm_test('dfs', 1, visualize=False)

    def test_case_1_best_fs(self):
        """Test Case 1 - Best-first Search"""

        self.run_algorithm_test('best_fs', 1, visualize=False)

    def test_case_1_a_star(self):
        """Test Case 1 - A*"""

        self.run_algorithm_test('a_star', 1, visualize=False)

    def test_case_2_bfs(self):
        """Test Case 2 - BFS"""

        self.run_algorithm_test('bfs', 2, visualize=False)

    def test_case_2_dfs(self):
        """Test Case 2 - DFS"""

        self.run_algorithm_test('dfs', 2, visualize=False)

    def test_case_2_best_fs(self):
        """Test Case 2 - Best-first Search"""

        self.run_algorithm_test('best_fs', 2, visualize=False)

    def test_case_2_a_star(self):
        """Test Case 2 - A*"""

        self.run_algorithm_test('a_star', 2, visualize=False)

    def test_case_3_bfs(self):
        """Test Case 3 - BFS"""

        self.run_algorithm_test('bfs', 3, visualize=False)

    def test_case_3_dfs(self):
        """Test Case 3 - DFS"""

        self.run_algorithm_test('dfs', 3, visualize=False)

    def test_case_3_best_fs(self):
        """Test Case 3 - Best-first Search"""

        self.run_algorithm_test('best_fs', 3, visualize=False)

    def test_case_3_a_star(self):
        """Test Case 3 - A*"""

        self.run_algorithm_test('a_star', 3, visualize=False)

    def test_case_4_bfs(self):
        """Test Case 4 - BFS"""

        self.run_algorithm_test('bfs', 4, visualize=False)

    def test_case_4_dfs(self):
        """Test Case 4 - DFS"""

        self.run_algorithm_test('dfs', 4, visualize=False)

    def test_case_4_best_fs(self):
        """Test Case 4 - Best-first Search"""

        self.run_algorithm_test('best_fs', 4, visualize=False)

    def test_case_4_a_star(self):
        """Test Case 4 - A*"""

        self.run_algorithm_test('a_star', 4, visualize=False)

    def test_case_5_bfs(self):
        """Test Case 5 - BFS"""

        self.run_algorithm_test('bfs', 5, visualize=False)

    def test_case_5_dfs(self):
        """Test Case 5 - DFS"""

        self.run_algorithm_test('dfs', 5, visualize=False)

    def test_case_5_best_fs(self):
        """Test Case 5 - Best-first Search"""

        self.run_algorithm_test('best_fs', 5, visualize=False)

    def test_case_5_a_star(self):
        """Test Case 5 - A*"""

        self.run_algorithm_test('a_star', 5, visualize=False)

    def test_case_6_bfs(self):
        """Test Case 6 - BFS"""

        self.run_algorithm_test('bfs', 6, visualize=False)

    def test_case_6_dfs(self):
        """Test Case 6 - DFS"""

        self.run_algorithm_test('dfs', 6, visualize=False)

    def test_case_6_best_fs(self):
        """Test Case 6 - Best-first Search"""

        self.run_algorithm_test('best_fs', 6, visualize=False)

    def test_case_6_a_star(self):
        """Test Case 6 - A*"""

        self.run_algorithm_test('a_star', 6, visualize=False)

    def test_case_7_bfs(self):
        """Test Case 7 - BFS"""

        self.run_algorithm_test('bfs', 7, visualize=False)

    def test_case_7_dfs(self):
        """Test Case 7 - DFS"""

        self.run_algorithm_test('dfs', 7, visualize=False)

    def test_case_7_best_fs(self):
        """Test Case 7 - Best-first Search"""

        self.run_algorithm_test('best_fs', 7, visualize=False)

    def test_case_7_a_star(self):
        """Test Case 7 - A*"""
        
        self.run_algorithm_test('a_star', 7, visualize=False)

if __name__ == "__main__":
    unittest.main()
