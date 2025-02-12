from board import Board
from constants import PLAYER2_PIECE_COLOR, PLAYER1_PIECE_COLOR
from copy import deepcopy
from move_node import MoveNode


class Game:
    def __init__(self):
        """
        Initializes the game.
        """

        self._init()

    def select(self, row, col):
        """
        Selects a piece based on the given row and column. If a piece is selected and moved successfully, the move is
        processed. If no valid move is made, the selection is cleared and re-attempted.

        Args:
            row (int): The row of the selected piece
            col (int): The column of the selected piece

        Returns:
            bool: True if the selection was successful, False otherwise
        """

        if self.selected:
            result = self._process_a_move(row, col)
            if not result:
                self.selected = None
                self.select(row, col)
        piece = self.board.get_piece(row, col)
        if piece != 0 and piece.color == self.turn:
            self.selected = piece
            # self.valid_moves = self.identify_valid_moves(piece)
            moves = self.find_moves(self.board, piece)
            self.valid_moves = {destination: captures for destination, captures, _ in moves}

            return True

        return False

    def get_valid_moves(self):
        """
        Retrieves the list of valid moves for the current game state. This function accesses and returns the
        `valid_moves` attribute, which should be populated by a previous call to a selection or move-generation method (such as `select()`).

        Returns:
            list: A list of valid moves, where each move is represented by its coordinates and other relevant information
        """

        return self.valid_moves

    def get_board(self):
        """
        Retrieves the current board state.

        Returns:
            Board: The current Board state
        """

        return self.board

    def change_turn(self):
        """
        Switches the turn between players.
        """

        self.valid_moves = []
        if self.turn == PLAYER1_PIECE_COLOR:
            self.turn = PLAYER2_PIECE_COLOR
        else:
            self.turn = PLAYER1_PIECE_COLOR

    def ai_move(self, board):
        """
        Updates the current board state with a new board state after an AI move.

        Args:
            board (Board): The updated Board state after the AI move
        """

        self.board = board
        self.change_turn()

    def winner(self):
        """
        Determines the winner of the game based on the number of remaining pieces.

        Returns:
            tuple: The RGB color of the pieces to evaluate, formatted as a tuple (e.g., (255, 240, 125)).
        """

        if len(self.board.get_all_pieces(PLAYER1_PIECE_COLOR)) <= 0:
            return PLAYER2_PIECE_COLOR
        elif len(self.board.get_all_pieces(PLAYER2_PIECE_COLOR)) <= 0:
            return PLAYER1_PIECE_COLOR
        return None

    def reset(self):
        """
        Resets the game state to its initial values.
        """

        self._init()

    def generate_all_moves(self, board, color):
        """
        Generates all possible moves for a given color.

        Args:
            board (Board): The current board state
            color (tuple): The RGB color of the pieces to evaluate, formatted as a tuple (e.g., (255, 240, 125)).

        Returns:
            list: A list of board states resulting from all possible moves for the given color
        """

        moves = []

        for piece in board.get_all_pieces(color):
            valid_moves = self.find_moves(board, piece)
            for move, captured_pieces, king_hopeful in valid_moves:
                temp_board = deepcopy(board)
                temp_piece = temp_board.get_piece(piece.row, piece.col)
                new_board = self.simulate_move(temp_piece, move, temp_board, captured_pieces)
                moves.append(new_board)

        return moves

    def find_moves(self, board, piece):
        """
        Finds all possible moves for the specified piece on the board, including multi-hop captures. This function calculates:

        - The total number of possible moves for the specified piece, accounting for all hops and captures
        - The total number of captures achievable through multi-hop sequences
        - The total number of king hopefuls via these moves (See Notes for evaluate()).

        Args:
            board (Board): The current board state
            piece (Piece): The piece for which possible moves are being evaluated.

        Returns:
            tuple: A tuple containing:
                - (int) Total number of valid moves
                - (int) Total number of possible captures
                - (int) Total number of king hopefuls.
        """

        all_moves = []

        if piece.color == PLAYER1_PIECE_COLOR or piece.king:
            moves_tree = self.traverse(board, piece, piece.row, piece.col, 'front', 'left')
            moves = self.dfs_collect_move_destinations(moves_tree)
            all_moves.extend(moves)

            moves_tree = self.traverse(board, piece, piece.row, piece.col, 'front', 'right')
            moves = self.dfs_collect_move_destinations(moves_tree)
            all_moves.extend(moves)

        if piece.color == PLAYER2_PIECE_COLOR or piece.king:
            moves_tree = self.traverse(board, piece, piece.row, piece.col, 'back', 'left')
            moves = self.dfs_collect_move_destinations(moves_tree)
            all_moves.extend(moves)

            moves_tree = self.traverse(board, piece, piece.row, piece.col, 'back', 'right')
            moves = self.dfs_collect_move_destinations(moves_tree)
            all_moves.extend(moves)

        return all_moves

    def traverse(self, board, piece, curr_row, curr_col, row_shift, col_shift, skipped=None, captured=None):
        """
        Traverses the board to find moves (including multi-hop moves) for the specified piece by recursively evaluating
        potential target squares based on row and column shifts. This function identifies if a move is valid, captures
        any opposing pieces in its path, and assesses if the move can promote the piece to a king.

        Args:
            board (Board): The current board state
            piece (Piece): The piece for which to calculate potential moves
            curr_row (int): The current row position of the piece
            curr_col (int): The current column position of the piece
            row_shift (str): 'front' or 'back' row shift for move calculation
            col_shift (str): 'left' or 'right' column shift for move calculation
            skipped (Piece, optional): A piece skipped over in a potential capture, if any
            captured (Piece, optional): A list of pieces captured in the current move sequence.

        Returns:
            MoveNode: A root MoveNode representing the move tree from the current position, including all valid moves,
            potential captures, and king hopefuls (See Notes for evaluate())
        """

        steps = {
            'left': -1,
            'right': 1,
            'front': -1,
            'back': 1,
            'left_hop': -2,
            'right_hop': 2,
            'front_hop': -2,
            'back_hop': 2
        }

        # Root of the moves tree at this level
        moves = MoveNode((curr_row, curr_col), None, None)

        target_row = curr_row + steps[row_shift]
        target_col = curr_col + steps[col_shift]

        if 0 <= target_row <= 7 and 0 <= target_col <= 7:
            target_piece = board.get_piece(target_row, target_col)
            if target_piece == 0:
                if skipped is None and not captured:
                    capture = None
                    king_hopeful = self.check_king_hopeful(piece, target_row, curr_row, curr_col)

                    new_node = MoveNode((target_row, target_col), capture, king_hopeful)
                    moves.add_child(new_node)
                elif skipped:
                    capture = skipped
                    king_hopeful = self.check_king_hopeful(piece, target_row, curr_row, curr_col)

                    new_node = MoveNode((target_row, target_col), capture, king_hopeful)
                    moves.add_child(new_node)

                    curr_row = target_row
                    curr_col = target_col

                    new_board = deepcopy(board)
                    new_piece = deepcopy(piece)
                    new_board = self.simulate_move(new_piece, (curr_row, curr_col), new_board, [skipped])

                    directions = [
                        ('front' if 'front' in row_shift else 'back', 'left'),
                        ('front' if 'front' in row_shift else 'back', 'right')
                    ]

                    if piece.king:
                        directions.extend([('back' if 'front' in row_shift else 'front', 'left' if 'left' in col_shift else 'right')])

                    for row_dir, col_dir in directions:
                        new_moves = self.traverse(new_board, new_piece, curr_row, curr_col, row_dir, col_dir, captured=capture)
                        new_node.children.extend(new_moves.children)

            else:
                if skipped is None and target_piece.color != piece.color:
                    skipped = target_piece
                    new_moves = self.traverse(board, piece, curr_row, curr_col,
                                                        'front_hop' if 'front' in row_shift else 'back_hop',
                                                        'left_hop' if 'left' in col_shift else 'right_hop',
                                              skipped=skipped)
                    moves.children.extend(new_moves.children)

        return moves

    def check_king_hopeful(self, piece, target_row, curr_row, curr_col):
        """
        Determines if a piece is eligible for promotion to king status based on its color and target row. If the piece
        reaches the opponent's baseline row and is not yet a king, the current position coordinates are returned as a
        marker for king promotion.

        Args:
            piece (Piece): The piece to check for king promotion eligibility
            target_row (int): The row the piece is moving to
            curr_row (int): The current row position of the piece
            curr_col (int): The current column position of the piece.

        Returns:
            tuple or None: Returns the current position (curr_row, curr_col) of the piece if it is eligible for promotion;
            otherwise, returns None.
        """

        if not piece.king and piece.color == PLAYER1_PIECE_COLOR and target_row == 0:
            return curr_row, curr_col
        if not piece.king and piece.color == PLAYER2_PIECE_COLOR and target_row == 7:
            return curr_row, curr_col

        return None

    def simulate_move(self, piece, move, board, skip):
        """
        Simulates a move by updating the board with the given move and removing any skipped pieces.

        Args:
            piece (Piece): The piece being moved
            move (tuple): A tuple indicating the target row and column to move the piece to
            board (Board): The current board state
            skip (list): A list of pieces that are captured by the move

        Returns:
            Board: The updated board state after the move
        """

        board.move_piece(piece, move[0], move[1])
        if skip:
            board.remove_pieces(skip)

        return board

    def dfs_collect_move_destinations(self, move_tree, current_path=None, captured_pieces=None, king_hopeful=False, destinations=None, is_root=True):
        """
        Performs a depth-first search (DFS) on a move tree to collect the final destinations for each possible move path,
        including any captured pieces along each path and potential king promotions. This function collects only the end points
        of paths, rather than full paths, along with any associated captured pieces and king hopeful status.

        Args:
            move_tree (Node): The root node of the move tree representing possible moves for a piece
            current_path (list, optional): A list tracking the sequence of moves to the current node. Defaults to None, where it initializes as an empty list
            captured_pieces (list, optional): A list of pieces captured along the current path. Defaults to None, initializing as an empty list
            king_hopeful (bool, optional): A flag indicating if any move in the path could result in a king. Defaults to False
            destinations (list, optional): A list to store all valid destination moves. Each entry in the list includes the final destination, captured pieces, and king hopeful status. Defaults to None, initializing as an empty list.
            is_root (bool, optional): A flag to indicate if the current node is the root of the DFS traversal. Defaults to True.

        Returns:
            list: A list of move destinations where each entry is a list containing:
                  - final destination coordinate,
                  - list of captured pieces along the path, and
                  - a boolean indicating if a king can be achieved on this path.

        Example:
            dfs_collect_move_destinations(root_node)  # Call on the root node of a move tree.
        """

        # Initialize path, captured pieces list, and destinations list if not provided
        if current_path is None:
            current_path = []
        if captured_pieces is None:
            captured_pieces = []
        if destinations is None:
            destinations = []

        # Always include the root in the path (the first move of the piece)
        if is_root or move_tree.move is not None:  # Include root move
            current_path.append(move_tree.move)

        # If there's a capture at this node, add it to captured_pieces
        if move_tree.capture is not None:
            captured_pieces.append(move_tree.capture)

        # If this node represents a king hopeful situation, update the flag
        if move_tree.king_hopeful is not None:
            king_hopeful = True

        # If we are at a leaf node (no more children), save only the final destination and its associated data
        if not move_tree.children and len(current_path) > 1:  # Make sure there's at least one move
            final_destination = current_path[-1]  # Only the final destination
            destinations.append([final_destination, captured_pieces[:],
                                 king_hopeful])  # Save destination, captures, and king hopeful status
            # print(f"Leaf reached: Destination: {final_destination}, Captures: {captured_pieces}, King Hopeful: {king_hopeful}")
        else:
            # Recur for each child node (DFS)
            for child in move_tree.children:
                self.dfs_collect_move_destinations(child, current_path[:], captured_pieces[:], king_hopeful,
                                                   destinations, is_root=False)

        return destinations

    def _init(self):
        """
        Resets the game state to its initial values. This is separated out from __init__() to let this resetting
        functionality be available when the game needs to be reset after a round of game is over.
        """

        self.selected = None
        self.board = Board()
        self.turn = PLAYER1_PIECE_COLOR
        self.valid_moves = {}

    def _process_a_move(self, row, col):
        """
        Processes the move for the selected piece to the specified row and column. Updates the board and handles the
        capturing of the opponent's pieces if necessary.

        Args:
            row (int): The target row position for the move
            col (int): The target column position for the move.

        Returns:
            bool: True if the move was successful, False otherwise
        """

        piece = self.board.get_piece(row, col)
        if self.selected and piece == 0 and (row, col) in self.valid_moves:
            self.board.move_piece(self.selected, row, col)
            skipped = self.valid_moves[(row, col)]
            if skipped:
                self.board.remove_pieces(skipped)
            self.change_turn()
        else:
            return False

        return True
