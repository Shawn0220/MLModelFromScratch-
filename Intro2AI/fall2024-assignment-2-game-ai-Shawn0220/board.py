from constants import PLAYER1_PIECE_COLOR, PLAYER2_PIECE_COLOR, ROWS, COLS
from piece import Piece

class Board:
    def __init__(self, board_config=None):
        """
        Initializes the game board with what is to be used as a 2-D array. Then it creates the board.
        """

        self.board = []
        if not board_config:
            self.create()
        else:
            self.create_specific(board_config)

    def create(self):
        """
        Initializes the board with pieces in their starting positions.
        """

        for row in range(ROWS):
            self.board.append([])
            for col in range(COLS):
                if col % 2 == ((row + 1) % 2):
                    if row < 3:
                        self.board[row].append(Piece(row, col, PLAYER2_PIECE_COLOR))
                    elif row > 4:
                        self.board[row].append(Piece(row, col, PLAYER1_PIECE_COLOR))
                    else:
                        self.board[row].append(0)
                else:

                    self.board[row].append(0)

    def create_specific(self, board_config):
        """
        Initializes the board with pieces in specific positions as specified by a given board configuration.

        Args:
            board_config: A 2-D array where 1 indicates Player 1 piece, 2 indicates Player 2 piece, and 0 indicates an empty square
        """

        for row in range(len(board_config)):
            self.board.append([])
            for col in range(len(board_config[row])):
                if board_config[row][col] == 1:
                    self.board[row].append(Piece(row, col, PLAYER1_PIECE_COLOR))
                elif board_config[row][col] == 11:
                    piece = Piece(row, col, PLAYER1_PIECE_COLOR)
                    piece.make_king()
                    self.board[row].append(piece)
                elif board_config[row][col] == 2:
                    self.board[row].append(Piece(row, col, PLAYER2_PIECE_COLOR))
                elif board_config[row][col] == 22:
                    piece = Piece(row, col, PLAYER2_PIECE_COLOR)
                    piece.make_king()
                    self.board[row].append(piece)
                else:
                    self.board[row].append(0)

    def get_piece(self, row, col):
        """
        Retrieves the piece at a given position on the board.

        Args:
            row (int): The row index of the piece
            col (int): The column index of the piece

        Returns:
            Piece: The piece at the given position or 0 if no piece is present
        """

        return self.board[row][col]

    def get_all_pieces(self, color):
        """
        Retrieves all pieces of a given color.

        Args:
            color (tuple): The RGB color of the pieces to retrieve, formatted as a tuple (e.g., (255, 240, 125)).

        Returns:
            list: A list of Piece objects of the given color
        """

        pieces = []
        for row in self.board:
            for piece in row:
                if piece != 0 and piece.color == color:
                    pieces.append(piece)
        return pieces

    def move_piece(self, piece, row, col):
        """
        Moves a piece to a new position on the board and converts it to a king if applicable.

        Args:
            piece (Piece): The piece to move
            row (int): The target row
            col (int): The target column
        """

        self.board[piece.row][piece.col], self.board[row][col] = self.board[row][col], self.board[piece.row][piece.col]
        piece.move(row, col)

        if row == ROWS - 1 or row == 0:
            piece.make_king()

    def remove_pieces(self, pieces):
        """
        Removes the given pieces from the board.

        Args:
            pieces (list): A list of Piece objects to remove from the board
        """

        for piece in pieces:
            self.board[piece.row][piece.col] = 0

    def to_board_config(self):
        """
        Converts the current board state to a configuration array format. 0 is an empty square, 1 is Player 1's piece,
        2 is Player 2's piece, 11 is Player 1's king, and 22 is Player 2's king.

        Returns:
            list: A 2-D array representing the board configuration
        """
        board_config = []
        for row in range(8):
            config_row = []
            for col in range(8):
                piece = self.get_piece(row, col)
                if piece == 0:
                    config_row.append(0)
                elif piece.color == PLAYER1_PIECE_COLOR:
                    if piece.king:
                        config_row.append(11)
                    else:
                        config_row.append(1)
                elif piece.color == PLAYER2_PIECE_COLOR:
                    if piece.king:
                        config_row.append(22)
                    else:
                        config_row.append(2)
            board_config.append(config_row)

        return board_config

    def print_board_config(self, board_config):
        """
        Prints a visual representation of the board configuration array.

        Args:
            board_config (list): A 2-D array representing the board configuration, where:
                - 0 represents an empty square,
                - 1 represents Player 1's piece,
                - 2 represents Player 2's piece,
                - 11 represents Player 1's king, and
                - 22 represents Player 2's king.

        This function iterates over each row of `board_config`, printing it as a list, providing a clear overview of
        the board layout.
        """

        print()
        for i, row in enumerate(board_config):
            if i < len(board_config) - 1:
                print(str(row) + ",")
            else:
                print(str(row))
        print()
