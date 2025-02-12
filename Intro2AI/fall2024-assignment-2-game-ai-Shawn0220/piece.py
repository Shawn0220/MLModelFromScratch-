class Piece:

    def __init__(self, row, col, color):
        """
        Initializes a piece with its row- and column-based board position and color.

        Args:
            row (int): The row index of the piece
            col (int): The column index of the piece
            color (tuple): The color of the piece. It is a tuple of RGB int values, e.g., (255, 240, 125)
        """

        self.row = row
        self.col = col
        self.color = color
        self.king = False

    def __repr__(self):
        """
        Provides a string representation of the piece using its color.

        Returns:
            str: The color of the piece as a string
        """

        return str(self.color)

    def move(self, row, col):
        """
        Updates the board position of the piece upon a move.

        Args:
            row (int): The new row index of the piece
            col (int): The new column index of the piece
        """

        self.row = row
        self.col = col

    def make_king(self):
        """
        Promotes the piece to a king by setting the king attribute to True.
        """

        self.king = True