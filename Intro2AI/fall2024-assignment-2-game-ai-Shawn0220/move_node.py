class MoveNode:
    """
    Represents a node in a move tree for a board game, capturing the details of a move, any captures made, and potential king promotions.

    Attributes:
        move (tuple): The coordinates of the move as a tuple (e.g., (row, col))
        capture (object): The piece captured in this move, if any; None otherwise
        king_hopeful (bool): Indicates if the move results in a promotion to king status
        children (list): A list of child MoveNode objects representing subsequent moves.

    Methods:
        add_child(child_node):
            Adds a child MoveNode to the list of children, allowing for the construction of a move tree.
    """

    def __init__(self, move, capture, king_hopeful):
        self.move = move  # A tuple representing the move coordinates
        self.capture = capture  # The piece captured in this move (if any)
        self.king_hopeful = king_hopeful  # If this move results in a king promotion
        self.children = []  # List of child MoveNode objects

    def add_child(self, child_node):
        """
        Adds a child MoveNode to the list of children, allowing for the construction of a move tree.

        Args:
            child_node (MoveNode): The child node to be added as a subsequent move
        """
        self.children.append(child_node)
